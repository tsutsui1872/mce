import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import stats
from scipy.optimize import curve_fit
from .forcing import RfAll

class InputSeries:
    def __init__(self, **kw):
        """Create an input data utility instance
        from pairs of key and dataframe given by keyword arguments
        """
        self.df = {}
        self.interp = {}

        kw_interp = {'bounds_error': False, 'fill_value': 'extrapolate'}

        for k, df in kw.items():
            self.df[k] = df
            self.interp[k] = interp1d(df.index, df.values.T, **kw_interp)

    def _mod_header(self, df):
        """Ensure a given dataframe with single-level column indexes
        and to be squeezed if a single time point is referenced

        Parameters
        ----------
        df
            Input dataframe

        Returns
        -------
            Modified datafarame
        """
        if df.columns.nlevels == 2:
            df = df.droplevel(1, axis=1)

        df = df.squeeze()
        if df.ndim == 1:
            df = df.rename(None)

        return df

    def get_data(self, kind, *args):
        """Return one of the input dataframe at all or specified time points

        Parameters
        ----------
        kind
            'emis', 'conc', or 'erf' is assumed

        Returns
        -------
            Selected dataframe
        """
        if len(args) == 0:
            df = self.df[kind]
        else:
            df = self.df[kind].loc[np.hstack(args)]

        return self._mod_header(df)

    def interp_eval(self, kind, *args):
        """Return an interpolated dataframe at given time points

        Parameters
        ----------
        kind
            'emis', 'conc', or 'erf' is assumed

        Returns
        -------
            Interpolated dataframe
        """
        x = np.hstack(args)
        df = pd.DataFrame(
            self.interp[kind](x).T, index=x, columns=self.df[kind].columns,
        )

        return self._mod_header(df)


class RfAllAR6(RfAll):
    def __init__(self, dfin_ref, din, *args, **kw):
        """Create a forcing instance for full-emissions runs

        Parameters
        ----------
        din
            Input data utility instance
        dfin_ref
            Input data in pre-industrial and reference years
        """
        super().__init__(*args, **kw)

        self.SAMPLES = 100000
        self.NINETY_TO_ONESIGMA = stats.norm.ppf(0.95)

        din_ref = {
            k: v.droplevel(0).drop('unit', axis=1).rename(columns=int).T
            for k, v in dfin_ref.groupby(level=0)
        }
        din_ref['conc']['ODS'] = self.eesc_total(din_ref['conc'])

        self.din_ref = din_ref
        self.din = din

        year_pi = 1750
        self.year_pi = year_pi

        # Adjust a parameter for the AR6 CO2 forcing scheme
        self.parms_ar6_ghg.update(C0_1750=din_ref['conc'].loc[year_pi, 'CO2'])

        # Set radiative efficiency parameters for halogenated species
        map_adj = {
            k[:-4]: v for k, v in self.parms_ar6_ghg().items()
            if k.endswith('_adj')
        }
        self.ghg__hc_eff = pd.Series({
            gas: (
                self.ghgs[gas].efficiency
                * 1e-3  # ppt assumed
                * (1. + map_adj.get(gas, 0.))
            )
            for gas in din_ref['conc']
            if gas not in ['CO2', 'CH4', 'N2O', 'ODS']
        })

        # Set a forcing factor for stratospheric water vapor
        d1 = din_ref['conc'].loc[2019]
        erf1 = self.c2erf_ar6('CH4', d1['CH4'], cn2o=d1['N2O'])
        self.h2o_strat__factor = 0.05 / erf1

        # Set a forcing factor for land use
        self.land_use__pi = din_ref['emis'].loc[year_pi, 'CO2 AFOLU']
        self.land_use__factor = -0.20 / (
            din_ref['emis_co2__cumsum'].loc[2019, 'CO2 AFOLU']
            - self.land_use__pi
        )

        # Set a forcing factor for light-absorbing particles on snow and ice
        self.bc_on_snow__factor = 0.08 / (
            din_ref['emis'].loc[2019, 'BC']
            - din_ref['emis'].loc[year_pi, 'BC']
        )

        # Species associated with aerosols and ozone forcers
        self.ari__species = ['BC', 'OC', 'SO2', 'NOx', 'NMVOC', 'NH3']
        # self.ari__gases = ['CH4', 'N2O', 'EESC']
        self.ari__gases = ['CH4', 'N2O', 'ODS']
        self.aci__species = ['SO2', 'BC', 'OC']
        self.o3__species = ['CH4', 'N2O', 'ODS', 'CO', 'NMVOC', 'NOx']

        # Coefficients to be set separately
        self.ari__radeff = None
        self.ari__scale = None
        self.o3__coeff = None

    def eesc_total(self, df_conc_ghg):
        """Calculate total equivalent effective stratospheric chlorine (EESC)

        Parameters
        ----------
        df_conc_ghg
            ODS concentrations in DataFrame or Series

        Returns
        -------
            EESC in Series or scalar
        """
        ods = self.ods
        df = df_conc_ghg
        df = df.reindex(list(ods), axis=df.ndim-1).dropna(axis=df.ndim-1)

        cl_atoms = pd.Series({k: ods[k].cl_atoms for k, _ in df.items()})
        br_atoms = pd.Series({k: ods[k].br_atoms for k, _ in df.items()})
        fracrel = pd.Series({k: ods[k].fracrel for k, _ in df.items()})
        br_cl_ratio = 45
        df = df.mul((cl_atoms + br_atoms * br_cl_ratio) * fracrel)

        return df.sum(axis=df.ndim-1)

    def init__ari(self, ari_emitted, ari_emitted_std):
        """Set radiative efficiencies of aerosol-radiation interactions

        Parameters
        ----------
        ari_emitted
            Reference values
        """
        species = self.ari__species
        gases = self.ari__gases
        SAMPLES = self.SAMPLES
        NINETY_TO_ONESIGMA = self.NINETY_TO_ONESIGMA

        emis = self.din_ref['emis'][species].T
        conc = self.din_ref['conc'][gases].T
        self.ari__din_ref = pd.concat([emis, conc])

        # Radiative efficiency per Mt, ppb, or ppt
        year_pi = self.year_pi
        radeff = pd.concat([
            ari_emitted[species] / (emis[2019] - emis[year_pi]),
            ari_emitted[gases] / (conc[2019] - conc[year_pi]),
        ])
        radeff_std = pd.concat([
            ari_emitted_std[species] / (emis[2019] - emis[year_pi]),
            ari_emitted_std[gases] / (conc[2019] - conc[year_pi]),
        ])

        df = self.ari__din_ref.T * radeff

        d1 = df.sub(df.loc[year_pi]).sum(axis=1)
        scale = -0.3 / d1.loc[2005:2014].mean()

        unc_scale = 0.3 / ((d1.loc[2005:2014].mean()/-0.22) * np.sqrt(
            (ari_emitted_std**2).sum()
        ) * NINETY_TO_ONESIGMA)

        self.ari__radeff = radeff * scale
        species_order = [
            'BC', 'NH3', 'NMVOC', 'NOx', 'OC', 'SO2', 'CH4', 'N2O', 'ODS',
        ]
        self.ari__radeff_ens = pd.DataFrame(
            stats.norm.rvs(
                radeff[species_order] * scale,
                radeff_std[species_order] * unc_scale,
                size=(SAMPLES, len(species_order)),
                random_state=3729329,
            ),
            columns=species_order,
        )

    def erf__ari(self, df_emis_slcf, df_conc_ghg, em=None):
        """Calculate forcing of aerosol-radiation interactions

        Parameters
        ----------
        df_emis_slcf
            Input emissions in DataFrame or Series including relevant species
        df_conc_ghg
            Input concentrations in DataFrame or Series
            including relevant species

        Returns
        -------
            Forcing in Series or scalar
        """
        if em is None:
            radeff = self.ari__radeff
        else:
            radeff = self.ari__radeff_ens.loc[em]

        species = self.ari__species
        gases = self.ari__gases

        df = [
            df_emis_slcf[species],
            df_conc_ghg[gases[:2]],
        ]
        eesc = self.eesc_total(df_conc_ghg)
        if isinstance(eesc, pd.Series):
            df.append(eesc.rename(gases[2]).rename_axis(None))
            df = pd.concat(df, axis=1)
        else:
            df.append(pd.Series({gases[2]: eesc}))
            df = pd.concat(df)

        d_pi = self.ari__din_ref[self.year_pi]

        df = df - d_pi
        self.save__ari_dfin = df

        return df.mul(radeff).sum(axis=df.ndim-1)

    def _aci(self, df_emis_slcf, em=None):
        """Ensemble calculation of radiative-cloud interactions

        Parameters
        ----------
        df_emis_slcf
            Input emissions in DataFrame or Series

        Returns
        -------
            Ensemble results
        """
        species = self.aci__species

        if isinstance(df_emis_slcf, pd.DataFrame):
            if em is None:
                df = pd.DataFrame(
                    np.log(
                        1. + np.stack([
                            v.values[:, None] * self.aci__sample.loc[k].values
                            for k, v in df_emis_slcf[species].items()
                        ]).sum(axis=0)
                    ) * 1.1,
                    index=df_emis_slcf.index,
                )
            else:
                df = pd.Series(
                    np.log(
                        1. + np.vstack([
                            v.values * self.aci__sample.loc[k, em]
                            for k, v in df_emis_slcf[species].items()
                        ]).sum(axis=0)
                    ) * 1.1,
                    index=df_emis_slcf.index,
                )
        else:
            if em is None:
                df = np.log(
                    1. + np.vstack([
                        v * self.aci__sample.loc[k].values
                        for k, v in df_emis_slcf[species].items()
                    ]).sum(axis=0)
                ) * 1.1
            else:
                df = np.log(
                    1. + np.array([
                        v * self.aci__sample.loc[k, em]
                        for k, v in df_emis_slcf[species].items()
                    ]).sum()
                ) * 1.1

        return df

    def init__aci(self, aci_cal):
        """Set ensemble coefficient values and reference forcing levels
        of radiative-cloud interactions

        Parameters
        ----------
        aci_cal
            Calibrated data
        """
        year_pi = self.year_pi
        
        df_emis_slcf = self.din_ref['emis']
        SAMPLES = self.SAMPLES
        NINETY_TO_ONESIGMA = self.NINETY_TO_ONESIGMA
        
        species = self.aci__species

        aci_cal = aci_cal.rename({'Sulfur': 'SO2'}, axis=1)

        kde = stats.gaussian_kde(
            [np.log(aci_cal[x]) for x in species],
            bw_method=0.1,
        )
        aci_sample = np.exp(kde.resample(size=SAMPLES * 1, seed=63648708))

        erfaci_sample = stats.norm.rvs(
            size=SAMPLES, loc=-1.0, scale=0.7/NINETY_TO_ONESIGMA,
            random_state=71271,
        )

        self.aci__sample = pd.DataFrame(aci_sample, index=species)
        self.aci__erf_sample = pd.Series(erfaci_sample)

        self.aci__pi = self._aci(df_emis_slcf.loc[year_pi])
        self.aci__ref = self._aci(df_emis_slcf.loc[2005:2014]).mean(axis=0)

    def erf__aci(self, df_emis_slcf, em=None):
        """Calculate forcing of radiative-cloud interactions

        Parameters
        ----------
        df_emis_slcf
            Input emissions in DataFrame or Series including relevant species

        Returns
        -------
            Forcing in Series or scalar
        """
        df = self._aci(df_emis_slcf, em)

        if em is None:
            ret = (
                (df - self.aci__pi)
                / (self.aci__ref - self.aci__pi)
                * self.aci__erf_sample
            )
            self.save__aci_ens = ret
            ret = ret.median(axis=df.ndim-1)
        else:
            ret = (
                (df - self.aci__pi[em])
                / (self.aci__ref[em] - self.aci__pi[em])
                * self.aci__erf_sample[em]
            )

        return ret

    def init__o3(self, temp_obs, skeie_ozone_trop, skeie_ozone_strat):
        """Set radiative efficiencies of ozone precursors

        Parameters
        ----------
        temp_obs
            Observed global-mean surface temperature anomalies
        skeie_ozone_trop
            Calibrated data of tropospheric ozone
        skeie_ozone_strat
            Calibrated data of stratospheric ozone
        """
        species = self.o3__species
        conc = self.din_ref['conc'][species[:3]].T
        emis = self.din_ref['emis'][species[3:]].T

        din_ref = pd.concat([conc, emis]).T
        self.o3__din_ref = din_ref

        delta_gmst = np.hstack([
            0.,
            [
                temp_obs.loc[slice(*period)].mean()
                for period in [
                    (1915, 1925),
                    (1925, 1935),
                    (1935, 1945),
                    (1945, 1955),
                    (1955, 1965),
                    (1965, 1975),
                    (1975, 1985),
                    (1985, 1995),
                    (1995, 2005),
                    (2002, 2012),
                    (2005, 2015),
                    (2009, 2019),
                ]
            ],
            # not used
            temp_obs.loc[2017],
            temp_obs.loc[2018],
        ])

        good_models = [
            'BCC-ESM1',
            'CESM2(WACCM6)',
            'GFDL-ESM4',
            'GISS-E2-1-H',
            'MRI-ESM2-0',
            'OsloCTM3',
        ]
        years = [1850] + skeie_ozone_trop.columns.tolist()
        skeie_total = sum([
            df
            .loc[good_models]
            .reindex(years, axis=1, fill_value=0.)
            .interpolate(axis=1, method='values', limit_area='inside')
            for df in [skeie_ozone_trop, skeie_ozone_strat]
        ])

        skeie_ssp245 = pd.concat([
            pd.Series({1750: -0.03}),
            pd.concat([
                skeie_total.drop('OsloCTM3') - (-0.037) * delta_gmst,
                skeie_total.loc[['OsloCTM3']],
            ]).mean(),
        ]) + 0.03

        skeie_ssp245 = pd.concat([
            skeie_ssp245.drop([2014, 2017, 2020]),
            skeie_total.loc['OsloCTM3', 2014:] - skeie_total.loc['OsloCTM3', 2010] + skeie_ssp245[2010],
        ])

        f = interp1d(
            skeie_ssp245.index, skeie_ssp245,
            bounds_error=False, fill_value='extrapolate',
        )
        years = np.arange(1750, 2021)
        o3_total = pd.Series(f(years), index=years)

        delta = din_ref.loc[2014] - din_ref.loc[1750]

        radeff = pd.Series({
            'CH4': 0.14,
            'N2O': 0.03,
            'ODS': -0.11, # excludes UKESM
            'CO': 0.067, # stevenson CMIP5 scaled to CO + VOC total
            'NMVOC': 0.043, # stevenson CMIP5 scaled to CO + VOC total
            'NOx': 0.20,
        }) / delta

        fac_cmip6_skeie = (radeff * delta).sum() / (
            o3_total.loc[2014] - o3_total.loc[1750]
        )

        def fit_precursors(x, rch4, rn2o, rods, rco, rvoc, rnox):
            return (
                rch4 * x[0] + rn2o * x[1] + rods * x[2] + rco * x[3] + rvoc * x[4] + rnox * x[5]
            )

        p, cov = curve_fit(
            fit_precursors,
            din_ref.loc[:2019].sub(din_ref.loc[1750]).T.loc[species].values,
            o3_total.loc[:2019].sub(o3_total.loc[1750]),
            bounds=[  # 90% range from Thornhill for each precursor
                np.array([
                    0.09 / delta['CH4'],
                    0.01 / delta['N2O'],
                    -0.21 / delta['ODS'],
                    0.010 / delta['CO'],
                    0 / delta['NMVOC'],
                    0.09 / delta['NOx'],
                ]) / fac_cmip6_skeie,
                np.array([
                    0.19 / delta['CH4'],
                    0.05 / delta['N2O'],
                    -0.01 / delta['ODS'],
                    0.124 / delta['CO'],
                    0.086 / delta['NMVOC'],
                    0.31 / delta['NOx'],
                ]) / fac_cmip6_skeie,
            ],
        )
        self.o3__coeff = pd.Series(p, index=species)

    def erf__o3(self, df_emis_slcf, df_conc_ghg):
        """Calculate ozone forcing

        Parameters
        ----------
        df_emis_slcf
            Input precursor emissions in DataFrame or Series
        df_conc_ghg
            Input precursor concentrations in DataFrame or Series

        Returns
        -------
            Forcing in Series or scalar
        """
        species = self.o3__species
        if isinstance(df_emis_slcf, pd.DataFrame):
            df = pd.concat([
                df_conc_ghg[species[:2]],
                self.eesc_total(df_conc_ghg).rename(species[2]),
                df_emis_slcf[species[3:]],
            ], axis=1)
        else:
            df = pd.concat([
                df_conc_ghg[species[:2]],
                pd.Series({species[2]: self.eesc_total(df_conc_ghg)}),
                df_emis_slcf[species[3:]],
            ])

        return df.sub(
            self.o3__din_ref.loc[self.year_pi]
        ).mul(self.o3__coeff).sum(axis=df.ndim-1)

    def erf__bc_on_snow(self, df_emis_slcf):
        """Calculate forcing of light-absorbing particles on snow and ice

        Parameters
        ----------
        df_emis_slcf
            Input emissions in DataFrame or Series including BC

        Returns
        -------
            Forcing in Series or scalar
        """
        d1 = df_emis_slcf['BC']
        if isinstance(d1, pd.Series):
            d1 = d1.rename(None)

        return (
            d1 - self.din_ref['emis'].loc[self.year_pi, 'BC']
        ) * self.bc_on_snow__factor

    def erf__ghg(self, df_conc_ghg, agg_minor=False):
        """Calculate GHGs forcing

        Parameters
        ----------
        df_conc_ghg
            Input concentrations in DataFrame or Series
        agg_minor, optional
            Halogenated species are aggregated if True, by default False

        Returns
        -------
            Forcing in DataFrame or Series
        """
        map_kw = {
            'CO2': {'cn2o': df_conc_ghg['N2O']},
            'CH4': {'cn2o': df_conc_ghg['N2O']},
            'N2O': {
                'cco2': df_conc_ghg['CO2'],
                'cch4': df_conc_ghg['CH4'],
            },
        }
        ghg_major = []
        ghg_minor = []
        for gas, _ in df_conc_ghg.items():
            if gas in ['CO2', 'CH4', 'N2O']:
                ghg_major.append(gas)
            else:
                ghg_minor.append(gas)

        df = {
            gas: self.c2erf_ar6(gas, d1, **map_kw[gas])
            for gas, d1 in df_conc_ghg[ghg_major].items()
        }

        if df_conc_ghg.ndim == 2:
            df = pd.DataFrame(df, index=df_conc_ghg.index)
        else:
            df = pd.Series(df)

        df_hc = (
            df_conc_ghg[ghg_minor]
            .sub(self.din_ref['conc'].loc[self.year_pi, ghg_minor])
            .mul(self.ghg__hc_eff[ghg_minor])
        )
        if agg_minor:
            if isinstance(agg_minor, str):
                name = agg_minor
            else:
                name = 'OTHER_WMGHG'

            if df_conc_ghg.ndim == 2:
                df_hc = df_hc.sum(axis=1).rename(name)
            else:
                df_hc = pd.Series({name: df_hc.sum()})

        df = pd.concat([df, df_hc], axis=df.ndim-1)

        return df

    def erf__land_use(self, df_emis_co2):
        """Calculate land use forcing

        Parameters
        ----------
        df_emis_co2
            Input emissions in DataFrame or Series including CO2 AFOLU
            When Series is given, it should contain cumulative sums
            at a speciic time point

        Returns
        -------
            Forcing in Series or scalar
        """
        d1 = df_emis_co2['CO2 AFOLU']
        if isinstance(d1, pd.Series):
            d1 = d1.cumsum().rename(None)

        return (d1 - self.land_use__pi) * self.land_use__factor

    def erf__h2o_strat(self, df_erf_ghg):
        """Calculate stratospheric water vapor forcing

        Parameters
        ----------
        df_erf_ghg
            Input forcing in DataFrame or Series including CH4

        Returns
        -------
            Forcing in Series or scalar
        """
        d1 = df_erf_ghg['CH4']
        if isinstance(d1, pd.Series):
            d1 = d1.rename(None)

        return d1 * self.h2o_strat__factor

    def erf__others(self, *args):
        """Return data given as forcing of categories not subject to calculation

        Returns
        -------
            Forcing in DataFrame or Series
        """
        if len(args) == 0:
            df = self.din.get_data('erf')
        else:
            time = np.hstack(args)
            if np.issubdtype(time.dtype, np.integer):
                df = self.din.get_data('erf', time)
            else:
                df = self.din.interp_eval('erf', time)

        return df