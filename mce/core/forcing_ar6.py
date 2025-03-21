import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from .. import MCEExecError
from .forcing import RfAll

class RfAllAR6(RfAll):
    def __init__(self, emis_hist, conc_hist, *args, **kw):
        super().__init__(*args, **kw)

        self.SAMPLES = 100000
        self.NINETY_TO_ONESIGMA = stats.norm.ppf(0.95)

        year_pi = 1750
        self.year_pi = year_pi
        self.conc_pi = conc_hist.loc[year_pi]

        # Adjust a parameter for the AR6 CO2 forcing scheme
        self.parms_ar6_ghg.update(C0_1750=self.conc_pi['CO2'])

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
            for gas in conc_hist
        })
        # Make sure gas order consistency
        if list(self.ghg__hc_eff.keys()) != list(self.conc_pi.keys()):
            raise MCEExecError('gas order inconsistency')

        # Set a forcing factor for stratospheric water vapor
        yref = 2019
        # erf1 = self.c2erf_ar6(
        #     'CH4', conc_hist.loc[yref, 'CH4'],
        #     cn2o=conc_hist.loc[yref, 'N2O'],
        # )
        # self.h2o_strat__factor = 0.05 / erf1
        self.h2o_strat__factor = 0.05 / (
            conc_hist.loc[yref, 'CH4'] - conc_hist.loc[year_pi, 'CH4']
        )

        # Set a forcing factor for light-absorbing particles on snow and ice
        self.bc_on_snow__factor = 0.08 / (
            emis_hist.loc[2019, 'BC']
            - emis_hist.loc[year_pi, 'BC']
        )

        self.emis_hist = emis_hist
        self.conc_hist = conc_hist

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

    def eesc_total(self, conc):
        """Calculate total equivalent effective stratospheric chlorine (EESC)

        Parameters
        ----------
        conc
            ODS concentrations in DataFrame or Series

        Returns
        -------
            EESC in Series or scalar
        """
        ods = self.ods
        df = conc
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

        emis = self.emis_hist[species].T
        conc = self.conc_hist.T.reindex(gases)
        conc.loc['ODS'] = self.eesc_total(self.conc_hist)
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
        aci__sample = self.aci__sample.loc[:, slice(em, em)]

        if isinstance(df_emis_slcf, pd.DataFrame):
            if em is None:
                vs = np.stack([
                    v.values[:, None] * aci__sample.loc[k].values
                    for k, v in df_emis_slcf[species].items()
                ])
                pd_method = pd.DataFrame
            else:
                vs = np.stack([
                    v.values * aci__sample.loc[k].values
                    for k, v in df_emis_slcf[species].items()
                ])
                pd_method = pd.Series
        else:
            vs = np.stack([
                v * aci__sample.loc[k].values
                for k, v in df_emis_slcf[species].items()
            ])
            pd_method = None

        df = np.log(1. + vs.sum(axis=0)) * 1.1

        if pd_method is not None:
            df = pd_method(df, index=df_emis_slcf.index)

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
        
        df_emis_slcf = self.emis_hist
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
        if species != ['CH4', 'N2O', 'ODS', 'CO', 'NMVOC', 'NOx']:
            raise MCEExecError('precursor order inconsistent')

        conc = self.conc_hist.T.reindex(species[:3])
        conc.loc['ODS'] = self.eesc_total(self.conc_hist)
        emis = self.emis_hist[species[3:]].T

        din_ref = pd.concat([conc, emis]).T
        self.o3__din_ref = din_ref

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

        delta_gmst = np.hstack([
            0., [
                temp_obs.loc[y-5:y+5].mean() for y in
                list(range(1920, 2000+1, 10)) + [2007, 2010, 2014]
            ],
            temp_obs.loc[2017:2018].values, # not used
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

        radeff = [0.14, 0.03, -0.11, 0.067, 0.043, 0.20]
        radeff = pd.Series(radeff, index=species) / delta

        fac_cmip6_skeie = (radeff * delta).sum() / (
            o3_total.loc[2014] - o3_total.loc[1750]
        )

        bounds_l = [0.09, 0.01, -0.21, 0.010, 0.,    0.09]
        bounds_u = [0.19, 0.05, -0.01, 0.124, 0.086, 0.31]
        p, _ = curve_fit(
            lambda x, rch4, rn2o, rods, rco, rvoc, rnox:
            sum([
                r1 * x1 for x1, r1 in
                zip(x, [rch4, rn2o, rods, rco, rvoc, rnox])
            ]),
            din_ref.loc[:2019].sub(din_ref.loc[1750]).T.loc[species].values,
            o3_total.loc[:2019].sub(o3_total.loc[1750]),
            bounds=[  # 90% range from Thornhill for each precursor
                np.array([
                    v / delta[k] for k, v in zip(species, bounds_l)
                ]) / fac_cmip6_skeie,
                np.array([
                    v / delta[k] for k, v in zip(species, bounds_u)
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

    def erf__ghg_major(self, conc):
        """Calculate forcing for each of CO2, CH4, and N2O

        Parameters
        ----------
        conc
            GHG concentrations in DataFrame or Series

        Returns
        -------
            Forcing in Series or scalar
        """
        kw = {
            'c{}'.format(gas.lower()): conc.get(gas)
            for gas in ['CO2', 'CH4', 'N2O']
        }
        df = {
            gas: self.c2erf_ar6(gas, conc[gas], **kw)
            for gas in ['CO2', 'CH4', 'N2O'] if gas in conc
        }
        if conc.ndim == 1:
            df = pd.Series(df)
        else:
            df = pd.DataFrame(df, index=conc.index)

        return df

    def erf__ghg_minor(self, conc):
        """Calculate forcing for each of halogenated species

        Parameters
        ----------
        conc
            GHG concentrations in DataFrame or Series

        Returns
        -------
            Forcing in Series or scalar
        """
        df = (conc - self.conc_pi) * self.ghg__hc_eff
        # nan given for CO2, CH4, and N2O if included
        df = df.dropna(axis=df.ndim-1)
        return df

