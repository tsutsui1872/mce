import numpy as np
import pandas as pd

class EnsembleERF:
    def __init__(self, idx_member):
        self.idx_member = idx_member
        self.df_erf_hist_ens = []
        self.df_erf_ssps_ens = []
        self.scenarios = None

    def add_erf_ensemble(self, df_erf_scale, df_erf_hist_in, df_erf_in):
        cats = [
            'co2', 'ch4', 'n2o', 'other_wmghg',
            'o3', 'h2o_stratospheric', 'contrails', 'bc_on_snow', 'land_use',
            'volcanic',
        ]
        self.df_erf_hist_ens.append(
            pd.concat({
                cat:
                pd.DataFrame(
                    d1.values[:, None] * df_erf_scale[cat].values,
                    index=d1.index, columns=self.idx_member,
                ).T
                for cat, d1 in df_erf_hist_in.loc[cats, :2019].iterrows()
            })
        )
        self.df_erf_ssps_ens.append(
            pd.concat({
                idx:
                pd.DataFrame(
                    d1.values[:, None] * df_erf_scale[idx[1]].values,
                    index=d1.index, columns=self.idx_member,
                ).T
                for idx, d1 in df_erf_in.loc[(slice(None), cats), 2015:2100].iterrows()
            })
        )
        self.scenarios = df_erf_in.index.get_level_values(0).unique()

    def aerosol_forcing(self, dfin, dfin_base, aer_coeff):
        ret = {}

        df = dfin.sub(dfin_base, axis=0)

        ret['aerosol-radiation_interactions'] = pd.DataFrame(
            df.loc[['SO2']].T.values * aer_coeff['beta_so2'] * 32./64.
            + df.loc[['BC']].T.values * aer_coeff['beta_bc']
            + df.loc[['OC']].T.values * aer_coeff['beta_oc']
            + df.loc[['NH3']].T.values * aer_coeff['beta_nh3'],
            index=dfin.columns, columns=self.idx_member,
        ).T

        df = -aer_coeff['beta'] * np.log(
            1.
            + dfin.loc[['SO2']].mul(32./64.).T.values / aer_coeff['aci_coeffs'][:, 0]
            + dfin.loc[['BC', 'OC']].sum().values[:, None] / aer_coeff['aci_coeffs'][:, 1]
        )
        d_base = -aer_coeff['beta'] * np.log(
            1.
            + dfin_base.loc[['SO2']].mul(32./64.).values / aer_coeff['aci_coeffs'][:, 0]
            + np.array([dfin_base.loc[['BC', 'OC']].sum()]) / aer_coeff['aci_coeffs'][:, 1]
        )
        ret['aerosol-cloud_interactions'] = pd.DataFrame(
            df - d_base,
            index=dfin.columns, columns=self.idx_member,
        ).T

        return pd.concat(ret)

    def add_aerosol_forcing(self, aer_coeff, df_emis_hist, df_emis):
        self.df_erf_hist_ens.append(
            self.aerosol_forcing(
                df_emis_hist, df_emis_hist.loc[:, 1750], aer_coeff,
            )
        )
        self.df_erf_ssps_ens.append(
            df_emis.groupby('Scenario').apply(
                lambda df:
                self.aerosol_forcing(
                    df.loc[:, 2015:2100].droplevel('Scenario'),
                    df_emis_hist.loc[:, 1750],
                    aer_coeff,
                )
            )
        )

    def add_solar_forcing(self, df_erf_scale, df_solar, trend_solar):
        df = pd.concat([
            pd.DataFrame(
                (df_solar.loc[1750:2019, 'solar_erf'].values[:, None]
                * df_erf_scale['solar'].values)
                + np.linspace(0, trend_solar, 270),
                index=df_solar.loc[1750:2019].index,
                columns=self.idx_member,
            ),
            pd.DataFrame(
                df_solar.loc[2020:2100, 'solar_erf'].values[:, None]
                * df_erf_scale['solar'].values
                + trend_solar,
                index=df_solar.loc[2020:2100].index,
                columns=self.idx_member,
            ),
        ]).T

        self.df_erf_hist_ens.append(pd.concat({'solar': df.loc[:, :2019]}))
        self.df_erf_ssps_ens.append(
            pd.concat({
                (scenario, 'solar'): df.loc[:, 2015:] for scenario in self.scenarios
            })
        )

    def compile_cats(self):
        df_erf_hist_ens = (
            pd.concat(self.df_erf_hist_ens)
            .sort_index()
            .rename_axis(index=['Category', 'Member'])
            .rename_axis(columns='Year')
        )
        df_erf_ssps_ens = (
            pd.concat(self.df_erf_ssps_ens)
            .sort_index()
            .rename_axis(index=['Scenario', 'Category', 'Member'])
            .rename_axis(columns='Year')
        )
        return df_erf_hist_ens, df_erf_ssps_ens