"""
API for handling input data from RCMIP.
"""

import numpy as np
import pandas as pd
import os

class InputRcmip(object):
    """
    Utilities for handling RCMIP data.

    Parameters
    ----------
    indir : str
        Input data directory.

    ver_s : str
        Scenario data version.

    ver_i : str, optional, default None
        Indicator data version.
    """
    def __init__(self, indir, ver_s, ver_i=None):
        self.dirpath = indir
        self.infiles = {
            'emissions': f'rcmip-emissions-annual-means-{ver_s}.csv',
            'concentrations': f'rcmip-concentrations-annual-means-{ver_s}.csv',
            'forcing': f'rcmip-radiative-forcing-annual-means-{ver_s}.csv',
        }
        if ver_i is not None:
            self.infiles['indicators'] = f'assessed-ranges-{ver_i}.csv'
        self.data = {}

    def get_data(self, kind, **kw):
        """
        Returns a data frame for given conditions.

        Parameters
        ----------
        kind : str
            Kind of data file.

        kw : dict, optional
            Data selection conditions given by
            key: data column identifier
            value: selected value(s) for the column

        Returns
        -------
        df : pandas.DataFrame
            Copy of a subset of the data frame
        """
        if kind not in self.data:
            self.data[kind] = pd.read_csv(os.path.join(
                self.dirpath, self.infiles[kind]))

        dfin = self.data[kind]
        cond = None

        for k, v in kw.items():
            if isinstance(v, list):
                cond1 = dfin[k] == v[0]
                for v1 in v[1:]:
                    cond1 = cond1 | (dfin[k] == v1)
            else:
                cond1 = dfin[k] == v

            if cond is None:
                cond = cond1
            else:
                cond = cond & cond1

        if cond is not None:
            df = dfin[cond].copy()
        else:
            df = dfin.copy()

        return df

    def get_data_series(
            self, kind, scenario, region='World', drop_elem_vname=[],
            id_vars=[], is_dropna=False, period=[], **kw):
        """
        Returns time series data for given conditions.

        Parameters
        ----------
        kind : str
            Kind of data file.

        scenario : str
            Scenario name.

        region : str, optional, default 'World'
            Region name.

        drop_elem_vname : list, optional, default []
            Elements of variable names to be dropped.

        id_vars : list, optional, default []
            Columns to be used for index. Other non-number columns are dropped.

        is_dropna : bool, optional, default False
            If true, columns including na values dropped.

        period : list, optional, default []
            Start and end years are specified with int or str type,
            where either can be None.
            After selected, column values are converted from str to int values.

        kw : dict, optional
            Data selection conditions given by
            key: data column identifier
            value: selected value(s) for the column

        Returns
        -------
        df : pandas.DataFrame
            Copy of selected time series
        """

        df = self.get_data(kind, Scenario=scenario, Region=region, **kw)

        if drop_elem_vname:
            df['Variable'] = df['Variable'].transform(
                lambda x: '|'.join(
                    [x1 for x1 in x.split('|') if x1 not in drop_elem_vname])
            )

        if id_vars:
            df = df.set_index(id_vars).select_dtypes(include=[np.number])

        if is_dropna:
            df = df.dropna(axis=1)

        if period:
            slc = slice(*[isinstance(x, int) and str(x) or x for x in period])
            df = df.loc[:, slc].rename(columns=int)

        return df

    def get_data_series_forcing(self, scenario, **kw):
        """
        Reads RCMIP time series of non-GHG forcing factors for a given
        scenario and returns the data.

        Parameters
        ----------
        scenario : str
            Scenario label.

        kw : dict, optional
            Keyword arguments to be passed to get_data_series method.

        Returns
        -------
        df : pandas.DataFrame
            Time series of non-GHG forcing factors in W/m2.
        """
        vn = 'Effective Radiative Forcing'
        variables_a = [
            'Aerosols',
            'Tropospheric Ozone',
            'Stratospheric Ozone',
            'Albedo Change',
            'Other|BC on Snow',
            'Other|CH4 Oxidation Stratospheric H2O',
            'Other|Contrails and Contrail-induced Cirrus',
            # 'Other|Other WMGHGs',
        ]
        variables_n = ['Solar', 'Volcanic']
        variables = (
            ['|'.join([vn, 'Anthropogenic', v]) for v in variables_a]
            +
            ['|'.join([vn, 'Natural', v]) for v in variables_n]
        )
        drop_elem_vname = [vn, 'Anthropogenic', 'Natural', 'Other']
        id_vars = ['Variable']
        # id_vars = [('Variable', 'Unit')]

        df = self.get_data_series(
            'forcing', scenario, Variable=variables,
            drop_elem_vname=drop_elem_vname, id_vars=id_vars, **kw)

        return df

    def get_input_cdrv(self, scenario, period=[1850, None], co2_only=False):
        """
        Reads RCMIP time series of CO2 concentrations and other forcing for a
        given scenario and returns the data in a form of MCE driver arguments.

        Parameters
        ----------
        scenario : str
            Scenario label.

        period : list, optional, default [1850, None]
            Start and end years specified with int or str type,
            where either can be None.

        co2_only : bool, optional, default False
            If True, non-CO2 data (df_ghg, df_other_erf) are not returned.

        Returns
        -------
        time : np.array of int values
            Time points in year. Not necessarily equally spaced.

        kwargs : dict
            Contains the following keys and values
            cco2 : numpy.array of CO2 concentrations in ppm
            df_ghg : pandas.DataFrame of GHG concentrations
                Units are written as an index level value
            df_erf_other : pandas.DataFrame of other forcing in W/m2
        """
        df = self.get_data_series(
            'concentrations', scenario,
            drop_elem_vname=['Atmospheric Concentrations'],
            id_vars=['Variable', 'Unit'],
            is_dropna=True, period=period,
        )
        time = df.columns.values
        kwargs = {
            'cco2': df.loc[('CO2', 'ppm')].values,
        }

        if not co2_only:
            kwargs.update(
                df_ghg = df,
                df_erf_other = self.get_data_series_forcing(
                    scenario, period=period
                ),
            )

        return time, kwargs

    def get_input_edrv(self, scenario, period=[1850, None], co2_only=False):
        """
        Reads RCMIP time series of CO2 emissions and other forcing for a
        given scenario and returns the data in a form of MCE driver arguments.

        Parameters
        ----------
        scenario : str
            Scenario label.

        period : list, optional, default [1850, None]
            Start and end years specified with int or str type,
            where either can be None.

        co2_only : bool, optional, default False
            If True, non-CO2 data (df_ghg, df_other_erf) are not returned.

        Returns
        -------
        time : np.array of int values
            Time points in year. Not necessarily equally spaced.

        kwargs : dict
            Contains the following keys and values
            eco2 : numpy.array of CO2 emissions in GtC
            df_ghg : pandas.DataFrame of GHG concentrations
                Units are written as an index level value
            df_erf_other : pandas.DataFrame of other forcing in W/m2
        """
        conv_gtc = 12./44.*1e-3
        df = self.get_data_series(
            'emissions', scenario, Variable='Emissions|CO2',
            id_vars=['Variable', 'Unit'], is_dropna=True, period=period,
        )
        time = df.columns.values
        kwargs = {
            'eco2': df.loc[('Emissions|CO2', 'Mt CO2/yr')].values * conv_gtc,
        }

        if not co2_only:
            _, kwargs2 = self.get_input_cdrv(scenario, period=period)
            kwargs.update(
                df_ghg = kwargs2['df_ghg'],
                df_erf_other = kwargs2['df_erf_other'],
            )

        return time, kwargs

    def get_indicator_range(self, name):
        """
        Reads the indicator file and returns quantile values.

        Parameters:
        -----------
        name : str
            RCMIP name.

        Returns:
        d1 : pandas.Series
            Quantile values.
        """
        pnames = [
            'very_likely__lower',
            'likely__lower',
            'central',
            'likely__upper',
            'very_likely__upper',
        ]
        id_vars = ['RCMIP name', 'unit']
        d1 = self.get_data(
            'indicators', **{'RCMIP name': name}
        ).set_index(id_vars)[pnames].squeeze()
        return d1


if __name__ == '__main__':
    obj = InputRcmip('./rcmip', 'v4-0-0', 'v2-2-0')
    df = obj.get_data_series(
        'concentrations', 'historical',
        drop_elem_vname=['Atmospheric Concentrations'],
        id_vars=['Variable', 'Unit'],
        is_dropna=True, period=[1750, None]
    )

