"""
API for manipulating perturbed parameters.
"""

import numpy as np
from netCDF4 import Dataset

class ParmsPerturbed(object):
    """
    Functions to prepare input model parameters for perturbed parameter
    ensemble runs.

    Parameters
    ----------
    path : str
        Parameter file path. Perturbed values are assemed to be save in a
        single NetCDF file.
    """
    def __init__(self, path):
        self.ncf = Dataset(path)

    def close(self):
        self.ncf.close()

    def get_parms_irm(self, m):
        """
        Returns climate response parameters for a given ensemble member.

        Parameters
        ----------
        m : int/str
            Index label of the member to be retrieved.

        Returns
        -------
        kw : dict
            Climate response parameters.
        """
        ncvars = self.ncf.variables

        kw = {
            'asj': np.array([
                ncvars[f'climate__{name}'][m].item()
                for name in ['a0', 'a1', 'a2']
            ]),
            'tauj': np.array([
                ncvars[f'climate__{name}'][m].item()
                for name in ['tau0', 'tau1', 'tau2']
            ]),
            'lamb': ncvars['climate__lambda'][m].item(),
        }

        return kw

    def get_parms_rfall(self, m, conc_pi={}, co2_only=False):
        """
        Returns forcing parameters for a given ensemble member.

        Parameters
        ----------
        m : int/str
            Index label of the member to be retrieved.

        conc_pi : dict, optional, default {}
            Preindustrial concentrations.
            Keys are a tuple of label and units, such as ('CO2', 'ppm').

        co2_only : bool, optional, default False
            If True, includes non-CO2 scaling factors.

        Returns
        -------
        kw : dict
            Forcing parameters.
        """
        ncvars = self.ncf.variables

        kw = {
            'alpha': ncvars['climate__alpha'][m].item(),
            'beta': ncvars['climate__beta'][m].item(),
        }

        if co2_only:
            if ('CO2', 'ppm') in conc_pi:
                kw['ccref'] = conc_pi[('CO2', 'ppm')]
        else:
            kw['conc_pi'] = conc_pi
            names = [
                'CH4', 'F-Gases', 'Montreal_Gases', 'N2O', 'Aerosols',
                'Albedo_Change', 'BC_on_Snow',
                'CH4_Oxidation_Stratospheric_H2O',
                'Stratospheric_Ozone', 'Tropospheric_Ozone',
            ]
            kw['scaling_factor'] = {
                name.replace('_', ' '): ncvars[f'forcing__{name}'][m].item()
                for name in names
            }

        return kw

    def get_parms_ocean(self, m, yinit=1850, conc_pi={}):
        """
        Returns ocean carbon cycle parameters for a given ensemble member.

        Parameters
        ----------
        m : int/str
            Index label of the member to be retrieved.

        yinit : int, optional, default 1850
            Year of initialization

        conc_pi : dict, optional, default {}
            Preindustrial concentrations.
            Keys are a tuple of label and units, such as ('CO2', 'ppm').

        Returns
        -------
        kw : dict
            Ocean carbon cycle parameters.
        """
        ncvars = self.ncf.variables

        names = ['hls', 'hl1', 'hl2', 'hl3', 'eta1', 'eta2', 'eta3']
        kw = {
            name: ncvars[f'carbon__{name}__init{yinit}'][m].item()
            for name in names
        }
        if ('CO2', 'ppm') in conc_pi:
            kw['cco2_pi'] = conc_pi[('CO2', 'ppm')]

        return kw

    def get_parms_land(self, m):
        """
        Returns land carbon cycle parameters for a given ensemble member.

        Parameters
        ----------
        m : int/str
            Index label of the member to be retrieved.

        Returns
        -------
        kw : dict
            Land carbon cycle parameters.
        """
        ncvars = self.ncf.variables

        kw = {
            name: ncvars[f'carbon__{name}'][m].item()
            for name in ['beta', 'fb_alpha']
        }

        return kw

    def get_parms_all(self, m, conc_pi={}, co2_only=False, yinit=1850):
        """
        Returns all-category parameters for a given ensemble member.

        Parameters
        ----------
        m : int/str
            Index label of the member to be retrieved.

        conc_pi : dict, optional, default {}
            Preindustrial concentrations.
            Keys are a tuple of label and units, such as ('CO2', 'ppm').

        co2_only : bool, optional, default False
            If True, includes non-CO2 scaling factors in forcing parameters.

        yinit : int, optional, default 1850
            Year of ocean carbon cycle initialization

        Returns
        -------
        kws : dict
            Pairs of category and parameters in a dictionary form.
        """
        kws = {
            'kw_irm': self.get_parms_irm(m),
            'kw_rfall': self.get_parms_rfall(
                m, conc_pi=conc_pi, co2_only=co2_only),
            'kw_ocean': self.get_parms_ocean(m, yinit=yinit, conc_pi=conc_pi),
            'kw_land': self.get_parms_land(m),
        }

        return kws

