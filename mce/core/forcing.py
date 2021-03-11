"""
API for computing effective radiative forcing (ERF) of CO2 and others.
Non-CO2 agents are partly supported.
"""

import numpy as np
from mce.core import ModelBase

class RfCO2(ModelBase):
    def init_process(self, *args, **kw):
        """
        Parameters
        ----------
        args : dummy, optional
            Not used.

        kw : dict, optional
            Keyword arguments to update module parameters.
            Valid parameters:
            alpha : scaling factor in W/m2
            beta : amplification factor from the first to second doubling
            ccref : pre-industrial CO2 concentration in ppm
        """
        self.parms = {
            # 'alpha': 4.701, # CMIP5 mean
            'alpha': 4.617, # adjusted CMIP5 mean for IRM-3
            # 'alpha': 4.561, # adjusted CMIP5 mean for IRM-2
            'beta': 1.068, # CMIP5 mean for IRM-3
            # 'beta': 1.062, # CMIP5 mean for IRM-2
            'ccref': 278.,
        }
        self.parms_update(**kw)

    def c2erf(self, *args):
        """
        Compute effective radiative forcing (ERF) of CO2 concentrations.

        Parameters
        ----------
        args : array-like
            Input values of CO2 concentrations in ppm.

        Returns
        -------
        erf : array-like
            Output values of ERF.
        """
        ccref = self.parms['ccref']
        return self.xl2erf(np.log(np.hstack(args)/ccref))

    def x2erf(self, *args):
        """
        Compute effective radiative forcing (ERF) of CO2 given by C/Cref
        where C is a CO2 concentration, and Cref is its reference value
        in a pre-industrial period.

        Parameters
        ----------
        args : array-like
            Input values of C/Cref.

        Returns
        -------
        erf : array-like
            Output values of ERF.
        """
        return self.xl2erf(np.log(np.hstack(args)))

    def xl2erf(self, *args):
        """
        Compute effective radiative forcing (ERF) of CO2 given by log(x)
        where x is a ratio of CO2 concentration to its reference value
        in a pre-industrial period.

        Parameters
        ----------
        args : array-like
            Input values of log(x).

        Returns
        -------
        erf : array-like
            Output values of ERF.
        """
        alpha = self.parms['alpha']
        beta = self.parms['beta']

        xl = np.hstack(args)
        erf = alpha * xl

        if beta != 1:
            f2x = alpha * np.log(2.)
            ix4 = erf > 2*f2x
            ix2 = (erf > f2x) & (erf <= 2*f2x)
            # erf[ix4] = beta*erf[ix4]
            # modified 2020-12-11
            erf[ix4] = (4.*beta-3.)*erf[ix4] - 6.*f2x*(beta-1.)
            erf[ix2] = (
                (beta-1) * (erf[ix2]-2*f2x) * (2/f2x*erf[ix2]-1)
                + beta*erf[ix2]
            )

        if len(erf) == 1:
            erf = erf.item()

        return erf


class RfAll(RfCO2):
    def init_process(self, *args, **kw):
        """
        Parameters
        ----------
        args : dummy, optional
            Not used.

        kw : dict, optional
            Keyword arguments to update module parameters.
        """
        self.parms = {
            # 'alpha': 4.701, # CMIP5 mean
            'alpha': 4.617, # adjusted CMIP5 mean for IRM-3
            # 'alpha': 4.561, # adjusted CMIP5 mean for IRM-2
            'beta': 1.068, # CMIP5 mean for IRM-3
            # 'beta': 1.062, # CMIP5 mean for IRM-2
            # AR5 WGI Table AII.1
            'ccref': 278., # ppm
            'cmref': 722., # ppb
            'cnref': 270., # ppb
            # Etminan et al. (2016) Table 1
            # units: W/m2/ppm for CO2, W/m2/ppb for N2O and CH4
            'co2_a1': -2.4e-7,
            'co2_b1': 7.2e-4,
            'co2_c1': -2.1e-4,
            'n2o_a2': -8.e-6,
            'n2o_b2': 4.2e-6,
            'n2o_c2': -4.9e-6,
            'ch4_a3': -1.3e-6,
            'ch4_b3': -8.2e-6,
        }
        """
        Radiative efficiency in W/m2/ppb adopted from IPCC WG1 AR5 Appendix 8.A
        """
        self.parms['efficiency'] = {
            'F-Gases|HFC|HFC125': 0.23,
            'F-Gases|HFC|HFC134a': 0.16,
            'F-Gases|HFC|HFC143a': 0.16,
            'F-Gases|HFC|HFC152a': 0.10,
            'F-Gases|HFC|HFC227ea': 0.26,
            'F-Gases|HFC|HFC23': 0.18,
            'F-Gases|HFC|HFC236fa': 0.24,
            'F-Gases|HFC|HFC245fa': 0.24,
            'F-Gases|HFC|HFC32': 0.11,
            'F-Gases|HFC|HFC365mfc': 0.22,
            'F-Gases|HFC|HFC4310mee': 0.42,
            'F-Gases|NF3': 0.20,
            'F-Gases|PFC|C2F6': 0.25,
            'F-Gases|PFC|C3F8': 0.28,
            'F-Gases|PFC|C4F10': 0.36,
            'F-Gases|PFC|C5F12': 0.41,
            'F-Gases|PFC|C6F14': 0.44,
            'F-Gases|PFC|C7F16': 0.50,
            'F-Gases|PFC|C8F18': 0.55,
            'F-Gases|PFC|CF4': 0.09,
            'F-Gases|PFC|cC4F8': 0.32,
            'F-Gases|SF6': 0.57,
            'F-Gases|SO2F2': 0.20,
            'Montreal Gases|CCl4': 0.17,
            'Montreal Gases|CFC|CFC11': 0.26,
            'Montreal Gases|CFC|CFC113': 0.30,
            'Montreal Gases|CFC|CFC114': 0.31,
            'Montreal Gases|CFC|CFC115': 0.20,
            'Montreal Gases|CFC|CFC12': 0.32,
            'Montreal Gases|CH2Cl2': 0.03,
            'Montreal Gases|CH3Br': 0.004,
            'Montreal Gases|CH3CCl3': 0.07,
            'Montreal Gases|CH3Cl': 0.01,
            'Montreal Gases|CHCl3': 0.08,
            'Montreal Gases|HCFC141b': 0.1,
            'Montreal Gases|HCFC142b': 0.19,
            'Montreal Gases|HCFC22': 0.21,
            'Montreal Gases|Halon1202': 0.27,
            'Montreal Gases|Halon1211': 0.29,
            'Montreal Gases|Halon1301': 0.30,
            'Montreal Gases|Halon2402': 0.31,
        }

        self.parms['scaling_factor'] = {}
        self.parms['conc_pi'] = {}
        if 'conc_pi' in kw:
            kw = kw.copy()
            self.set_conc_pi(kw.pop('conc_pi'))

        self.parms_update(**kw)

    def c2erf_etminan(self, *args, **kw):
        """
        Return ERF for CO2 concentrations, valid in the range 180-2000 ppm.

        Parameters
        ----------
        args : array-like
            Input values of concentrations in ppm.

        kw : dict, optional
            cn2o : array-like
                N2O concentrations in ppb.

        Returns
        -------
        erf : array-like
            Output values of ERF in W/m2.
        """
        cc = np.hstack(args)
        cn = np.array(kw.get('cn2o', self.parms['cnref']))
        ccref = self.parms['ccref']
        cnref = self.parms['cnref']
        a1 = self.parms['co2_a1']
        b1 = self.parms['co2_b1']
        c1 = self.parms['co2_c1']
        cnbar = 0.5*(cn + cnref)
        ret = (
            a1*(cc-ccref)**2 + b1*np.fabs(cc-ccref) + c1*cnbar + 5.36
        ) * np.log(cc/ccref)
        return ret

    def cm2erf(self, *args, **kw):
        """
        Return ERF for CH4 concentrations, valid in the range 340-3500 ppb.

        Parameters
        ----------
        args : array-like
            Input values of concentrations in ppb.

        kw : dict, optional
            cn2o : array-like
                N2O concentrations in ppb.

        Returns
        -------
        erf : array-like
            Output values of ERF in W/m2.
        """
        cm = np.hstack(args)
        cn = np.array(kw.get('cn2o', self.parms['cnref']))
        cmref = self.parms['cmref']
        cnref = self.parms['cnref']
        a3 = self.parms['ch4_a3']
        b3 = self.parms['ch4_b3']
        cmbar = 0.5*(cm + cmref)
        cnbar = 0.5*(cn + cnref)
        ret = (a3*cmbar + b3*cnbar + 0.043)*(np.sqrt(cm) - np.sqrt(cmref))
        return ret

    def cn2erf(self, *args, **kw):
        """
        Return ERF for N2O concentrations, valid in the range 200-525 ppb.

        Parameters
        ----------
        args : array-like
            Input values of concentrations in ppb.

        kw : dict, optional
            cco2 : array-like
                CO2 concentrations in ppm.
            cch4 : array-like
                CH4 concentrations in ppb.

        Returns
        -------
        erf : array-like
            Output values of ERF in W/m2.
        """
        cn = np.hstack(args)
        cc = np.array(kw.get('cco2', self.parms['ccref']))
        cm = np.array(kw.get('cch4', self.parms['cmref']))
        cnref = self.parms['cnref']
        ccref = self.parms['ccref']
        cmref = self.parms['cmref']
        a2 = self.parms['n2o_a2']
        b2 = self.parms['n2o_b2']
        c2 = self.parms['n2o_c2']
        ccbar = 0.5*(cc + ccref)
        cnbar = 0.5*(cn + cnref)
        cmbar = 0.5*(cm + cmref)
        ret = (
            (a2*ccbar + b2*cnbar + c2*cmbar + 0.117)
            * (np.sqrt(cn) - np.sqrt(cnref))
        )
        return ret

    def set_conc_pi(self, conc_pi):
        """
        Set preindustrial concentrations of greenhouse gases.

        Parameters
        ----------
        conc_pi : dict
            Preindustrial concentrations.
            Keys must be given by a tuple, such as ('CO2', 'ppm').

        """
        self.parms['conc_pi'].update(conc_pi)
        for k1, k2 in [
                (('CO2', 'ppm'), 'ccref'),
                (('CH4', 'ppb'), 'cmref'),
                (('N2O', 'ppb'), 'cnref'),
        ]:
            if k1 in conc_pi:
                self.parms[k2] = conc_pi[k1]


if __name__ == '__main__':
    import os
    import pandas as pd

    indir = '../../untracked/rcmip'
    ver = 'v4-0-0'
    datain = {
        'emissions': f'rcmip-emissions-annual-means-{ver}.csv',
        'concentrations': f'rcmip-concentrations-annual-means-{ver}.csv',
        'forcing': f'rcmip-radiative-forcing-annual-means-{ver}.csv',

    }
    dfin = pd.read_csv(os.path.join(indir, datain['concentrations']))


