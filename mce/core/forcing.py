"""
API for computing effective radiative forcing (ERF) of CO2.
Non-CO2 forcing agents have not supported yet.
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
            ix = erf > f2x
            erf[ix] = (beta-1) * (erf[ix]-2*f2x) * (2/f2x*erf[ix]-1) \
                + beta*erf[ix]

        if len(erf) == 1:
            erf = erf.item()

        return erf

