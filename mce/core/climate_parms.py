"""
API for diagnosing forcing-response parameters.
"""

import numpy as np
from lmfit import Parameters, minimize
from .forcing import RfCO2
from .climate import IrmBase

class ParmEstimate(object):
    def __init__(self, nl=3):
        """
        Parameters
        ----------
        nl : int, optional, default 3
            Number of box-model layers.
        """
        if nl not in [2, 3]:
            raise ValueError('invalid number of layers {}'.format(nl))

        self.nl = nl
        self.irm = IrmBase(nl)
        self.forcing = RfCO2()

    def initpars(self, af, t, **kw):
        """
        Create Parameters object with transformed amplitudes `af` and time
        constants `t` and additional parameters.

        Parameters
        ----------
        af : array-like
            Amplitudes transformed as follows:
            af[0] = asj[1] / asj[0]
            af[1] = asj[2] / asj[0]

        t : array-like
            Time constants transformed as follows:
            t[0] = tauj[0]
            t[1] = tauj[1] / tauj[0]
            t[2] = tauj[2] / tauj[1]

        kw : dict, optional
            Additional parameters.

        Results
        -------
        px : Parameters
            Parameters object defined with given parameters.
        """
        afmin = 0.01
        txmin = {0: 0.}
        txmin_default = 2.

        px = Parameters()

        for i, afx in enumerate(af):
            px.add('af%d' % (i+1,), value=afx, min=afmin)
        for i, tx in enumerate(t):
            px.add('t%d' % i, value=tx, min=txmin.get(i, txmin_default))
        for k, v in kw.items():
            px.add(k, value=v)

        return px

    def get_a_tau(self, px):
        """
        Retrieve amplitudes `asj` and time constants `tauj`.

        Parameters
        ----------
        px : Parameters
            Parameters object.

        Returns
        -------
        (asj, tauj) : tuple
        """
        pvals = list(px.valuesdict().items())
        nb = self.nl -1
        b = np.array([v for k, v in pvals[:nb]])
        asj = np.zeros(nb+1) + 1./(1.+b.sum())
        asj[1:] *= b
        tauj = np.array([v for k, v in pvals[nb:nb+nb+1]]).cumprod()
        return asj, tauj

    def irm_wrap(self, time, **kw):
        """
        Compute abrupt-4xCO2 and 1pctCO2 response.

        Parameters
        ----------
        time : 1-D array
            Time points in year.

        kw : dict, optional
            Forcing and response parameters.

        Returns
        -------
        results : list
            list[0] = rndt4x
            list[1] = ts4x
            list[2] = rndt1p
            list[3] = ts1p
        """
        forcing = self.forcing
        irm = self.irm

        forcing.parms.update(**{
            k: kw[k] for k in forcing.parms() if k in kw
        })
        irm.parms.update(**{
            k: kw[k] for k in irm.parms() if k in kw
        })

        len4x = kw.get('len4x', 150)
        len1p = kw.get('len1p', 140)
        time4x = time[:len4x]
        time1p = np.hstack([0., time[:len1p]])

        f4x = forcing.x2erf(4)
        rndt4x = irm.response_ideal(time4x, variable='flux')
        ts4x = 1 - rndt4x
        rndt4x = rndt4x * f4x
        ts4x = ts4x * f4x / irm.parms.lamb

        f1p = forcing.xl2erf(time1p * np.log(1.01))
        ts1p = irm.response(time1p, f1p)
        rndt1p = f1p - irm.parms.lamb * ts1p

        results = [rndt4x, ts4x, rndt1p[1:], ts1p[1:]]

        return results

    def residual(self, px):
        time = self.datain['time']
        gcm = self.datain['gcm']
        std = self.datain['std']

        asj, tauj = self.get_a_tau(px)
        lamb = px['lamb'].value
        ts4xeq = px['ts4xeq'].value
        beta = px['beta'].value
        alpha = ts4xeq * lamb / ( beta * np.log(4) )

        irm = self.irm_wrap(
            time, asj=asj, tauj=tauj, lamb=lamb, alpha=alpha, beta=beta)

        # `std` is a list of two elements
        res = [ (x1 - x2) / w1 for x1, x2, w1 in zip(gcm, irm, std*2) ]

        return np.hstack(res)

    def minimize_wrap(self, time, gcm, std):
        """
        Wrapper method to estimate impulse response model parameters.

        Parameters
        ----------
        time : 1-D array
            Time points in year.

        gcm : list of four 1-D arrays
            [`rtnt_4x`, `tas_4x`, `rtnt_1p`, `tas_1p`]

        std : list of two floats
            Standard deviation of rtnt and tas of piControl.

        Returns
        -------
        alpha, beta, lamb, asj, tauj
            Estimated parameters.

        Notes
        -----
        Return object from `minimize()` is stored as `self.ret_minimize`.
        """
        self.datain = {
            'time': time,
            'gcm': gcm,
            'std': std,
        }

        if self.nl == 3:
            par_args = [[1., 1.], [1., 10., 20.]]
        else:
            par_args = [[1.], [2., 100.]]

        par_kw = dict(lamb=1., ts4xeq=7., beta=1.)
        px = self.initpars(*par_args, **par_kw)

        ret = minimize(self.residual, px)

        asj, tauj = self.get_a_tau(ret.params)
        lamb = ret.params['lamb'].value
        ts4xeq = ret.params['ts4xeq'].value
        beta = ret.params['beta'].value
        alpha = ts4xeq * lamb / beta / np.log(4)

        self.ret_minimize = ret

        return alpha, beta, lamb, asj, tauj
