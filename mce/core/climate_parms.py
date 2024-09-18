"""
API for diagnosing forcing-response parameters.
"""

import numpy as np
import pandas as pd
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


def get_rwf_ramp(dfin, tp):
    """Get realized warming fractions
    when the forcing increases linearly

    Parameters
    ----------
    df
        Input parameters in dict, Series, or DataFrame

    Returns
    -------
        Realized warming fractions in the same format as in the input
    """
    if isinstance(dfin, (dict, pd.Series)):
        df = pd.DataFrame([dfin])
    else:
        df = dfin

    nl = 3 if 'tau2' in df else 2

    tp = np.hstack([tp])
    asj = df[[f'a{j}' for j in range(nl)]].values[:, :, None]
    tauj = df[[f'tau{j}' for j in range(nl)]].values[:, :, None]

    ret = pd.DataFrame(
        1. -
        (asj * tauj * (1 - np.exp(-tp[None, None, :]/tauj)))
        .sum(axis=1)
        / tp[None, :],
        index=df.index, columns=tp,
    )

    # if len(tp) == 1:
    #     rwf = rwf.squeeze(axis=1).rename(None)
    if isinstance(dfin, dict):
        ret = ret.squeeze(axis=0).to_dict()
    elif isinstance(dfin, pd.Series):
        ret = ret.squeeze(axis=0).rename(dfin.name)

    return ret


def get_amp_full(dfin):
    """Derive sub-surface amplitude parameters
    from impulse response parameters based on the Laplace transform
    of the energy balance equations

    Parameters
    ----------
    df
        Input parameters in dict, Series, or DataFrame

    Returns
    -------
        Derived parameters in the same format as in the input
    """
    if isinstance(dfin, (dict, pd.Series)):
        df = pd.DataFrame([dfin])
    else:
        df = dfin

    nl = 3 if 'tau2' in df else 2

    asj = df[[f'a{j}' for j in range(nl)]].values
    tauj = df[[f'tau{j}' for j in range(nl)]].values
    lamb = df['lambda'].values

    xitot = (asj*tauj).sum(axis=1) * lamb
    xis = lamb / (asj/tauj).sum(axis=1)

    if nl == 2:
        a1j = (
            np.array([-tauj[:, 0], tauj[:, 1]])
            / (tauj[:, 1] - tauj[:, 0])
        ).T
        # errors are negligible, but ensure accurate normalization
        a1j = a1j / a1j.sum(axis=1)[:, None]

        ret = pd.DataFrame({
            'a10': a1j[:, 0],
            'a11': a1j[:, 1],
        }, index=df.index)

    else:
        lamb1 = (asj/(tauj*tauj)).sum(axis=1) / lamb * xis * xis - lamb
        x2 = tauj.sum(axis=1) - (xitot/lamb) - (xitot-xis)/lamb1

        det = tauj[:, 2]/tauj[:, 0] - tauj[:, 1]/tauj[:, 0] \
            + tauj[:, 0]/tauj[:, 1] - tauj[:, 2]/tauj[:, 1] \
            + tauj[:, 1]/tauj[:, 2] - tauj[:, 0]/tauj[:, 2]
        det2 = np.array(
            [tauj[:, 0]/tauj[:, 1] - tauj[:, 0]/tauj[:, 2],
                tauj[:, 1]/tauj[:, 2] - tauj[:, 1]/tauj[:, 0],
                tauj[:, 2]/tauj[:, 0] - tauj[:, 2]/tauj[:, 1]] )
        det1 = det2 - np.array(
            [x2/tauj[:, 1] - x2/tauj[:, 2],
                x2/tauj[:, 2] - x2/tauj[:, 0],
                x2/tauj[:, 0] - x2/tauj[:, 1]] )
        a1j = (det1/det).T
        a2j = (det2/det).T

        # errors are negligible, but ensure accurate normalization
        a1j = a1j / a1j.sum(axis=1)[:, None]
        a2j = a2j / a2j.sum(axis=1)[:, None]

        ret = pd.DataFrame({
            'a10': a1j[:, 0],
            'a11': a1j[:, 1],
            'a12': a1j[:, 2],
            'a20': a2j[:, 0],
            'a21': a2j[:, 1],
            'a22': a2j[:, 2],
        }, index=df.index)

    if isinstance(dfin, dict):
        ret = ret.squeeze().to_dict()
    elif isinstance(dfin, pd.Series):
        ret = ret.squeeze().rename(dfin.name)

    return ret


def get_ebm(dfin):
    """Derive heat transfer coefficients and heat capacities
    from impulse response parameters based on the Laplace transform
    of the energy balance equations

    Parameters
    ----------
    df
        Input parameters in dict, Series, or DataFrame

    Returns
    -------
        Derived parameters in the same format as in the input
    """
    if isinstance(dfin, (dict, pd.Series)):
        df = pd.DataFrame([dfin])
    else:
        df = dfin

    nl = 3 if 'tau2' in df else 2

    asj = df[[f'a{j}' for j in range(nl)]].values
    tauj = df[[f'tau{j}' for j in range(nl)]].values
    lamb = df['lambda'].values

    xitot = (asj*tauj).sum(axis=1) * lamb
    xis = lamb / (asj/tauj).sum(axis=1)

    if nl == 2:
        xi1 = xitot - xis
        lamb1 = xis * xi1 / lamb / tauj.prod(axis=1)

        ret = pd.DataFrame({
            'gamma1': lamb1,
            'xis': xis,
            'xi1': xi1,
        }, index=df.index)

    else:
        lamb1 = (asj/(tauj*tauj)).sum(axis=1) / lamb * xis * xis - lamb
        x2 = tauj.sum(axis=1) - (xitot/lamb) - (xitot-xis)/lamb1
        xi1 = tauj.prod(axis=1) * lamb * lamb1 / (xis * x2)
        xi2 = xitot - (xis + xi1)
        lamb2 = xi2 / x2

        ret = pd.DataFrame({
            'gamma1': lamb1,
            'gamma2': lamb2,
            'xis': xis,
            'xi1': xi1,
            'xi2': xi2,
        }, index=df.index)

    if isinstance(dfin, dict):
        ret = ret.squeeze().to_dict()
    elif isinstance(dfin, pd.Series):
        ret = ret.squeeze().rename(dfin.name)

    return ret


def ebm2irm(dfin):
    """Derive impulse response parameters from energy balance model parameters

    Parameters
    ----------
    dfin
        Input parameters in dict, Series, or DataFrame

    Returns
    -------
        Derived parameters in the same format as in the input
    """
    if isinstance(dfin, (dict, pd.Series)):
        df = pd.DataFrame([dfin])
    else:
        df = dfin

    if 'gamma2' in df:
        nl = 3
        zeros = pd.Series(0., index=df.index)
        msyst = np.array([
            [
                (df['lambda'] + df['gamma1']) / df['xis'],
                - df['gamma1'] / df['xis'],
                zeros,
            ],
            [
                - df['gamma1'] / df['xi1'],
                (df['gamma1'] + df['gamma2']) / df['xi1'],
                - df['gamma2'] / df['xi1'],
            ],
            [
                zeros,
                - df['gamma2'] / df['xi2'],
                df['gamma2'] / df['xi2'],
            ]
        ])
        xik = df[['xis', 'xi1', 'xi2']].values
    else:
        nl = 2
        msyst = np.array([
            [
                (df['lambda'] + df['gamma1']) / df['xis'],
                - df['gamma1'] / df['xis'],
            ],
            [
                - df['gamma1'] / df['xi1'],
                df['gamma1'] / df['xi1'],
            ],
        ])
        xik = df[['xis', 'xi1']].values

    eigval, eigvec = np.linalg.eig(msyst.transpose(2, 0, 1))
    tauj = 1. / eigval

    def mkdiag(xin):
        """Construct diagonal matrixes
        """
        ndim = xin.shape[1]
        x = np.zeros(xin.shape + (ndim,))
        for i in range(ndim):
            x[:, i, i] = xin[:, i]
        return x

    akj = np.linalg.solve(
        eigvec[:, :, :] / eigvec[:, 0:1, :],
        mkdiag(df['lambda'].values[:, None] / xik),
    )
    akj = np.stack([
        akj[:, j] / eigval[:, j, None] for j in range(nl)
    ]).transpose(1, 2, 0)

    ret = pd.concat(
        [
            pd.DataFrame(
                tauj,
                index=df.index,
                columns=[f'tau{j}' for j in range(nl)],
            )
        ]
        +
        [
            pd.DataFrame(
                akj[:, k],
                index=df.index,
                columns=[f'a{k}{j}' for j in range(nl)],
            )
            for k in range(akj.shape[1])
        ],
        axis=1,
    ).rename(columns=lambda x: x.replace('a0', 'a'))

    if isinstance(dfin, dict):
        ret = ret.squeeze().to_dict()
    elif isinstance(dfin, pd.Series):
        ret = ret.squeeze().rename(dfin.name)

    return ret
