"""
API for computing temperature response to forcing changes based on an impulse
response model.
"""

import numpy as np
from .. import MCEExecError
from . import ParmsBase
from . import ModelBase

class IrmParms(ParmsBase):
    def __init__(self, nl=3):
        # CMIP5 mean
        self.add(
            'asj',
            [0.2616, 0.3362, 0.4022] if nl == 3 else [0.5565, 0.4435],
            'Non-dimensional amplitudes in the surface layer',
            'none',
            False,
        )
        self.add(
            'tauj',
            [1.033, 10.48, 260.6] if nl == 3 else [3.799, 218.8],
            'Time constants',
            'yr',
            False,
        )
        self.add(
            'lamb',
            # 1.136 if nl == 3 else 1.102,
            1.053 if nl == 3 else 1.029, # adjusted
            'Climate feedback parameter',
            'W m-2 degC-1',
            False,
        )

class IrmBase(ModelBase):
    def init_process(self, *args, **kw):
        """
        Parameters
        ----------
        args : tuple, optional
            args[0] : int (2 or 3), default 3
                Number of layers in the box model.

        kw : dict, optional
            Keyword arguments to set module parameters.
            Default values are set to CMIP5 25 model means.
            asj : 1-d array
                Non-dimensional amplitudes in the surface layer.
            tauj : 1-d array
                Time constants in year.
            lamb : float
                Climate feedback parameter in W/m2/degC.
        """
        nl = len(args) > 0 and args[0] or 3
        if nl not in [2, 3]:
            mesg = 'invalid number of layers {}'.format(nl)
            self.logger.error(mesg)
            raise MCEExecError(mesg)

        self.parms = IrmParms(nl)
        self.parms.update(**kw)

        self.tkjlast = 0

    def response(self, time, erf, init=True, **kw):
        """
        Numerically compute temperature response to arbitrary forcing changes.

        Response components with different time constants are separately
        computed, and the total response is returned.
        Temperatures in the full layers or in the surface only are computed,
        depending on whether the non-dimensional amplitudes are defined
        for the full layers or not.

        Computing scheme is accurate under the assumption of piecewise-linear
        forcing changes.

        Parameters
        ----------
        time : 1-d array
            Time points in year.

        erf : 1-d array
            Effective radiative forcing in W/m2 at given time points.

        init : bool, optional, default True
            If true, initialize internal parameter `tkjlast` that keeps
            the response components at the end of the time in the last run.
            If false, computing is conducted in a restart mode.

        kw : dict, optional
            Keyword arguments to explicitly set basic parameters for
            `asj`, `tauj`, and `lamb`, instead of using those from the
            corresponding module parameters.
            Full-layer amplitudes 'akj' can be used instead of `asj`.

        Returns
        -------
        tres : 2-D or 1-D array
            Temperature response in degC with (time, layer) dimension.
            If the amplitudes are defined in the surface layer only,
            the layser dimension is squeezed.
        """
        if init:
            self.tkjlast = 0

        parms = self.parms

        time = np.array(time).astype('d')
        erf = np.array(erf).astype('d')

        akj = np.array(kw.get('akj', kw.get('asj', parms.asj))).astype('d')
        tauj = np.array(kw.get('tauj', parms.tauj)).astype('d')
        lamb = np.array(kw.get('lamb', parms.lamb)).astype('d')

        dt = np.nan

        tkj = np.zeros((len(time),) + akj.shape)
        tkj[0] = self.tkjlast

        for i in range(len(time)-1):
            if time[i+1] - time[i] != dt:
                dt = time[i+1] - time[i]
                rd = np.exp(-dt/tauj)
                rs = akj - akj*rd
                rr = akj - rs*tauj/dt

            tkj[i+1] = rd*tkj[i] + rs*erf[i] + rr*(erf[i+1]-erf[i])

        tres = tkj.sum(axis=tkj.ndim-1) / lamb
        self.tkjlast = tkj[-1]

        return tres

    def get_parms_ebm(self, ret_depth=False, **kw):
        """
        Get derived parameters for an equivalent box model.

        Parameters
        ----------
        ret_depth : bool, optional, default False
            If true, heat capacities are converted into equivalent ocean depths.

        kw : dict, optional
            Keyword arguments to explicitly give basic parameter values for
            `asj`, `tauj`, and `lamb`, instead of using those from the
            corresponding module parameters.

        Returns
        -------
        lambk : 1-d array
            Heat exchange coefficients in W/m2/degC.
            The first element is the same as `lamb`.

        xik : 1-d array
            Heat capacities in J/m2/degC divided by annual total seconds.
            if `ret_depth` is true, equivalent ocean depths in meter are
            returned.

        akj : 2-d array
            Non-dimensional amplitudes for the full layers.
            The first layer elements are the same as `asj`.
        """
        asj = np.array(kw.get('asj', self.parms.asj)).astype('d')
        tauj = np.array(kw.get('tauj', self.parms.tauj)).astype('d')
        lamb = np.array(kw.get('lamb', self.parms.lamb)).astype('d')

        nl = len(tauj)
        if nl not in [2, 3]:
            self.logger.error('invalid number of layers {}'.format(nl))
            raise MCEExecError

        xitot = (asj*tauj).sum() * lamb
        xis = lamb / (asj/tauj).sum()

        if nl == 3:
            lamb1 = (asj/(tauj*tauj)).sum() / lamb * xis * xis - lamb
            x2 = tauj.sum() - (xitot/lamb) - (xitot-xis)/lamb1
            xi1 = tauj.prod() * lamb * lamb1 / (xis * x2)
            xi2 = xitot - (xis + xi1)
            lamb2 = xi2 / x2

            lambk = [lamb1, lamb2]
            xik = [xi1, xi2]

            det = tauj[2]/tauj[0] - tauj[1]/tauj[0] \
                + tauj[0]/tauj[1] - tauj[2]/tauj[1] \
                + tauj[1]/tauj[2] - tauj[0]/tauj[2]
            det2 = np.array(
                [tauj[0]/tauj[1] - tauj[0]/tauj[2],
                 tauj[1]/tauj[2] - tauj[1]/tauj[0],
                 tauj[2]/tauj[0] - tauj[2]/tauj[1]] )
            det1 = det2 - np.array(
                [x2/tauj[1] - x2/tauj[2],
                 x2/tauj[2] - x2/tauj[0],
                 x2/tauj[0] - x2/tauj[1]] )
            akj = [det1/det, det2/det]

            # errors are negligible, but ensure accurate normalization
            akj[0] = akj[0] / akj[0].sum()
            akj[1] = akj[1] / akj[1].sum()

        else:
            xi1 = xitot - xis
            lamb1 = xis * xi1 / lamb / tauj.prod()

            lambk = lamb1
            xik = xi1

            akj = np.array([-tauj[0], tauj[1]]) / (tauj[1] - tauj[0])

            # errors are negligible, but ensure accurate normalization
            akj = akj / akj.sum()


        lambk = np.hstack([lamb, lambk])
        xik = np.hstack([xis, xik])
        akj = np.vstack([asj, akj])

        if ret_depth:
            spy = 3.15569e7 # seconds per year
            frac_a = 0.71 # ocean area fraction
            rhow = 1.03e3 # density of sea water in kg/m3
            cpw = 4.18e3 # specific heat of sea water in J/kg/degC
            xik = xik * spy / (rhow * cpw * frac_a)

        return lambk, xik, akj

    def ebm_to_irm(self, lambk, xik):
        """
        Get impulse response parameters from box model parameters.

        Parameters
        ----------
        lambk : 1-d array
            Heat exchange coefficients in W/m2/degC.

        xik : 1-d array
            Heat capacities in J/m2/degC divided by annual total seconds.

        Returns
        -------
        tauj : 1-d array
            Time constants in year.

        akj : 2-d array
            Non-dimensional amplitudes for the full layers.
        """
        nl = len(lambk)

        if nl == 2:
            msyst = np.array([
                [(lambk[0] + lambk[1]) / xik[0], -lambk[1] / xik[0]],
                [-lambk[1] / xik[1], lambk[1] / xik[1]],
            ])
        elif nl == 3:
            msyst = np.array([
                [(lambk[0] + lambk[1]) / xik[0], -lambk[1] / xik[0], 0.],
                [-lambk[1] / xik[1], (lambk[1] + lambk[2]) / xik[1], -lambk[2] / xik[1]],
                [0., -lambk[2] / xik[2], lambk[2] / xik[2]],
            ])
        else:
            mesg = 'invalid number of layers {}'.format(nl)
            self.logger.error(mesg)
            raise MCEExecError(mesg)

        eigval, eigvec = np.linalg.eig(msyst)
        
        tauj = 1./eigval
        akj = np.dot(
            np.diag(1./eigval),
            np.linalg.solve(
                eigvec/eigvec[0, :],
                np.diag(lambk[0] / np.array(xik)),
            ),
        ).T

        return tauj, akj

    def response_ideal(self, time, kind='step', variable='tres', **kw):
        """
        Analytically compute response to step- or ramp-shaped forcing.
        Response is normalized based on the energy balance equation
        N(t)/F = 1 - T(t) / (F/lamb) for step response,
        N(t)/(F1*t) = 1 - T(t) / (F1*t/lamb) for ramp response,
        where t is time, N is downward surface heat flux, F is step forcing,
        F1 is forcing change rate, lamb is climate feedback parameter,
        and T is surface temperature response.

        Parameters
        ----------
        time : 1-d array
            Time points in year.

        kind : str, optional, default 'step'
            Forcing type; step or ramp.

        variable : str, optional, default 'tres'
            Name of the variable to be computed.
            'tres' : `T(t)/(F/lamb)` corresponding to temperature response
            'flux' : `N(t)/F` corresponding to downward surface heat flux
            'heat' : time integration of `N(t)/F` corresponding to accumulated
                heat divided by seconds per year

        kw : dict, optional
            Keyword arguments to explicitly give basic parameter values for
            `asj`, `tauj`, and `lamb`, instead of using those from the
            corresponding module parameters.
            Full-layer amplitudes 'akj' can be used instead of `asj`.
        """
        if variable not in ['tres', 'flux', 'heat']:
            self.logger.error('invalid variable {}'.format(variable))
            raise MCEExecError

        parms = self.parms
        akj = np.array(kw.get('akj', kw.get('asj', parms.asj))).astype('d')
        tauj = np.array(kw.get('tauj', parms.tauj)).astype('d')
        lamb = np.array(kw.get('lamb', parms.lamb)).astype('d')

        isscalar = akj.ndim == 1 and np.isscalar(time)

        time = np.array(time).astype('d')
        time, akj = np.broadcast_arrays(
            time.reshape((-1,) + (1,)*akj.ndim),
            akj.reshape((1,) + akj.shape) )
        if kind == 'ramp':
            # avoid zero division warning
            time = np.where(np.equal(time, 0), np.nan, time)

        ja = akj.ndim - 1

        if variable in ['tres', 'flux']:
            if kind == 'step':
                ret = (akj*np.exp(-time/tauj)).sum(axis=ja)
            else:
                ret = akj*tauj/time*(1-np.exp(-time/tauj))
                # this term is `akj` when time approaches zero
                ret = np.where(np.isnan(ret), akj, ret).sum(axis=ja)

            if variable == 'tres':
                ret = 1 - ret

        else:
            if kind == 'step':
                ret = (akj*tauj*(1-np.exp(-time/tauj))).sum(axis=ja)
            else:
                ret = akj*tauj - akj*tauj*tauj/time*(1-np.exp(-time/tauj))
                # the second term is `akj*tauj` when time approaches zero
                ret = np.where(np.isnan(ret), 0, ret).sum(axis=ja)

        if isscalar:
            ret = ret.item()

        return ret

