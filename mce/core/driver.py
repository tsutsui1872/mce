"""
API for model drivers.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from mce import MCEExecError, get_logger
from mce.core.forcing import RfAll as RfAllBase
from mce.core.climate import IrmBase
from mce.core.carbon import ModelOcean
from mce.core.carbon import ModelLand
from collections import namedtuple

class RfAll(RfAllBase):
    def ghg_concentrations_to_forcing(self, dfin, co2_method='c2erf'):
        """
        Convert GHG concentrations to forcing.

        Parameters
        ----------
        dfin : pandas.DataFrame
            Input concentration data.

        co2_method : str, optional, default 'c2erf'
            Choice of CO2 forcing scheme.

        Returns
        -------
        df : pandas.DataFrame
            Output forcing data.
        """
        kw_co2 = {}
        if co2_method == 'c2erf_etminan':
            kw_co2 = {'cn2o': dfin.loc[('N2O', 'ppb')]}

        df = [
            dfin.loc[[('CO2', 'ppm')]].transform(
                getattr(self, co2_method), axis=1, **kw_co2),
            dfin.loc[[('CH4', 'ppb')]].transform(
                self.cm2erf, axis=1,
                cn2o=dfin.loc[('N2O', 'ppb')]),
            dfin.loc[[('N2O', 'ppb')]].transform(
                self.cn2erf, axis=1,
                cco2=dfin.loc[('CO2', 'ppm')],
                cch4=dfin.loc[('CH4', 'ppb')]),
        ]

        eff = self.parms['efficiency']
        fac = {'ppt': 1e-3}
        # Efficiency units: W/m^2/ppb

        for name in ['Montreal Gases', 'F-Gases']:
            df.append(
                dfin
                [[x.startswith(name)
                  for x in dfin.index.get_level_values('Variable')]]
                .transform(
                    lambda xa:
                    (xa - self.parms['conc_pi'].get((xa.name[0], 'ppt'), 0.))
                    * eff[xa.name[0]] * fac[xa.name[1]], axis=1)
            )

        df = pd.concat(df)
        df = df.groupby(
            [x.split('|')[0] for x in df.index.get_level_values('Variable')]
        ).sum()

        return df


def ndiff_l3(f, h, f0e=None, f1e=None):
    """
    Calculate numerical differentiation at each data point
    with 3-point Lagrange interpolation.
    Data points are spaced with a fixed interval.

    Parameters
    ----------
    f : 1-D array_like
        Function values.

    h : float
        Data point interval.

    f0e : float, optional, default None
        Function value at the point shifted by -h/2 from the first point.
        If None, forward differentiation is used at the first point.

    f1e : float, optional, default None
        Function value at the point shifted by +h/2 from the last point.
        If None, backward differentiation is used at the last point.

    Returns
    fp : 1-D numpy.ndarray
        First derivative at the data points.
    """
    f = np.array(f)
    fp = np.zeros(len(f))

    if f0e is None:
        fp[0] = (-3*f[0] + 4*f[1] - f[2]) / (2*h)
    else:
        fp[0] = (-4*f0e + 3*f[0] + f[1]) / (3*h)

    fp[1:-1] = (f[2:] - f[:-2]) / (2*h)

    if f1e is None:
        fp[-1] = (f[-3] - 4*f[-2] + 3*f[-1]) / (2*h)
    else:
        fp[-1] = (-f[-2] - 3*f[-1] + 4*f1e) / (3*h)

    return fp


class Driver:
    """
    Driver of climate and carbon-cycle equation systems.

    The following output time series variables are defined:

    abf : Airborne fraction of cumulative CO2 emissions
    catm : Excess carbon in the atmosphere in GtC
    cbsf : Atmosphere-to-land CO2 flux in GtC/yr
    cbst : Accumulated carbon over land in GtC
    cco2 : Atmospheric CO2 concentration in ppm
    cocf : Atmosphere-to-ocean CO2 flux in GtC/yr
    coct : Accumulated carbon over ocean in GtC
    ctot : Cumulative CO2 emissions in GtC
    eco2 : CO2 emissions in GtC/yr
    erf : Total effective radiative forcing in W/m^2
    rtnt : Total heat uptake in W/m^2
    tak : Temperature change in degC
    tas : Surface temperature change in degC
    tcre : Transient climate response to cumulative CO2 emissions
        in degC/1000 GtC
    thc : Total heat content change in J/spy/m^2 (spy=seconds per year)

    Components of effective radiative forcing are also defined,
    such as 'erf|CO2'

    Parameters
    ----------
    time : 1-D array_like
        Time points of input cco2/eco2/erf_nonco2 data.
        Not necessarily equally spaced, but time integration is evaluated
        at equally spaced points with dt interval over the range.

    cco2 : 1-D array_like, default None
        Input CO2 concentrations in ppm.
        If not None, the model is driven by the concentrations,
        otherwise driven by the emissions.

    eco2 : 1-D array_like, default None
        Input CO2 emissions in GtC.
        When cco2 is None, eco2 should be given.

    erf_nonco2 : 1-D array_like, default None
        Input non-CO2 effective radiative forcing in W/m^2.

    dt : numeric, default 1
        Time interval.

    df_ghg : pandas.DataFrame
        Non-CO2 well-mixed GHG concentrations.
        Must be defined at the equally-spaced time points.

    df_erf_other : pandas.DataFrame
        Other agents' forcing.
        Must be defined at the equally-spaced time points.

    kw_irm : dict, default {}
        Keyword arguments passed to IrmBase().

    kw_rfall : dict, default {}
        Keyword arguments passed to RfAll().

    kw_ocean : dict, default {}
        Keyword arguments passed to ModelOcean().
        If None, carbon cycle is not activated

    kw_land : dict, default {}
        Keyword arguments passed to ModelLand().
        If None, land carbon cycle is not activated.

    """
    def __init__(
            self, time, cco2=None, eco2=None, erf_nonco2=None,
            dt=1, df_ghg=None, df_erf_other=None,
            kw_irm={}, kw_rfall={}, kw_ocean={}, kw_land={}):

        self.logger = get_logger('mce')
        if cco2 is None and eco2 is None:
            self.logger.error('cco2 or eco2 should be given')
            raise MCEExecError

        self.irm = IrmBase(**kw_irm)
        self.forcing = RfAll(**kw_rfall)
        if kw_ocean is not None:
            self.ocean = ModelOcean(**kw_ocean)
        else:
            self.ocean = None
        if kw_land is not None:
            self.land = ModelLand(**kw_land)
        else:
            self.land = None

        if not hasattr(time, 'dtype'):
            time = np.array(time, dtype=np.result_type(*time))

        self.time = np.arange(
            time[0], time[0]+(int((time[-1]-time[0])/dt)+1)*dt, dt)
        self.dt = dt
        self.interp = {}

        if cco2 is not None:
            cco2 = np.array(cco2)
            self.interp['cco2'] = interp1d(time, cco2)
            self.cco2 = self.interp['cco2'](self.time)
        elif eco2 is not None:
            eco2 = np.array(eco2)
            self.interp['eco2'] = interp1d(time, eco2)
            self.eco2 = self.interp['eco2'](self.time)

        self.df_erf = None

        if erf_nonco2 is not None:
            erf_nonco2 = np.array(erf_nonco2)
            self.interp['erf_nonco2'] = interp1d(time, erf_nonco2)
            self.erf_nonco2 = self.interp['erf_nonco2'](self.time)

        elif df_ghg is not None and df_erf_other is not None:
            sca = self.forcing.parms['scaling_factor']
            self.df_erf = pd.concat(
                [self.forcing.ghg_concentrations_to_forcing(df_ghg)
                 .transform(lambda x: x * sca.get(x.name, 1), axis=1),
                 df_erf_other
                 .transform(lambda x: x * sca.get(x.name, 1), axis=1)])
            df1 = self.df_erf.drop('CO2').sum()
            self.interp['erf_nonco2'] = interp1d(df1.index.values, df1.values)
            self.erf_nonco2 = df1.loc[self.time].values

        lambk, xik, akj = self.irm.get_parms_ebm()
        self.irm_lamb = lambk[0]
        self.irm_akj = akj

        self.MCEVariable = namedtuple('MCEVariable', ['long_name', 'units'])
        self.variables = {
            x[0]: self.MCEVariable(*x[1:]) for x in [
                ('abf', 'Airborne fraction of cumulative CO2 emissions',
                 'dimensionless'),
                ('catm', 'Excess carbon in the atmosphere', 'GtC'),
                ('cbsf', 'Atmosphere-to-land CO2 flux', 'GtC/yr'),
                ('cbst', 'Accumulated carbon over land', 'GtC'),
                ('cco2', 'Atmospheric CO2 concentration', 'ppm'),
                ('cocf', 'Atmosphere-to-ocean CO2 flux', 'GtC/yr'),
                ('coct', 'Accumulated carbon over ocean', 'GtC'),
                ('ctot', 'Cumulative CO2 emissions', 'GtC'),
                ('eco2', 'CO2 emissions', 'GtC/yr'),
                ('erf', 'Total effective radiative forcing', 'W/m^2'),
                ('erf|CO2', 'Effective radiative forcing of CO2', 'W/m^2'),
                ('rtnt', 'Total heat uptake', 'W/m^2'),
                ('tak', 'Temperature change', 'degC'),
                ('tas', 'Surface temperature change', 'degC'), # =tak[0]
                ('tcre',
                 'Transient climate response to cumulative CO2 emissions',
                 'degC/1000 GtC'),
                ('thc', 'Total heat content change', 'J/spy/m^2'),
                # spy=seconds per year
            ]
        }
        if self.df_erf is not None:
            for name in self.df_erf.index:
                if name == 'CO2':
                    continue
                self.variables[f'erf|{name}'] = self.MCEVariable(
                    f'Effective radiative forcing of {name}', 'W/m^2'
                )

    def _func_cdrv(self, t, y):
        """
        Returns time derivatives of Y(t) in a concentration-driven
        equation system.

        Parameters
        ----------
        t : scalar
            Time point.

        y : 1-D numpy.ndarray
            Y(t) in the equation system.

        Returns
        -------
        ydot : 1-D numpy.ndarray
            Y'(t) in the equation system..
        """
        lamb = self.irm_lamb
        akj = self.irm_akj
        tauj = self.irm.parms['tauj']

        ocean = self.ocean
        land = self.land

        cco2 = self.interp['cco2'](t)
        erf = self.forcing.c2erf(cco2)
        if 'erf_nonco2' in self.interp:
            erf = erf + self.interp['erf_nonco2'](t)

        nl = len(tauj)
        takj = np.array(y[:nl*nl].reshape((nl, nl)))
        ydot = [(erf/lamb * akj / tauj - takj / tauj).ravel()]

        if ocean is not None:
            n0 = nl*nl
            tas = takj[0, :].sum()
            catm = (cco2-ocean.parms['cco2_pi'])/ocean.parms['alphaa']

            coc = np.zeros(4)
            coc[1:] = y[n0:n0+3]
            hls = ocean.parms['hls']
            coc[0] = ocean.partit_inv(catm, hls, tas) - catm
            ydot.append(-np.dot(ocean.msyst[1:,:], coc))

            if land is not None:
                n0 = n0 + 3
                cbs = np.array(y[n0:n0+4])
                fert = land.fertmm(catm)
                tau_l = land.get_tau(tas)
                ydot.append(-cbs/tau_l + fert*land.parms['amp']*tau_l)

        return np.hstack(ydot)

    def run_irm(self, **kw):
        """
        Analytically solves an impulse response system of climate module.

        Parameters
        ----------
        kw : dict, optional, default {}
            Pre-industrial CO2 concentration in ppm and non-CO2 forcing
            in W/m^2 can be given explicitly for a special purpose.

        Returns
        -------
        ret : dict
            Resulting time series of each variable.
        """
        cco2_pi = kw.get('cco2_pi', self.forcing.parms['ccref'])
        erf_nonco2_pi = kw.get('erf_nonco2_pi', 0.)
        erf_pi = self.forcing.c2erf(cco2_pi) + erf_nonco2_pi

        time = self.time
        irm = self.irm
        lambk, xik, akj = irm.get_parms_ebm()

        cco2 = self.interp['cco2'](time)
        erf = self.forcing.c2erf(cco2)
        if hasattr(self, 'erf_nonco2'):
            erf += self.erf_nonco2

        if erf_pi != 0.:
            irm.tkjlast = erf_pi / irm.parms['lamb'] * akj
            init = False
        else:
            init = True

        tak = irm.response(time, erf, init=init, akj=akj).transpose()
        ret = {'cco2': cco2, 'erf': erf, 'tak': tak}

        return ret

    def run_cdrv(self, **kw):
        """
        Performs a concentration-driven run.

        Parameters
        ----------
        kw : dict, optional, default {}
            Pre-industrial CO2 concentration in ppm and non-CO2 forcing
            in W/m^2 can be given explicitly.

        Returns
        -------
        ret : dict
            Resulting time series of each variable.
        """
        cco2_pi = kw.get('cco2_pi', self.forcing.parms['ccref'])
        cco2_pi_ocean = kw.get('cco2_pi', self.ocean.parms['cco2_pi'])
        erf_nonco2_pi = kw.get('erf_nonco2_pi', 0.)
        erf_pi = self.forcing.c2erf(cco2_pi) + erf_nonco2_pi

        time = self.time
        cco2 = self.cco2
        irm = self.irm
        ocean = self.ocean
        land = self.land

        nl = len(irm.parms['tauj'])
        lambk, xik, akj = irm.get_parms_ebm()
        takj_pi = erf_pi / irm.parms['lamb'] * akj
        y0 = [takj_pi.ravel()]

        if ocean is not None:
            tas_pi = takj_pi.sum(axis=1)[0]
            catm_pi = (
                cco2_pi_ocean - ocean.parms['cco2_pi']
            ) / ocean.parms['alphaa']

            hls = ocean.parms['hls']
            coc0_pi = ocean.partit_inv(catm_pi, hls, tas_pi)
            coc_pi = np.array(
                [ocean.parms['hl1'], ocean.parms['hl2'], ocean.parms['hl3']]
            ) / hls * (coc0_pi - catm_pi)
            ctot_pi = coc0_pi + coc_pi.sum()
            coct_pi =  ctot_pi - catm_pi
            y0.append(coc_pi)

            if land is not None:
                cbs_pi0 = (
                    land.parms['fnpp0'] * land.parms['amp']
                    * land.parms['tau']**2
                )
                fert = land.fertmm(catm_pi)
                tau_l = land.get_tau(tas_pi)
                cbs_pi = fert * land.parms['amp'] * tau_l**2
                ctot_pi = ctot_pi + (cbs_pi - cbs_pi0).sum()
                y0.append(cbs_pi)

        y0 = np.hstack(y0)
        t_span = (time[0], time[-1])
        sol = solve_ivp(
            self._func_cdrv, t_span, y0, t_eval=time,
            max_step=1., first_step=1.)
        sol_y = sol.y

        tak = sol_y[:nl*nl, :].reshape((nl, nl, -1)).sum(axis=1)
        # - takj_pi.sum(axis=1).reshape((nl, -1))

        ret = {'cco2': cco2, 'tak': tak}

        if ocean is not None:
            n0 = nl*nl
            ydot = [
                self._func_cdrv(time[i], sol_y[:, i])
                for i in range(len(time))]
            ydot = np.array(ydot)

            catm = (cco2 - ocean.parms['cco2_pi']) / ocean.parms['alphaa']
            coc = np.array(
                [ocean.partit_inv(catm[i], hls, tak[0, i])
                 for i in range(len(catm))])
            coc = np.vstack([coc - catm, sol_y[n0:n0+3, :]])
            coct = coc.sum(axis=0)
            ctot = catm + coct

            if land is not None:
                n0 = n0 + 3
                cbs = sol_y[n0:n0+4, :] - cbs_pi0.reshape((4, 1))
                cbsf = ydot[:, n0:n0+4].sum(axis=1)
                cbst = cbs.sum(axis=0)
                ctot = ctot + cbst
                ret.update(cbst=cbst, cbsf=cbsf)

            eco2 = ndiff_l3(ctot, self.dt, f0e=ctot_pi)
            cocf = ndiff_l3(coct, self.dt, f0e=coct_pi)

            ret.update(eco2=eco2, catm=catm, coct=coct, cocf=cocf, ctot=ctot)

        return ret

    def _func_edrv(self, t, y):
        """
        Returns time derivatives of Y(t) in a emission-driven
        equation system.

        Parameters
        ----------
        t : scalar
            Time point.

        y : 1-D numpy.ndarray
            Y(t) in the equation system.

        Returns
        -------
        ydot : 1-D numpy.ndarray
            Y'(t) in the equation system..
        """
        lamb = self.irm_lamb
        akj = self.irm_akj
        tauj = self.irm.parms['tauj']

        ocean = self.ocean
        land = self.land

        eco2 = self.interp['eco2'](t)
        if 'erf_nonco2' in self.interp:
            erf_nonco2 = self.interp['erf_nonco2'](t)
        else:
            erf_nonco2 = 0.

        nl = len(tauj)
        takj = np.array(y[:nl*nl].reshape((nl, nl)))
        n0 = nl*nl
        coc = np.array(y[n0:n0+4])
        if self.land is not None:
            n0 = n0 + 4
            cbs = np.array(y[n0:n0+4])

        tas = takj[0, :].sum()
        hls = ocean.parms['hls']

        catm = ocean.partit(coc[0], hls, tas)
        coc[0] -= catm
        cocdot = -np.dot(ocean.msyst, coc)
        cocdot[0] += eco2

        if self.land is not None:
            fert = land.fertmm(catm)
            tau_l = land.get_tau(tas)
            cbsdot = -cbs/tau_l + fert*land.parms['amp']*tau_l
            cocdot[0] -= cbsdot.sum()

        cco2 = catm * ocean.parms['alphaa'] + ocean.parms['cco2_pi']
        erf = self.forcing.c2erf(cco2) + erf_nonco2

        ydot = [(erf/lamb * akj / tauj - takj / tauj).ravel(), cocdot]
        if self.land is not None:
            ydot.append( cbsdot )

        return np.hstack(ydot)


    def run_edrv(self, frac_init_eco2=0.5, **kw):
        """
        Performs an emission-driven run.

        Parameters
        ----------
        frac_init_eco2: float, optional, default 0.5
            Fraction of time step for which initial emissions are added.

        kw : dict, optional, default {}
            Pre-industrial non-CO2 forcing in W/m^2 can be given explicitly.

        Returns
        -------
        ret : dict
            Resulting time series of each variable.
        """
        erf_nonco2_pi = kw.get('erf_nonco2_pi', 0.)

        time = self.time
        eco2 = self.eco2
        irm = self.irm
        ocean = self.ocean

        nl = len(irm.parms['tauj'])
        lambk, xik, akj = irm.get_parms_ebm()
        takj_pi = np.full_like(akj, 0.)

        hls = ocean.parms['hls']
        coc_pi = np.zeros(4)
        coct_pi = 0.

        y0 = np.hstack([takj_pi.ravel(), coc_pi])

        if self.land is not None:
            lparms = self.land.parms
            cbs_pi0 = lparms['fnpp0'] * lparms['amp'] * lparms['tau']**2
            y0 = np.hstack([y0, cbs_pi0])

        if erf_nonco2_pi != 0.:
            # Correction for accumulated carbon over ocean may be needed
            # for consistency
            interp_eco2 = self.interp['eco2']
            interp_erf_nonco2 = self.interp['erf_nonco2']

            t_span = (0., 10000.)
            self.interp['eco2'] = interp1d(np.array(t_span), np.zeros(2))
            self.interp['erf_nonco2'] = interp1d(
                np.array(t_span), np.array([erf_nonco2_pi, erf_nonco2_pi]))

            sol = solve_ivp(self._func_edrv, t_span, y0)
            y0 = sol.y[:, -1]

            tak_pi = y0[:nl*nl].reshape((nl, nl)).sum(axis=1)
            tas_pi = tak_pi[0]
            n0 = nl*nl
            coc_pi = y0[n0:n0+4]
            if self.land is not None:
                n0 = n0 + 4
                cbs_pi = y0[n0:n0+4] - cbs_pi0

            catm_pi = ocean.partit(coc_pi[0], hls, tas_pi)
            coct_pi = coc_pi.sum() - catm_pi

            self.interp['eco2'] = interp_eco2
            self.interp['erf_nonco2'] = interp_erf_nonco2

        y0[nl*nl] += frac_init_eco2 * self.interp['eco2'](time[0])

        t_span = (time[0], time[-1])
        sol = solve_ivp(self._func_edrv, t_span, y0, t_eval=time, max_step=1.)
        sol_y = sol.y
        nt = sol_y.shape[1]
        ydot = [self._func_edrv(time[i], sol_y[:, i]) for i in range(nt)]
        ydot = np.array(ydot)

        tak = sol_y[:nl*nl, :].reshape((nl, nl, -1)).sum(axis=1)
        n0 = nl*nl

        coc = sol_y[n0:n0+4, :]

        if self.land is not None:
            n0 = n0 + 4
            cbs = sol_y[n0:n0+4, :] - cbs_pi0.reshape((4, 1))
            cbsf = ydot[:, n0:n0+4].sum(axis=1)

        catm = np.array(
            [ocean.partit(coc[0, i], hls, tak[0, i]) for i in range(nt)])
        coc[0, :] -= catm
        cco2 = catm * ocean.parms['alphaa'] + ocean.parms['cco2_pi']

        coct = coc.sum(axis=0)
        cocf = ndiff_l3(coct, self.dt, f0e=coct_pi)

        if self.land is not None:
            cbst = cbs.sum(axis=0)
            ctot = catm + coct + cbst
        else:
            ctot = catm + coct

        ret = {
            'cco2': cco2,
            'catm': catm,
            'coct': coct,
            'ctot': ctot,
            'cocf': cocf,
            'tak': tak,
        }
        if self.land is not None:
            ret.update(cbst=cbst, cbsf=cbsf)

        if len(eco2) == len(time):
            ret['eco2'] = eco2
        else:
            ret['eco2'] = self.interp['eco2'](time)

        return ret

    def run(self, outform=None, irm_only=False, **kw):
        """
        Wrapper method for run_irm(), run_cdrv(), and run_edrv().
        Several output variables are added, and resulting time series can be
        returned with a DataFrame variable when its specifications are given.

        Parameters
        ----------
        outform : dict, optional, default None
            Contains 'variable', 'variable_derived', and 'conversion' lists.
            See _mkout method document.

        irm_only : bool, optional, default False
            When True, run_irm() is performed.

        kw : dict, optional
            Keyward arguments passed to each run method.

        Returns
        -------
        ret/df : dict/pandas.DataFrame
            Resulting time series of each variable.
            When outform is not None, df is returned, and ret is saved
            as an instance variable.
        """
        if irm_only:
            ret = self.run_irm(**kw)
        else:
            if hasattr(self, 'cco2'):
                ret = self.run_cdrv(**kw)
            else:
                ret = self.run_edrv(**kw)

        erf = self.forcing.c2erf(ret['cco2'])
        if hasattr(self, 'erf_nonco2'):
            erf += self.erf_nonco2

        lambk, xik, akj = self.irm.get_parms_ebm()
        thc = (ret['tak'] * xik.reshape((-1, 1))).sum(axis=0)
        tas = ret['tak'][0, :]
        rtnt = erf - lambk[0] * tas

        ret.update(erf=erf, tas=tas, thc=thc, rtnt=rtnt)

        if self.ocean is not None and not irm_only:
            abf = np.divide(
                ret['catm'], ret['ctot'],
                out=np.full_like(ret['catm'], np.nan),
                where=np.equal(np.isclose(ret['ctot'], 0), False))
            tcre = np.divide(
                tas, ret['ctot'], out=np.full_like(tas, np.nan),
                where=np.equal(np.isclose(ret['ctot'], 0), False)) * 1e3

            ret.update(abf=abf, tcre=tcre)

        if outform is not None:
            self.ret = ret
            df = self._mkout(**outform)
            return df
        else:
            return ret

    def _mkout(self, **kw):
        """
        Make results in a custom DataFrame.

        Parameters
        ----------
        kw['variables'] : list, optional
            Native variables list.
            If not given, all the output variables are processed.

        kw['variables_derived'] : list, optional
            Derived variables given in a form
            (name, list of elements, long_name, units),
            where the elements are composed of already-defined variable names,
            numbers including literals that can be evaluated with eval(),
            and arithmetic operators applied in reversed Polish notation.

        kw['conversion'] : list, optional
            Units conversion specifications given in a form
            (name, factor, new units).

        Returns
        -------
        df : pandas.DataFrame
            Resulting time series of each variable.
        """
        # Retrieve data for given variables
        dfx = []
        for name in kw.get('variables', list(self.ret)):
            if name in ['tak']:
                for k in range(self.ret[name].shape[0]):
                    namek = f'{name}[{k}]'
                    dfx.append((namek, self.ret[name][k]))
                    self.variables[namek] = self.variables[name]._replace(
                        long_name='{} in layer {}'.format(
                            self.variables[name].long_name, k))

            elif name == 'erf|CO2':
                dfx.append((name, self.forcing.c2erf(self.ret['cco2'])))

            elif name.startswith('erf|'):
                if name[4:] in self.df_erf.index:
                    dfx.append(
                        (name, self.df_erf.loc[name[4:], self.time].values))
                else:
                    self.logger.warning(f'{name} is not defined')

            else:
                dfx.append((name, self.ret[name]))

        self.dfx = dfx
        df = pd.DataFrame(dict(dfx), index=self.time).T

        # Add derived variables specified in reversed Polish notation
        map_binop = {
            '+': 'add',
            '-': 'sub',
            '*': 'mul',
            '/': 'div',
        }
        for name, elems, long_name, units in kw.get('variables_derived', []):
            stack = []

            for e1 in elems:
                if e1 in df.index:
                    stack.append(df.loc[e1])
                elif e1 in map_binop:
                    d1 = stack.pop()
                    d0 = stack.pop()
                    stack.append(getattr(d0, map_binop[e1])(d1))
                elif isinstance(e1, str):
                    stack.append(eval(e1))
                else:
                    stack.append(e1)

            df = df.append(stack[-1].to_frame(name).T)
            self.variables[name] = self.MCEVariable(long_name, units)

        # Apply units conversion
        for name, factor, units in kw.get('conversion', []):
            df.loc[name] *= factor
            self.variables[name] = self.variables[name]._replace(units=units)

        return df

    def get_climate_sensitivity(self, **kw):
        """
        Return climate sensitivity values.

        Parameters
        ----------
        kw : dict, optional
            Parameter values.
            If not given, retrieved from forcing and irm parms.

        Returns
        -------
        ret : dict
            Contains ecs, ecs_reg, and tcr.
            ecs : Equilibrium climate sensitivity (ECS) estimated from
                doubling CO2 forcing.
            ecs_reg : ECS estimated from quadrupling CO2 forcing.
                Equivalent to that diagnosed from an abrupt quadrupling CO2
                increase experiment with the regression method of Gregory
                et al. (2004, Geophysical Research Letters).
            tcr : Transient climate response (TCR), analytically calculated
                with a realized warming fraction at doubling time point
                along a 1%-per-year CO2 increase pathway.
        """
        alpha = kw.get('alpha', self.forcing.parms['alpha'])
        beta = kw.get('beta', self.forcing.parms['beta'])
        asj = kw.get('asj', self.irm.parms['asj'])
        tauj = kw.get('tauj', self.irm.parms['tauj'])
        lamb = kw.get('lamb', self.irm.parms['lamb'])

        t70 = np.log(2.) / np.log(1.01)
        rwf = 1 - (asj*tauj*(1-np.exp(-t70/tauj))).sum() / t70
        f2x = alpha*np.log(2)
        ecs = f2x / lamb
        ret = {
            'ecs': ecs,
            'ecs_reg': ecs * beta,
            'tcr': ecs * rwf,
        }
        return ret

