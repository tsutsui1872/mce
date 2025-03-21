import numpy as np
from scipy.integrate import solve_ivp

from .. import MCEExecError
from .forcing import RfAll
from .climate import IrmBase
from .carbon import ModelOcean, ModelLand
from . import ModelBase

class DriverBase:
    def __init__(self, **kw):
        """Base driver
        """
        self.climate = IrmBase(**kw.get('kw_irm', {}))
        self.forcing = RfAll(**kw.get('kw_rfall', {}))
        self.ocean = ModelOcean(**kw.get('kw_ocean', {}))
        self.land = ModelLand(**kw.get('kw_land', {}))
        self.gascycle = ModelBase() # dummy

        components = ['climate', 'ocean', 'land', 'gascycle']
        self.save = {}
        self.preproc(**{k: v for k, v in kw.items() if k in components})

    def _getta(self, y):
        if 'climate' not in self.map_y:
            raise MCEExecError(f'climate component not used')

        nl = len(self.climate.parms.tauj)
        takj = y[self.map_y['climate']].reshape((nl, nl))

        return takj

    def preproc(self, **kw_comp):
        """Generic preprocessing
        Keyword arguments specify the component models to be used
        Valid components: climate, ocean, land, gascycle
        """
        y0 = {}
        jn = [0]
        comp_order = []

        for comp, config in kw_comp.items():
            if comp == 'climate':
                tauj = self.climate.parms.tauj
                nl = len(tauj)
                lambk, xik, akj = self.climate.get_parms_ebm()
                self.save['climate'] = {
                    'lambk': lambk,
                    'xik': xik,
                    'akj': akj,
                }
                nv = nl * nl
                y0[comp] = np.zeros(nv)

            elif comp == 'ocean':
                # emission-driven default
                is_cdrv = config.get('is_cdrv', False)
                nv = 3 if is_cdrv else 4
                y0[comp] = np.zeros(nv)
                self.save['ocean'] = {'is_cdrv': is_cdrv}
            
            elif comp == 'land':
                nv = 4
                p = self.land.parms
                cbs_pi0 = p.fnpp0 * p.amp * (p.tau * p.tau)
                y0[comp] = cbs_pi0
                self.save['land'] = {'cbs_pi0': cbs_pi0}
            
            elif comp == 'gascycle':
                nv = len(config)

                map_unit_w = {'CH4': 'Mt', 'N2O': 'Mt'}
                map_unit_c = {'CH4': 'ppb', 'N2O': 'ppb'}

                ghg_order = []
                conc = []
                life = []
                w2c = []

                for gas, v in config.items():
                    unit_w = map_unit_w.get(gas, 'kt')
                    unit_c = map_unit_c.get(gas, 'ppt')
                    ghg_order.append(gas)
                    conc.append(v)
                    life.append(self.forcing.ghgs[gas].lifetime)
                    w2c.append(self.forcing.weight2conc(1, unit_w, gas, unit_c))

                y0[comp] = np.array(conc)
                self.save['gascycle'] = {
                    'ghg_order': ghg_order,
                    'life': np.array(life),
                    'w2c': np.array(w2c),
                }

            else:
                raise MCEExecError(f'invalid component {comp}')

            jn.append(nv)
            comp_order.append(comp)

        jn = np.array(jn).cumsum()
        self.map_y = dict(zip(
            comp_order,
            [slice(jn[i-1], jn[i]) for i in range(1, len(jn))],
        ))
        self.y0 = y0

    def erf_in(self, t, y):
        """To be overridden to evaluate forcing
        based on a given scenario

        Parameters
        ----------
        t
            Time point in year
        y
            Value of each prediction variable at the time point

        Returns
        -------
            Forcing in W m-2 at the time point
        """
        return 0.

    def emis_co2_in(self, t):
        """To be overridden when using the carbon cycle component
        to distinguish AFOLU emissions from FFI emissions
        based on a given scenario

        AFOLU=Agriculture, Forestry and Other Land Use
        FFI=Fossil Fuel combustion and Industrial processes

        FFI emissions are requierd for emission-driven runs

        Parameters
        ----------
        t
            Time point in year

        Returns
        -------
            Emission rates in Gt C yr-1 from FFI and AFOLU
            and cumulative emissions in Gt C from AFOLU
            at the time point
        """
        emis = {'FFI': 0., 'AFOLU': 0.}
        ecum = {'AFOLU': 0.}

        return emis, ecum

    def emis_ghg_in(self, t):
        """To be overridden when using the gas cycle component
        to evaluate non-CO2 GHG emissions based on a given scenario

        Parameters
        ----------
        t
            Time point in year

        Returns
        -------
            Emission rate of each GHG at the time point
            in Mt <gas> yr-1 for CH4 and N2O
            or in kt <gas> yr-1 for halogenated species
        """
        pass

    def conc_co2_in(self, t):
        """To be overridden for concentraion-driven runs
        to evaluate CO2 concentration based on a given scenario

        Parameters
        ----------
        t
            Time point in year

        Returns
        -------
            CO2 concentration in ppm at the time point
        """
        return self.forcing.parms.ccref

    def conc_co2_full_in(self):
        """To be overridden for concentraion-driven runs
        to get time series of input CO2 concentration

        Returns
        -------
            Time series of CO2 concentration in ppm
            from a given scenario
        """
        return []

    def func(self, t, y):
        """Function to be passed on to the solver

        Parameters
        ----------
        t
            Time point in year
        y
            Values of prediction variables at the time point

        Returns
        -------
            Time derivative of each prediction variable at the time point
        """
        emis_co2, ecum_co2 = self.emis_co2_in(t)

        if 'climate' in self.map_y:
            takj = self._getta(y)
            tas = takj[0, :].sum()

        ydot = []

        for comp, slc in self.map_y.items():
            model = getattr(self, comp)

            if comp == 'climate':
                lambk = self.save[comp]['lambk']
                akj = self.save[comp]['akj']
                takjdot = (self.erf_in(t, y) * (akj / lambk[0]) - takj) / model.parms.tauj
                ydot.append(takjdot.ravel())

            elif comp == 'ocean':
                hls = model.parms.hlk[0]

                if self.save['ocean']['is_cdrv']:
                    cco2 = self.conc_co2_in(t)
                    catm = (cco2 - model.parms.cco2_pi) / model.parms.alphaa
                    coc0 = model.partit_inv(catm, hls, tas) - catm
                    coc = np.hstack([coc0, y[slc]])
                    cocdot = -np.dot(model.msyst[1:, :], coc)

                else:
                    coc = np.array(y[slc])
                    catm = model.partit(coc[0], hls, tas)
                    coc[0] -= catm
                    cocdot = -np.dot(model.msyst, coc)
                    cocdot[0] += emis_co2.get('FFI', 0.)

                ydot.append(cocdot)

            elif comp == 'land':
                cbs = y[slc]
                fert = model.fertmm(catm)

                # Assumption 1
                # Decrease in primary production is proportional
                # to cumulative AFOLU emissions
                cbst_pi = self.y0[comp].sum()
                fert *= 1. - ecum_co2.get('AFOLU', 0.) / cbst_pi
                tau_l = model.get_tau(tas)
                cbsdot = -cbs/tau_l + fert * model.parms.amp * tau_l

                # Assumption 2
                # AFOLU emissions affect wood component
                cbsdot[2] -= emis_co2.get('AFOLU', 0.)

                ydot.append(cbsdot)

            elif comp == 'gascycle':
                conc = y[slc]
                life = self.save['gascycle']['life']
                w2c = self.save['gascycle']['w2c']
                conc_dot = - conc / life + self.emis_ghg_in(t) * w2c

                ydot.append(conc_dot)

            else:
                raise MCEExecError(f'unexpected component {comp}')

        if 'ocean' in self.map_y and not self.save['ocean']['is_cdrv']:
            # Changes in land carbon pool contribute to opposite changes
            # in the composit atmosphere and mixed-ocean layer
            cocdot[0] -= cbsdot.sum()

        return np.hstack(ydot)

    def preproc_add(self, *args, **kw):
        """To be overridden to define scenario-dependent preprocessing
        """
        pass

    def run(self, din, time, *args, **kw):
        """Generic model run method

        Parameters
        ----------
        din
            Input time series
        time
            Time points

        Returns
        -------
            Output from postproc method
        """
        self.time = time
        self.din = din

        self.preproc_add(*args, **kw)

        y0 = np.hstack([self.y0[comp] for comp in self.map_y])
        sol = solve_ivp(
            self.func, [time[0], time[-1]], y0, t_eval=time, max_step=1.,
        )
        self.sol = sol

        out = self.postproc(**{
            comp: sol.y[slc].copy()
            for comp, slc in self.map_y.items()
        })

        return out

    def postproc(self, **out):
        """Generic postprocessing

        Returns
        -------
            Time integration results
        """
        seconds_per_year = 3.15569e7
        earth_area = 5.10e14 # m^2

        data = {'time': self.time}

        for comp, y in out.items():
            model = getattr(self, comp)

            if comp == 'climate':
                nl = len(model.parms.tauj)
                tak = y.reshape((nl, nl, -1)).sum(axis=1)
                tas = tak[0]
                xik = self.save[comp]['xik']
                thc = (tak * xik[:, None]).sum(axis=0)
                data['tas'] = tas
                # convert J spy-1, m-2 to ZJ
                factor = seconds_per_year * earth_area * 1e-21
                data['thc'] = thc * factor

            elif comp == 'ocean':
                coc = y
                hls = model.parms.hlk[0]
                if self.save['ocean']['is_cdrv']:
                    # Assuming a series returned
                    cco2 = self.conc_co2_full_in()
                    catm = (cco2 - model.parms.cco2_pi) / model.parms.alphaa

            elif comp == 'land':
                cbs = y
                cbs_pi0 = self.save[comp]['cbs_pi0']
                cbst = (cbs - cbs_pi0[:, None]).sum(axis=0)
                data['cbst'] = cbst

            elif comp == 'gascycle':
                conc = y
                for j, gas in enumerate(self.save['gascycle']['ghg_order']):
                    data[f'ghg|{gas}'] = conc[j]

        if 'ocean' in self.map_y:
            model = self.ocean

            if self.save['ocean']['is_cdrv']:
                coc0 = np.array([
                    model.partit_inv(catm[i], hls, tas[i])
                    for i in range(coc.shape[1])
                ])
                coc = np.vstack([coc0, coc])
            else:
                catm = np.array([
                    model.partit(coc[0, i], hls, tas[i])
                    for i in range(coc.shape[1])
                ])
                cco2 = catm * model.parms.alphaa + model.parms.cco2_pi
                data['cco2'] = cco2

            coct = coc.sum(axis=0) - catm
            data['catm'] = catm
            data['coct'] = coct

        return data

