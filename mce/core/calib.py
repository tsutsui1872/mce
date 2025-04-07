import copy
import numpy as np
from lmfit import Parameters, minimize

class CalibBase:
    """Base class for calibration
    """
    def __init__(self, *Drivers, **kw_all):
        """Create calibration instance
        """
        self.kw_ref = copy.deepcopy(kw_all)

        self.value_min = {}
        self.value_max = {}
        self.value_min_default = -np.inf
        self.value_max_default = +np.inf

        self.Driver = Drivers
        self.data = None # to be defined in minimize_wrap()

    def init_parms(self, **parms_in):
        """Define a dictionary of Parameter objects

        Parameters
        ----------
        parms_in
            Initial values of control parameters to be optimized,
            grouped by models with arbitrary identifiers for convenience
        """
        px = Parameters()

        for kind, pn_var in parms_in.items():
            for pn, pv in pn_var.items():
                px.add(
                    f'{kind}__{pn}', value=pv,
                    min=self.value_min.get(pn, self.value_min_default),
                    max=self.value_max.get(pn, self.value_max_default),
                )

        return px
    
    def retrieve_model_parms(self, kw_all, **parms):
        """To be overridden to retrieve model parameters
        from optimized control parameters

        Parameters
        ----------
        kw_all
            Reference model parameters (value-copied in get_model_kw())

        parms
            Optimized values of control parameters
        """
        pass

    def get_model_kw(self, px):
        """Return model parameters given to each component model
        according to optimization results

        Parameters
        ----------
        px
            Dictionary of Parameter objects
        """
        kw_all = copy.deepcopy(self.kw_ref)

        parms = {}
        for k, v in px.items():
            [kind, pn] = k.split('__')
            d = parms.setdefault(kind, {})
            d[pn] = v.value

        self.retrieve_model_parms(kw_all, **parms)

        return kw_all

    def get_model_config(self):
        """To be overridden to define model configurations
        """
        return [{}]

    def get_run_args(self):
        """To be overridden to define model run arguments
        """
        return [((), {})]

    def get_diff(self, results):
        """To be overridden to define residual series

        Parameters
        ----------
        results
            List of output from Driver.run()
        """
        # Reference ESM results
        data = self.data

        xdiff = []
        for df in results:
            xdiff.append(np.zeros(len(df)))

        return np.hstack(xdiff)

    def residual(self, px):
        """Return residual series to be minimized

        Parameters
        ----------
        px
            Dictionary of Parameter objects
        """
        kw_all = self.get_model_kw(px)
        results = []

        for Driver, cfg, (args, kw) in zip(
            self.Driver,
            self.get_model_config(),
            self.get_run_args(),
        ):
            drv = Driver(**{**kw_all, **cfg})
            results.append(drv.run(*args, **kw))

        return self.get_diff(results)

    def minimize_wrap(self, *data, **parms_in):
        """Wrapper for minimize()

        Parameters
        ----------
        data
            Results of target ESM runs

        parms_in
            Initial values of control parameters to be optimized,
            grouped by models with arbitrary identifiers for convenience

        Returns
        -------
            Optimized model parameters
        """
        self.data = data
        px = self.init_parms(**parms_in)
        self.ret = minimize(self.residual, px)
        kw_all = self.get_model_kw(self.ret.params)

        return kw_all
