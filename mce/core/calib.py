import copy
import numpy as np
from lmfit import Parameters, minimize

class CalibBase:
    def __init__(self, *args, **kw):
        """Base class of parameter calibration
        """
        self.kw_ref = [copy.deepcopy(kw)]

        self.value_min = {}
        self.value_max = {}
        self.value_min_default = -np.inf
        self.value_max_default = +np.inf

        self.Driver = None

    def init_parms(self, **kw):
        """Define a dictionary of Parameter objects
        """
        px = Parameters()

        for kind, pn_var in kw.items():
            for pn in pn_var:
                px.add(
                    f'{kind}__{pn}', value=1.,
                    min=self.value_min.get(pn, self.value_min_default),
                    max=self.value_max.get(pn, self.value_max_default),
                )

        return px
    
    def adjust_parms(self, kw_all, **kw):
        """To be overridden to conform to get_model_kw()
        """
        pass

    def get_model_kw(self, px):
        """Return parameters given to each component model
        according to optimization results
        """
        kw_all = copy.deepcopy(self.kw_ref[-1])

        self.adjust_parms(
            kw_all, **{name: p1.value for name, p1 in px.items()},
        )

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
        """To be overridden to define residual series from model results
        """
        return 1.

    def residual(self, px):
        """Return residual series to be minimized
        """
        kw_all = self.get_model_kw(px)
        results = []

        for cfg, (args, kw) in zip(
            self.get_model_config(),
            self.get_run_args(),
        ):
            drv = self.Driver(**{**kw_all, **cfg})
            results.append(drv.run(*args, **kw))

        return self.get_diff(results)

    def minimize_wrap(self, *args, **kw):
        """Wrapper for minimize()
        """
        px = self.init_parms(*args, **kw)
        self.ret = minimize(self.residual, px)
        kw_all = self.get_model_kw(self.ret.params)

        return kw_all
