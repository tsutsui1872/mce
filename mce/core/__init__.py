"""
MCE core API
"""

import numpy as np
from mce import get_logger

class ModelBase(object):
    def __init__(self, *args, **kw):
        self.parms = {}
        self.init_process(*args, **kw)
        self.logger = get_logger('mce')

    def init_process(self, *args, **kw):
        """
        To be overridden to define additional initial processes.
        """
        pass

    def parms_update(self, **kw):
        """
        Update module parameters.

        Parameters
        ----------
        kw : dict, optional
            Keyword arguments to set new parameter values.
        """
        parms = self.parms

        for k, v in kw.items():
            if k not in parms:
                self.logger.warning('parameter "{}" not defined'.format(k))
                continue

            if np.isscalar(v):
                parms[k] = v
            else:
                parms[k] = np.array(v)

