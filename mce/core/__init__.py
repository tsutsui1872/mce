"""
MCE core API
"""

import numpy as np

from .. import MCEExecError

class ParmsBase:
    """
    Manage model parameters using class variables defined as property
    """
    def add(self, name, value, longname, unit, read_only=True):
        """
        Helper function to define property for each parameter
        """
        field_name = f'_{name}'

        def fchk(x):
            if not np.isscalar(x):
                x = np.array(x)
            return x

        def fget(_):
            return getattr(self, field_name)

        def fset(_, x):
            if read_only:
                raise MCEExecError(f'{name} is read-only')
            setattr(self, field_name, fchk(x))

        # set initial value
        setattr(self, field_name, fchk(value))

        # set property with doc string
        doc = '{}|{}|{}'.format(
            longname, unit, 'r' if read_only else 'rw'
        )
        setattr(self.__class__, name, property(fget, fset, None, doc))

    def names(self):
        """
        Return the parameter list
        """
        return [
            k for k, v in self.__class__.__dict__.items()
            if isinstance(v, property)
        ]

    def describe(self, name):
        """
        Return the description of a given parameter
        """
        longname, unit, mode = getattr(self.__class__, name).__doc__.split('|')
        return '{} ({}){}'.format(longname, unit, ', read-only' if mode=='r' else '')

    def update(self, **kw):
        """
        Update parameter values
        """
        names = self.names()
        for k, v in kw.items():
            if k not in names:
                raise MCEExecError(f'{k} is not defined')
            setattr(self, k, v)

    def __str__(self):
        ret = [
            '{}: {}'.format(name, self.describe(name))
            for name in self.names()
        ]
        return '\n'.join(ret)

    def __call__(self):
        return {
            name: getattr(self, name)
            for name in self.names()
        }


class ModelBase(object):
    def __init__(self, *args, **kw):
        self.parms = None
        self.init_process(*args, **kw)

    def init_process(self, *args, **kw):
        """
        To be overridden to define additional initial processes.
        """
        pass
