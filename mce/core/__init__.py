"""
MCE core API
"""

import pathlib
import numpy as np

try:
    import h5py
except ModuleNotFoundError:
    pass

try:
    from scipy.interpolate import interp1d
except ModuleNotFoundError:
    pass

from .. import MCEExecError, get_logger

logger = get_logger(__name__)

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


class ScenarioBase:
    def __init__(self, *args, **kw):
        kw = kw.copy()

        self.outpath_dummy = 'dummy.h5'
        outpath = kw.pop('outpath', self.outpath_dummy)
        self.outpath = outpath

        path = pathlib.Path(outpath)
        if path.exists():
            logger.info(f'{outpath} already exists')
            kw_h5 = {'mode': 'r'}
        else:
            kw_h5 = {'mode': 'w'}

        if outpath == self.outpath_dummy:
            kw_h5['driver'] = 'core'
            kw_h5['backing_store'] = False

        kw_h5.update({
            k: kw.pop(k)
            for k in ['mode', 'driver', 'backing_store']
            if k in kw
        })

        self.open(**kw_h5)
        self.init_process(*args, **kw)

    def init_process(self, *args, **kw):
        """To be overridden to define scenarios
        """
        pass

    def open(self, mode='r', **kw_h5):
        outpath = self.outpath

        if outpath == self.outpath_dummy:
            mesg = 'in-memory file opened'
        else:
            mesg = 'file {} opened with mode={}'.format(outpath, mode)

        self.file = h5py.File(outpath, mode=mode, **kw_h5)
        logger.info(mesg)

    def close(self):
        if self.file is None:
            logger.warning('file not opened')

        else:
            outpath = self.outpath

            if outpath == self.outpath_dummy:
                mesg = 'in-memory file closed'
            else:
                mesg = f'file {outpath} closed'

            self.file.close()
            self.file = None
            logger.info(mesg)

    def __call__(self, *args, **kwds):
        if self.file is None:
            logger.warning('file not created')
            ret = []

        else:
            ret = list(self.file)

        return ret

    def get_scenario(self, name):
        din = self.file[f'{name}/input']
        kw_interp = {'bounds_error': False, 'fill_value': 'extrapolate'}
        ret = {}

        for cat, grp in din.items():
            ret[cat] = {
                'time': grp['time'][:],
                'variables': [vn for vn in grp if vn != 'time'],
                'data': np.stack([
                    dset[:] for vn, dset in grp.items() if vn != 'time'
                ], axis=1),
                'attrs': {
                    vn: {k: v for k, v, in dset.attrs.items()}
                    for vn, dset in grp.items() # including time
                },
            }
            r = ret[cat]
            try:
                r['interp'] = interp1d(r['time'], r['data'].T, **kw_interp)
            except:
                logger.error('interp1d not available')

        return ret

