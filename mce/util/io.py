"""
Utility for handling data files
"""

import pandas as pd
import netCDF4
from mce import get_logger

logger = get_logger('mce')

def read_ncfile(path, *args, **kw):
    """
    Return data contained in a given netcdf file

    Parameters:
    -----------
    path : str
        netcdf path

    args : tuple of str, optional
        netcdf variables
        all variables are read when the length of args is zero

    kw : dict, optional
        get_attrs : bool
            if true, return global attributes

    Returns:
    --------
    data : array or dict of array
        data for specified or all variables
        dict is returned for multimple variables
    """
    get_attrs = kw.get('get_attrs', False)

    logger.info('reading {}'.format(path))
    nc = netCDF4.Dataset(path)

    if len(args) == 0:
        names = list(nc.variables)
    else:
        names = args

    data = {}
    for name in names:
        data[name] = nc.variables[name][:]

    if get_attrs:
        attrs = dict([(x, nc.getncattr(x)) for x in nc.ncattrs()])

    nc.close()

    if len(names) == 1:
        data = data[names[0]]

    if get_attrs:
        return data, attrs
    else:
        return data


def get_irm_parms(path, project=None):
    """
    Fetch thermal response parameters for CMIP models

    Parameters
    ----------
    path : str
        Parameter file path

    project : str, optional, default None
        Name of CMIP era, such as CMIP5 and CMIP6

    Returns
    -------
    parms : pandas.DataFrame
        Thermal response parameters of CMIP models
    """
    parms = read_ncfile(path)
    parms['dataset'] = [
        ''.join(x.astype(str)).strip() for x in parms['dataset']]
    parms = pd.DataFrame(parms).set_index('dataset')

    map_labels = {
        'amplitude_0': 'a0',
        'amplitude_1': 'a1',
        'amplitude_2': 'a2',
        'time_constant_0': 'tau0',
        'time_constant_1': 'tau1',
        'time_constant_2': 'tau2',
    }
    parms = parms.rename(columns=map_labels)

    columns = sorted(parms.columns.tolist())

    if project is not None:
        parms['mip'] = project
        columns = ['mip'] + columns

    parms = parms.reindex(columns=columns)

    return parms

