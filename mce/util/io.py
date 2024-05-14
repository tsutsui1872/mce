"""
Utility for handling data files
"""
import urllib.request
import pandas as pd
import netCDF4

from . import __name__ as module_name
from .. import get_logger

logger = get_logger(module_name)

def read_url(url, decode=True, queries={}):
    """Read internet resource contents

    Parameters
    ----------
    url
        Base URL
    decode, optional
        Whether or not decoded, by default True
    queries, optional
        Query key-value pairs, by default {}

    Returns
    -------
        Decoded contents
    """
    if queries:
        url = '{}?{}'.format(
            url,
            '&'.join(['{}={}'.format(*item) for item in queries.items()]),
        )

    with urllib.request.urlopen(url) as f1:
        enc = f1.info().get_content_charset(failobj='utf-8')
        ret = f1.read()

    if decode:
        ret = ret.decode(enc)

    return ret


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


def write_nc(path, var_dict, gatts, dim_unlimited='time'):
    """Write variable data and attributes to netcdf file

    Parameters
    ----------
    path
        Output netcdf path
    var_dict
        Variable dictionary where dict values are defined
        as tuple of data, dimensions, and attributes
    gatts
        Global attribute dictionary
    dim_unlimited, optional
        Dimension treated as "unlimited", by default 'time'
    """
    # Inspect used dimensions
    # Size of time dimension is set to "unlimited"
    dim_dict = {}
    for data, dims, _ in var_dict.values():
        for name, size in zip(dims, data.shape):
            if name in dim_dict:
                continue
            dim_dict[name] = size if name != dim_unlimited else None

    ncf = Dataset(path, 'w')

    for name, size in dim_dict.items():
        ncf.createDimension(name, size)

    for name, (data, dims, atts) in var_dict.items():
        ncv = ncf.createVariable(
            name, data.dtype, dims,
            fill_value=getattr(data, 'fill_value', None),
        )
        ncv.setncatts(atts)
        # Skip data writing for variables with unlimited dimension
        if dim_unlimited not in dims:
            ncv[:] = data

    ncf.setncatts(gatts)

    for name, (data, dims, _) in var_dict.items():
        # Writing data for variables with time dimension
        if dim_unlimited in dims:
            ncf.variables[name][:] = data

    ncf.close()


def read_ncvar(ncf, name, d1_start=None, d1_stop=None, ret_atts=False):
    """Read netcdf variable

    Parameters
    ----------
    ncf
        Dataset instance
    name
        Variable name
    d1_start, optional
        Start in slicing the first dimension, by default None
    d1_stop, optional
        Stop in slicing the first dimension, by default None
    ret_atts, optional
        Whether or not attributes added, by default False

    Returns
    -------
        Data or list of data, dimensions, and attributes
    """
    v1 = ncf.variables[name]
    slc = slice(d1_start, d1_stop)
    if ret_atts:
        atts = {k: v1.getncattr(k) for k in v1.ncattrs()}
        return [v1[slc], v1.dimensions, atts]
    else:
        return v1[slc]


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


def dffilter(df, dropna=False, **kw):
    """Filter time series data in pandas.DataFrame
    Filtering conditions are given by keyword=cond,
    where keyword is a name of multiIndex or a name of columns,
    depending on the type of input DataFrame,
    and cond is list, callable, or str.

    Parameters
    ----------
    df
        Time series data in pandas.DataFrame
    dropna, optional
        If true, time points are dropped when all values are nan

    Returns
    -------
        Filtered data
    """
    bl = pd.Series(True, df.index)
    is_midx = isinstance(df.index, pd.MultiIndex)

    for k, v in kw.items():
        if is_midx:
            d1 = pd.Series(df.index.get_level_values(k), index=df.index)
        else:
            d1 = df[k]

        if isinstance(v, list):
            bl = bl & d1.transform(lambda x: x in v)
        elif callable(v):
            bl = bl & d1.transform(v)
        else:
            bl = bl & (d1 == v)

    if dropna:
        ret = df.loc[bl].dropna(axis=1, how='all')
    else:
        ret = df.loc[bl]

    return ret
