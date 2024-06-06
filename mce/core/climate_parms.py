"""
API for diagnosing forcing-response parameters.
"""

import os
import numpy as np
from scipy import stats
import iris
import cf_units
from lmfit import Parameters, minimize
from .. import MCEExecError, get_logger
from .forcing import RfCO2
from .climate import IrmBase

logger = get_logger('mce')

class ParmEstimate(object):
    def __init__(self, nl=3):
        """
        Parameters
        ----------
        nl : int, optional, default 3
            Number of box-model layers.
        """
        if nl not in [2, 3]:
            logger().error('invalid number of layers {}'.format(nl))
            raise MCEExecError

        self.nl = nl
        self.irm = IrmBase(nl)
        self.forcing = RfCO2()

    def initpars(self, af, t, **kw):
        """
        Create Parameters object with transformed amplitudes `af` and time
        constants `t` and additional parameters.

        Parameters
        ----------
        af : array-like
            Amplitudes transformed as follows:
            af[0] = asj[1] / asj[0]
            af[1] = asj[2] / asj[0]

        t : array-like
            Time constants transformed as follows:
            t[0] = tauj[0]
            t[1] = tauj[1] / tauj[0]
            t[2] = tauj[2] / tauj[1]

        kw : dict, optional
            Additional parameters.

        Results
        -------
        px : Parameters
            Parameters object defined with given parameters.
        """
        afmin = 0.01
        txmin = {0: 0.}
        txmin_default = 2.

        px = Parameters()

        for i, afx in enumerate(af):
            px.add('af%d' % (i+1,), value=afx, min=afmin)
        for i, tx in enumerate(t):
            px.add('t%d' % i, value=tx, min=txmin.get(i, txmin_default))
        for k, v in kw.items():
            px.add(k, value=v)

        return px

    def get_a_tau(self, px):
        """
        Retrieve amplitudes `asj` and time constants `tauj`.

        Parameters
        ----------
        px : Parameters
            Parameters object.

        Returns
        -------
        (asj, tauj) : tuple
        """
        pvals = list(px.valuesdict().items())
        nb = self.nl -1
        b = np.array([v for k, v in pvals[:nb]])
        asj = np.zeros(nb+1) + 1./(1.+b.sum())
        asj[1:] *= b
        tauj = np.array([v for k, v in pvals[nb:nb+nb+1]]).cumprod()
        return asj, tauj

    def irm_wrap(self, time, **kw):
        """
        Compute abrupt-4xCO2 and 1pctCO2 response.

        Parameters
        ----------
        time : 1-D array
            Time points in year.

        kw : dict, optional
            Forcing and response parameters.

        Returns
        -------
        results : list
            list[0] = rndt4x
            list[1] = ts4x
            list[2] = rndt1p
            list[3] = ts1p
        """
        forcing = self.forcing
        irm = self.irm

        forcing.parms.update(**{
            k: kw[k] for k in forcing.parms() if k in kw
        })
        irm.parms.update(**{
            k: kw[k] for k in irm.parms() if k in kw
        })

        len4x = kw.get('len4x', 150)
        len1p = kw.get('len1p', 140)
        time4x = time[:len4x]
        time1p = np.hstack([0., time[:len1p]])

        f4x = forcing.x2erf(4)
        rndt4x = irm.response_ideal(time4x, variable='flux')
        ts4x = 1 - rndt4x
        rndt4x = rndt4x * f4x
        ts4x = ts4x * f4x / irm.parms.lamb

        f1p = forcing.xl2erf(time1p * np.log(1.01))
        ts1p = irm.response(time1p, f1p)
        rndt1p = f1p - irm.parms.lamb * ts1p

        results = [rndt4x, ts4x, rndt1p[1:], ts1p[1:]]

        return results

    def residual(self, px):
        time = self.datain['time']
        gcm = self.datain['gcm']
        std = self.datain['std']

        asj, tauj = self.get_a_tau(px)
        lamb = px['lamb'].value
        ts4xeq = px['ts4xeq'].value
        beta = px['beta'].value
        alpha = ts4xeq * lamb / ( beta * np.log(4) )

        irm = self.irm_wrap(
            time, asj=asj, tauj=tauj, lamb=lamb, alpha=alpha, beta=beta)

        # `std` is a list of two elements
        res = [ (x1 - x2) / w1 for x1, x2, w1 in zip(gcm, irm, std*2) ]

        return np.hstack(res)

    def minimize_wrap(self, time, gcm, std):
        """
        Wrapper method to estimate impulse response model parameters.

        Parameters
        ----------
        time : 1-D array
            Time points in year.

        gcm : list of four 1-D arrays
            [`rtnt_4x`, `tas_4x`, `rtnt_1p`, `tas_1p`]

        std : list of two floats
            Standard deviation of rtnt and tas of piControl.

        Returns
        -------
        alpha, beta, lamb, asj, tauj
            Estimated parameters.

        Notes
        -----
        Return object from `minimize()` is stored as `self.ret_minimize`.
        """
        self.datain = {
            'time': time,
            'gcm': gcm,
            'std': std,
        }

        if self.nl == 3:
            par_args = [[1., 1.], [1., 10., 20.]]
        else:
            par_args = [[1.], [2., 100.]]

        par_kw = dict(lamb=1., ts4xeq=7., beta=1.)
        px = self.initpars(*par_args, **par_kw)

        ret = minimize(self.residual, px)

        asj, tauj = self.get_a_tau(ret.params)
        lamb = ret.params['lamb'].value
        ts4xeq = ret.params['ts4xeq'].value
        beta = ret.params['beta'].value
        alpha = ts4xeq * lamb / beta / np.log(4)

        self.ret_minimize = ret

        return alpha, beta, lamb, asj, tauj


def regression_method(cubes_n, cubes_t):
    """
    Estimate forcing and feedback parameters by the regression method

    Parameters
    ----------
    cubes_n : dict of iris.cube.Cube
        Anomaly data of the TOA net downward total radiation in W/m2.

    cubes_t : dict of iris.cube.Cube
        Anomaly data of the surface temperature.

    Returns
    -------
    result : dict
    """
    regress = stats.linregress(cubes_t['4x'].data, cubes_n['4x'].data)
    lambda_reg = -regress.slope
    ecs_reg = 0.5 * regress.intercept / lambda_reg
    tcr_gcm = cubes_t['1p'].data[70-10:70+10].mean()

    result = {
        'lambda_reg': lambda_reg,
        'ecs_reg': ecs_reg,
        'tcr_gcm': tcr_gcm,
    }

    return result


def emulator_method(cubes_n, cubes_t, nl=3):
    """
    Estimte forcing and response parameters by the emulator method

    Parameters
    ----------
    cubes_n : dict of iris.cube.Cube
        Anomaly data of the TOA net downward total radiation in W/m2.

    cubes_t : dict of iris.cube.Cube
        Anomaly data of the surface temperature.

    nl : int, optional, default=3
        Number of exponentials in the impulse respone model.

    Returns
    -------
    result : dict
    """
    time = np.arange(150) + 0.5
    data_gcm = [
        cubes_n['4x'].data, cubes_t['4x'].data,
        cubes_n['1p'].data, cubes_t['1p'].data,
    ]
    data_std = [cubes_n['pi'].data.std(), cubes_t['pi'].data.std()]

    obj = ParmEstimate(nl=nl)
    if nl == 3:
        par_args = [[1., 1.], [1., 10., 20.]]
    else:
        par_args = [[1.], [2., 100.]]
    par_kw = dict(lamb=1., ts4xeq=7., beta=1.)

    px = obj.initpars(*par_args, **par_kw)
    alpha, beta, lamb, asj, tauj = obj.minimize_wrap(time, data_gcm, data_std)

    ecs = obj.forcing.x2erf(2.) / lamb
    t70 = np.log(2.) / np.log(1.01)
    tcr = ecs * (1 - (asj * tauj * (1 - np.exp(-t70/tauj))).sum() / t70)

    result = {
        'ecs': ecs,
        'tcr': tcr,
        'alpha': alpha,
        'beta': beta,
        'lambda': lamb,
    }
    for j, (as1, tau1) in enumerate(zip(asj, tauj)):
        result['amplitude_{}'.format(j)] = as1
        result['time_constant_{}'.format(j)] = tau1

    return result, obj


class DataCubes(object):
    def __init__(self):
        self.var_attrs = {
            'alpha': {
                'var_name': 'alpha',
                'long_name': 'Scaling factor of CO2 forcing',
                'units': cf_units.Unit('W m-2 degC-1'),
            },
            'beta': {
                'var_name': 'beta',
                'long_name': 'Amplification factor of CO2 forcing',
                'units': cf_units.Unit('no_unit'),
            },
            'time_constant_0': {
                'var_name': 'time_constant_0',
                'long_name': 'Time constant 0',
                'units': cf_units.Unit('year'),
            },
            'time_constant_1': {
                'var_name': 'time_constant_1',
                'long_name': 'Time constant 1',
                'units': cf_units.Unit('year'),
            },
            'time_constant_2': {
                'var_name': 'time_constant_2',
                'long_name': 'Time constant 2',
                'units': cf_units.Unit('year'),
            },
            'amplitude_0': {
                'var_name': 'amplitude_0',
                'long_name': 'Non-dimensional amplitude 0',
                'units': cf_units.Unit('no_unit'),
            },
            'amplitude_1': {
                'var_name': 'amplitude_1',
                'long_name': 'Non-dimensional amplitude 1',
                'units': cf_units.Unit('no_unit'),
            },
            'amplitude_2': {
                'var_name': 'amplitude_2',
                'long_name': 'Non-dimensional amplitude 2',
                'units': cf_units.Unit('no_unit'),
            },
            'lambda': {
                'var_name': 'lambda',
                'long_name': 'Climate feedback parameter',
                'units': cf_units.Unit('W m-2 degC-1'),
            },
            'ecs': {
                'var_name': 'ecs',
                'long_name': 'Equilibrium climate sensitivity',
                'units': cf_units.Unit('degC'),
            },
            'tcr': {
                'var_name': 'tcr',
                'long_name': 'Transient climate response',
                'units': cf_units.Unit('degC'),
            },
            'lambda_reg': {
                'var_name': 'lambda_reg',
                'long_name': 'Climate feedback parameter by regression',
                'units': cf_units.Unit('W m-2 degC-1'),
            },
            'ecs_reg': {
                'var_name': 'ecs_reg',
                'long_name': 'Equilibrium climate sensitivity by regression',
                'units': cf_units.Unit('degC'),
            },
            'tcr_gcm': {
                'var_name': 'tcr_gcm',
                'long_name': 'Transient climate response by AOGCM data',
                'units': cf_units.Unit('degC'),
            },
        }


def save_result(result, dataset, outpath):
    """
    Save result to a netcdf file.

    Parameters
    ----------
    result : dict
        Result from parameter estimate. 

    dataset: str
        Name of AOGCM.

    outpath: str
        Netcdf file path.

    Returns
    -------
    cube : iris.cube.CubeList
        Multi-model parameter data saved in a netcdf file.
    """
    if os.path.isfile(outpath):
        logger.info('Updating {}'.format(outpath))
        data = {}
        for cube in iris.load(outpath):
            data[cube.var_name] = dict(
                zip(cube.aux_coords[0].points, cube.data))
    else:
        logger.info('Creating {}'.format(outpath))
        data = dict((k, {}) for k in result.keys())

    for k, v in result.items():
        data[k][dataset] = v

    k1 = list(data)[0]
    datasets = sorted(data[k1], key=str.lower)
    dataset_coord = iris.coords.AuxCoord(datasets, long_name='dataset')

    obj = DataCubes()

    cubes = [
        iris.cube.Cube(
            np.array([data[k][dataset] for dataset in datasets]),
            aux_coords_and_dims=[(dataset_coord, 0)],
            attributes={}, **obj.var_attrs[k])
        for k in data]

    cube = iris.cube.CubeList(cubes)
    iris.save(cube, outpath)
    logger.info('Results of {} models saved'.format(len(datasets)))

    return cube


def get_gcm(dataset, project, indir, name_t):
    """
    Read anomaly data of the TOA net downward total radiation and the
    surface temperature

    Parameters
    ----------
    dataset : str
        model name

    project : str
        CMIP5 or CMIP6

    indir : str
        input directory

    name_t : str
        Variable name: 'tas' or 'ts'

    Returns
    -------
    cubes_n, cubes_t : list of iris.cube.Cube
    """
    map_experiments = {
        'pi': 'piControl',
        '4x': project=='CMIP5' and 'abrupt4xCO2' or 'abrupt-4xCO2',
        '1p': '1pctCO2'}

    cubes_n = {}
    cubes_t = {}

    for expid in map_experiments:
        expname = map_experiments[expid]

        path = '{}/rtnt_{}_{}_anom.nc'.format(indir, dataset, expname)
        logger.info('Reading {}'.format(path))
        cubes_n[expid] = iris.load_cube(path)

        path = '{}/{}_{}_{}_anom.nc'.format(indir, name_t, dataset, expname)
        logger.info('Reading {}'.format(path))
        cubes_t[expid] = iris.load_cube(path)

    if dataset == 'GISS-E2-1-G':
        # Special treatment for GISS-E2-1-G
        # 1pctCO2 data in the latter 70 years are replaced with
        # 1pctCO2-4xext data
        path = '{}/rtnt_{}_1pctCO2-4xext_anom.nc'.format(indir, dataset)
        logger.info('Reading {}'.format(path))
        cube_ext = iris.load_cube(path)
        year1 = cube_ext.coord('year').points[0]
        data = cubes_n['1p'].data
        coord_year = cubes_n['1p'].coord('year')
        i1 = np.where(coord_year.points==year1)[0][0]
        data[i1:] = cube_ext.data
        cubes_n['1p'] = cubes_n['1p'].copy(data)

        path = '{}/{}_{}_1pctCO2-4xext_anom.nc'.format(indir, name_t, dataset)
        logger.info('Reading {}'.format(path))
        cube_ext = iris.load_cube(path)
        data = cubes_t['1p'].data
        data[i1:] = cube_ext.data
        cubes_t['1p'] = cubes_t['1p'].copy(data)

    return cubes_n, cubes_t

