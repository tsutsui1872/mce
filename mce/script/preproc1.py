"""
Driver for creating globally-averaged, yearly-aggregated data.

Fetch netcdf files for a specified dataset from a local CMIP archive
and process global averaging and yearly aggregation for rtnt, tas,
and ts, where rtnt is derived by rsdt - rsut - rlut.

Typical usage:
%run preproc2.py IPSL-CM6A-LR

The argument is a dataset name (CMIP5 or CMIP6 model).
Multiple datasets (CMIP5 and/or CMIP6 models) can be specified.
Available datasets are displayed by running it with no arguments.
Several CMIP5 models have different names between the official dataset name
and the name used for data files. Here, the latter is used.

The local CMIP archive is specified in <ddir>/dataset.yml with the directory
and file name structures and the ensemble id for each model experiment,
where <ddir> is replaced with its optional argument (default ../data).
See help message displayed with '-h' option.

Output data are saved to <ddir>/preproc1

The script processes three variables rtnt, tas, and ts, but ts can be omitted
by deleting it from the variable loop in the code.
Output files have the following convention:
<ddir>/preproc1/<variable>_<dataset>_<experiment>.nc
"""

import os
import re
from glob import glob
import hashlib
import numpy as np
import iris.coord_categorisation
import iris
from mce import get_logger

logger = get_logger('mce')

class CmipData(object):
    def __init__(self, rootpath, input_dir, input_file):
        """
        Parameters
        ----------
        rootpath : str
            Root path that has CMIP5 or CMIP6 directory

        input_dir : str
            Directory name structure used in a local archive, such as
            '[dataset][exp]'

        input_file : str or list of str
            File name structure used in a local archive, such as
            '[short_name]_[mip]_[dataset]_[exp]_[ensemble]_*.nc'
            Multiple patterns can be given in a list, and in this case
            file search is tried in order and stops when some files are found
        """
        self.rootpath = rootpath
        self.input_dir = input_dir
        if isinstance(input_file, str):
            input_file = [input_file]
        self.input_file = input_file

        self.cfg_variables = {
            'tas': {
                'standard_name': 'air_temperature',
                'long_name': 'Near-Surface Air Temperature',
                'units': 'K',
                'var_name': 'tas'},
            'ts': {
                'standard_name': 'surface_temperature',
                'long_name': 'Surface Temperature',
                'units': 'K',
                'var_name': 'ts'},
            'rtnt': {
                'standard_name': None,
                'long_name': 'TOA Net Downward Total Radiation',
                'units': 'W m-2',
                'var_name': 'rtnt'},
            'rlut': {
                'standard_name': 'toa_outgoing_longwave_flux',
                'long_name': 'TOA Outgoing Longwave Radiation',
                'units': 'W m-2',
                'var_name': 'rlut'},
            'rsdt': {
                'standard_name': 'toa_incoming_shortwave_flux',
                'long_name': 'TOA Incident Shortwave Radiation',
                'units': 'W m-2',
                'var_name': 'rsdt'},
            'rsut': {
                'standard_name': 'toa_outgoing_shortwave_flux',
                'long_name': 'TOA Outgoing Shortwave Radiation',
                'units': 'W m-2',
                'var_name': 'rsut'},
        }

        self.files = []

    def input_files(self, dataset, expname, varname, ensemble, hash=None):
        """
        Return netcdf paths.

        Parameters
        ----------
        dataset : str
            model name

        expname : str
            experiment name

        varname : str
            variable name

        ensemble str
            ensemble name

        hash : str, optional, default None
            Method to calculate hash values

        Returns
        -------
        files : list
            list of netcdf paths
            or list of dict that contains path and hash value
        """
        mapstr = {
            'frequency': 'mon',
            'mip': 'Amon',
            'dataset': dataset,
            'exp': expname,
            'short_name': varname,
            'ensemble': ensemble,
        }

        dirname = os.path.join(
            *[mapstr.get(elem[1:-1], '*')
             for elem in re.findall(r'\[.*?\]', self.input_dir)] )

        for input_file in self.input_file:
            patdic = dict(
                [(elem, mapstr.get(elem[1:-1], '*'))
                 for elem in re.findall(r'\[.*?\]', input_file)] )
            for k, v in patdic.items():
                input_file = input_file.replace(k, v)

            pathtmpl = os.path.join(self.rootpath, dirname, input_file)
            files = sorted(glob(pathtmpl))
            if len(files) > 0:
                break

        if len(files) == 0:
            raise RuntimeError('file not found for {}'.format(pathtmpl))

        if hash is not None:
            for i in range(len(files)):
                obj = getattr(hashlib, hash)()
                with open(files[i], 'rb') as f1:
                    while True:
                        chunk = f1.read(2048 * obj.block_size)
                        if len(chunk) == 0:
                            break
                        obj.update(chunk)

                files[i] = {'path': files[i], hash: obj.hexdigest() }

        self.files.extend(files)

        return files

    def get_cubes(self, dataset, expname, varname, ensemble, hash='sha256'):
        """
        Fetch netcdf files and return the contents in a list of iris.cube.Cube

        Parameters:
        -----------
        dataset : str
            model name

        expname : str
            experiment name

        varname : str
            variable name

        ensemble : str
            ensemble name

        hash : str, optional, default 'sha256'

        Returns
        -------
        cubes : list
            list of iris.cube.Cube
        """
        files = self.input_files(dataset, expname, varname, ensemble, hash=hash)
        standard_name = self.cfg_variables[varname]['standard_name']
        cubes = []

        for file1 in files:
            if isinstance(file1, dict):
                path = file1['path']
            else:
                path = file1
            logger.info('Loading {}'.format(path))
            cube = iris.load_cube(path, standard_name)
            cubes.append( cube )

        return cubes

    def add_aux_coords(self, cube):
        """
        Add auxiliary time coordinates

        Parameters:
        -----------
        cube : iris.cube.Cube
        """
        coords = [coord.name() for coord in cube.aux_coords]

        if 'day_of_month' not in coords:
            iris.coord_categorisation.add_day_of_month(cube, 'time')

        if 'day_of_year' not in coords:
            iris.coord_categorisation.add_day_of_year(cube, 'time')

        if 'month_number' not in coords:
            iris.coord_categorisation.add_month_number(cube, 'time')

        if 'year' not in coords:
            iris.coord_categorisation.add_year(cube, 'time')

    def preproc1(self, dataset, expname, varname, ensemble, **kw):
        """
        Process global averaging and yearly aggregation

        Parameters:
        -----------
        dataset : str
            model name

        expname : str
            experiment name

        varname : str
            variable name

        ensemble : str
            ensemble name

        kw : dict, optional

        Returns:
        --------
        cube : iris.cube.Cube
            Processed data
        """
        if varname == 'rtnt':
            cubes_rsdt = self.get_cubes(dataset, expname, 'rsdt', ensemble)
            cubes_rsut = self.get_cubes(dataset, expname, 'rsut', ensemble)
            cubes_rlut = self.get_cubes(dataset, expname, 'rlut', ensemble)

            cubes = []
            for rsdt, rsut, rlut in zip(cubes_rsdt, cubes_rsut, cubes_rlut):
                # if dataset == 'FGOALS-s2':
                #     # something wrong with latitude coordinate in rlut
                #     rlut.remove_coord('latitude')
                #     rlut.add_dim_coord(rsdt.coord('latitude'), 1)
                data = rsdt.data - rsut.data - rlut.data
                atts = self.cfg_variables['rtnt'].copy()
                atts['dim_coords_and_dims'] = [
                    x[::-1] for x in enumerate(rsdt.dim_coords)]
                cubes.append( iris.cube.Cube(data, **atts) )

        else:
            cubes = self.get_cubes(dataset, expname, varname, ensemble)

        coord_names = ['longitude', 'latitude']
        points = []
        bounds = []
        data = []

        coord_time_0 = cubes[0].coord('time')

        for cube in cubes:
            for coord_name in coord_names:
                if not cube.coord(coord_name).has_bounds():
                    cube.coord(coord_name).guess_bounds()

            grid_areas = iris.analysis.cartography.area_weights(cube)
            cube1 = cube.collapsed(
                coord_names, iris.analysis.MEAN, weights=grid_areas)

            slc = slice(None)

            if dataset == 'HadGEM2-ES' and expname == 'piControl':
                # excluding overwrap points
                self.add_aux_coords(cube1)
                for i, (year, month) in enumerate(
                        zip(cube1.coord('year').points,
                            cube1.coord('month_number').points)):
                    if year == 2099 and month == 11:
                        slc = slice(i, None)
                        break

            coord_time = cube1.coord('time')
            if coord_time.units != coord_time_0.units:
                coord_time.convert_units(coord_time_0.units)
            points.append(coord_time.points[slc])
            bounds.append(coord_time.bounds[slc])
            data.append(cube1.data[slc])

        attrnames = [
            'standard_name', 'long_name', 'var_name', 'units',
            'attributes', 'cell_methods' ]
        opts = dict([(k, getattr(cube1, k)) for k in attrnames])

        time_coord = cube1.coord('time').copy(
            points=np.hstack(points), bounds=np.vstack(bounds))
        opts['dim_coords_and_dims'] = [(time_coord, 0)]

        cube = iris.cube.Cube(np.hstack(data), **opts)
        self.add_aux_coords(cube)

        cube1 = cube.aggregated_by('year', iris.analysis.MEAN)

        return cube1


if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ddir', default='../data', help='data directory, default ../data')
    parser.add_argument('dataset', nargs='*')

    args = parser.parse_args()

    outdir = '{}/preproc1'.format(args.ddir)

    with open('{}/dataset.yml'.format(args.ddir), 'r') as infile:
        cfg = yaml.safe_load(infile)

    text = [
        'Available datasets:',
        'CMIP5 {} models'.format(len(cfg['CMIP5']['source'])),
        ', '.join(cfg['CMIP5']['source']),
        'CMIP6 {} models'.format(len(cfg['CMIP6']['source'])),
        ', '.join(cfg['CMIP6']['source']),
    ]
    if len(args.dataset)==0:
        parser.print_usage()
        logger.info('\n'.join(text))

    for dataset in args.dataset:
        if dataset in cfg['CMIP5']['source']:
            project = 'CMIP5'
        elif dataset in cfg['CMIP6']['source']:
            project = 'CMIP6'
        else:
            raise RuntimeError('no such dataset {}'.format(dataset))

        cmip = CmipData(
            cfg['rootpath'][project],
            cfg[project]['input_dir'],
            cfg[project]['input_file'])

        cfg1 = cfg[project]['source'][dataset]

        for expname in cfg1:
            # for varname in ['rtnt', 'tas', 'ts']:
            for varname in ['rtnt', 'tas']:
                outpath = '{}/{}_{}_{}.nc'.format(
                    outdir, varname, dataset, expname)
                if os.path.isfile(outpath):
                    logger.info('File {} exists'.format(outpath))
                    continue
                ensemble = cfg1[expname]['ensemble']
                cube = cmip.preproc1(dataset, expname, varname, ensemble)
                iris.save(cube, outpath)

