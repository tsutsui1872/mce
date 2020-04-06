"""
Driver for creating anomaly data.

Calculate a linear trend of piControl over a period corresponding to both the
150 years of abrupt-4xCO2 and the 140 years of 1pctCO2. Subtract this linear
trend from the original output, and save the results as anomaly data over the
limited period. The length of the period may exceed 150 years, depending on the
branch years of abrupt-4xCO2 and 1pctCO2. The years of abrupt-4xCO2 and 1pctCO2
are adjusted so that their time coordinates are the same as in piControl.

Typical usage:
%run preproc2.py IPSL-CM6A-LR MIROC-ES2L

The arguments are dataset names (CMIP5 and/or CMIP6 models).
Available datasets are displayed by running it with no arguments.
Several CMIP5 models have different names between the official dataset name
and the name used for data files. Here, the latter is used.

Input data, processed with preproc1.py, are fetched from <ddir>/preproc1,
and output data are saved to <ddir>/preproc2, where <ddir> is replaced with
its optional argument (default ../data).
See help message displayed with '-h' option.

The branch years, which should be carefully set for each dataset, are specified
in <ddir>/dataset.yml.

The script processes three variables rtnt, tas, and ts, but ts can be omitted
by deleting it from the variable loop in the code.

The file names have the following convention:

Input
<ddir>/preproc1/<variable>_<dataset>_<experiment>.nc
Output
<ddir>/preproc2/<variable>_<dataset>_<experiment>_anom.nc
"""

import os
import argparse
import yaml
import numpy as np
import iris
from scipy.stats import linregress
from mce import get_logger
from mce.script.preproc1 import CmipData

logger = get_logger('mce')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--ddir', default='../data', help='data directory, default ../data')
parser.add_argument('dataset', nargs='*')

args = parser.parse_args()

indir = '{}/preproc1'.format(args.ddir)
outdir = '{}/preproc2'.format(args.ddir)

with open('{}/dataset.yml'.format(args.ddir), 'r') as infile:
    cfg = yaml.safe_load(infile)

text = [
    'Available datasets:',
    'CMIP5 {} models'.format(len(cfg['CMIP5']['source'])),
    ' '.join(cfg['CMIP5']['source']),
    'CMIP6 {} models'.format(len(cfg['CMIP6']['source'])),
    ' '.join(cfg['CMIP6']['source']),
]

if len(args.dataset)==0:
    parser.print_usage()
    logger.info('\n'.join(text))

cmip = CmipData('none', 'none', 'none')
cfg_variables = cmip.cfg_variables

map_length = {
    'abrupt-4xCO2': 150,
    'abrupt4xCO2': 150,
    '1pctCO2': 140,
    '1pctCO2-4xext': 70,
}

for dataset in args.dataset:
    if dataset in cfg['CMIP5']['source']:
        project = 'CMIP5'
    elif dataset in cfg['CMIP6']['source']:
        project = 'CMIP6'
    else:
        raise RuntimeError('no such dataset {}'.format(dataset))

    cfg1 = cfg[project]['source'][dataset]

    # for varname in ['rtnt', 'tas', 'ts']:
    for varname in ['rtnt', 'tas']:
        cubes = {}
        y_start = {}
        y_end = {}

        for expname in cfg1:
            path = '{}/{}_{}_{}.nc'.format(indir, varname, dataset, expname)
            cubes[expname] = iris.load_cube(path)

            if expname == 'piControl':
                continue

            y_start[expname] = cfg1[expname]['branch_year']
            y_end[expname] = y_start[expname] + map_length[expname] -1

        y_period = (min(y_start.values()), max(y_end.values()))

        coord_pi = cubes['piControl'].coord('year')
        slc_period = slice(
            np.where(coord_pi.points==y_period[0])[0][0],
            np.where(coord_pi.points==y_period[1])[0][0] + 1)
        pi_reg = linregress(
            coord_pi.points[slc_period], cubes['piControl'].data[slc_period])

        attrs = {
            'model': dataset,
            'linear_trend_slope': pi_reg.slope,
            'linear_trend_interception': pi_reg.intercept}

        for expname, cube in cubes.items():
            if expname == 'piControl':
                slc_coord = slc_period
                slc_data = slc_period
            else:
                i0 = np.where(coord_pi.points==y_start[expname])[0][0]
                length = map_length[expname]
                slc_coord = slice(i0, i0+length)
                if dataset == 'HadGEM2-ES':
                    # skip the initial time point that has December only
                    slc_data = slice(1, 1+length)
                else:
                    slc_data = slice(0, length)

            trend = pi_reg.slope * coord_pi.points[slc_coord] + pi_reg.intercept

            cube_anom = iris.cube.Cube(
                cube.data[slc_data] - trend,
                attributes=attrs,
                aux_coords_and_dims=[(coord_pi[slc_coord], 0)],
                **cfg_variables[varname])

            outpath = '{}/{}_{}_{}_anom.nc'.format(
                outdir, varname, dataset, expname)

            if os.path.isfile(outpath):
                logger.warning('File {} exists'.format(outpath))
            else:
                logger.info('Save as {}'.format(outpath))
                iris.save(cube_anom, outpath)

