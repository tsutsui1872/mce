"""
Driver for diagnosing forcing-response parameters.

Typical usage:
%run diag_climate.py IPSL-CM6A-LR MIROC-ES2L

The arguments are dataset names (CMIP5 and/or CMIP6 models).
Available datasets are displayed by running it with no arguments.
Input data, processed with preproc2.py, are fetched from <ddir>/preproc2,
where <ddir> is replaced with its optional argument (default ../data).
Output is saved to '<ddir>/parms/parms_irm-<nl>_rtnt-<name_t>_<mip>.nc',
where <nl> and <name_t> are replaced with their optional arguments
(default 3 and tas), and <mip> is cmip5 or cmip6.

See help message for the optional arguments, displayed with '-h' option.
"""

import numpy as np
import argparse
import yaml
from mce import get_logger
from mce.core.climate_parms \
    import get_gcm, regression_method, emulator_method, save_result
from mce.util.io import get_irm_parms

logger = get_logger('mce')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--ddir', default='../data', help='data directory, default ../data')
parser.add_argument(
    '--name_t', default='tas', choices=['tas', 'ts'],
    help='temperature variable')
parser.add_argument(
    '--nl', type=int, default=3, help='number of box-model layers')
parser.add_argument('dataset', nargs='*', help='CMIP5 and/or CMIP6 models')

args = parser.parse_args()

cfgpath = '{}/dataset.yml'.format(args.ddir)
indir = '{}/preproc2'.format(args.ddir)
outdir = '{}/parms'.format(args.ddir)
name_t = args.name_t
nl = args.nl

with open(cfgpath, 'r') as infile:
    cfg = yaml.safe_load(infile)

map_models = {
    'ACCESS1-0': 'ACCESS1.0',
    'ACCESS1-3': 'ACCESS1.3',
    'bcc-csm1-1': 'BCC-CSM1.1',
    'CSIRO-Mk3-6-0': 'CSIRO-Mk3.6.0',
    'inmcm4': 'INM-CM4', 
}
map_sources = dict([x[::-1] for x in map_models.items()])

datasets = {}
text = ['Available datasets:']
    
for mip in ['CMIP5', 'CMIP6']:
    datasets[mip] = sorted(
        [map_models.get(x, x) for x in cfg[mip]['source']], key=str.lower)
    text.append('{} {} models'.format(mip, len(datasets[mip])))
    text.append(' '.join(datasets[mip]))

if len(args.dataset) == 0:
    parser.print_usage()
    logger.info('\n'.join(text))

for model in args.dataset:
    if model in datasets['CMIP5']:
        mip = 'CMIP5'
    elif model in datasets['CMIP6']:
        mip = 'CMIP6'
    else:
        raise RuntimeError('data not found for {}'.format(model))

    cubes_n, cubes_t = get_gcm(
        map_sources.get(model, model), mip, indir, name_t)

    logger.info('Run parameter estimate with nl={}'.format(nl))
    result, obj = emulator_method(cubes_n, cubes_t, nl=nl)
    result.update(**regression_method(cubes_n, cubes_t))

    outpath = '{}/parms_irm-{}_rtnt-{}_{}.nc'.format(
        outdir, nl, name_t, mip.lower())
    data = save_result(result, model, outpath)

