"""
Class for handling estimated forcing-response parameters
"""

import numpy as np
import pandas as pd
from mce.util.io import read_ncfile
from mce.core.forcing import RfCO2
from mce.core.climate import IrmBase

class DiagResult(object):
    def __init__(self, **kw):
        self.dataroot = kw.get('dataroot', '../data')
        self.nl = kw.get('nl', 3)
        self.var_n = kw.get('var_n', 'rtnt')
        self.var_t = kw.get('var_t', 'tas')
        self.sources_cmip5 = [
            'ACCESS1.0', 'ACCESS1.3', 'BCC-CSM1.1', 'BNU-ESM', 'CCSM4',
            'CNRM-CM5', 'CSIRO-Mk3.6.0', 'CanESM2', 'FGOALS-s2', 'GFDL-CM3',
            'GFDL-ESM2G', 'GFDL-ESM2M', 'GISS-E2-H', 'GISS-E2-R', 'HadGEM2-ES',
            'INM-CM4', 'IPSL-CM5A-LR', 'IPSL-CM5B-LR', 'MIROC-ESM', 'MIROC5',
            'MPI-ESM-LR', 'MPI-ESM-MR', 'MPI-ESM-P', 'MRI-CGCM3', 'NorESM1-M',
        ]
        self.sources_cmip6 = [
            'BCC-CSM2-MR', 'BCC-ESM1', 'CanESM5', 'CESM2', 'CESM2-WACCM',
            'CNRM-CM6-1', 'CNRM-ESM2-1', 'GFDL-CM4', 'GISS-E2-1-G',
            'GISS-E2-1-H', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MIROC6',
            'MIROC-ES2L', 'MRI-ESM2-0', 'SAM0-UNICON', 'UKESM1-0-LL',
            'EC-Earth3-Veg', 'CAMS-CSM1-0', 'E3SM-1-0', 'NESM3', 'NorESM2-LM',
        ]
        self.map_sources = {
            'ACCESS1.0': 'ACCESS1-0',
            'ACCESS1.3': 'ACCESS1-3',
            'BCC-CSM1.1': 'bcc-csm1-1',
            'CSIRO-Mk3.6.0': 'CSIRO-Mk3-6-0',
            'INM-CM4': 'inmcm4',
        }

    def get_names(self, name, ret_rename=False):
        names = ['{}{}'.format(name, j) for j in range(self.nl)]
        if ret_rename:
            map_names = dict([x[::-1] for x in enumerate(names)])
            return names, map_names
        else:
            return names

    def get_gcm(self, dataset):
        project = dataset in self.sources_cmip5 and 'CMIP5' or 'CMIP6'
        exp4x = project=='CMIP6' and 'abrupt-4xCO2' or 'abrupt4xCO2'
        exp1p = '1pctCO2'
        pathtmpl = '{}/preproc2/{}_{}_{}_anom.nc'.format(
            self.dataroot, '{}', self.map_sources.get(dataset, dataset), '{}')
        var_n = self.var_n
        var_t = self.var_t

        gcm = {}

        for k, v, e in [
                ('4x_n', var_n, exp4x),
                ('1p_n', var_n, exp1p),
                ('4x_t', var_t, exp4x),
                ('1p_t', var_t, exp1p) ]:
            path = pathtmpl.format(v, e)
            # logger.info('reading {} for {}'.format(path, v))
            gcm[k] = read_ncfile(path, v)

        if dataset == 'GISS-E2-1-G':
            # Last 70 years in 1pctCO2 are replaced with 1pctCO2-4xext
            for k, v in [('1p_n', var_n), ('1p_t', var_t)]:
                path = pathtmpl.format(v, '1pctCO2-4xext')
                # logger.info('reading {} for {}'.format(path, v))
                d1 = read_ncfile(path, v)
                gcm[k][70:] = d1[:]

        return gcm

    def get_irm_parms(self, project, **kw):
        nl = kw.get('nl', self.nl)
        var_n = kw.get('var_n', self.var_n)
        var_t = kw.get('var_t', self.var_t)

        path = '{}/parms/parms_irm-{}_{}-{}_{}.nc'.format(
            self.dataroot, nl, var_n, var_t, project.lower())
        # path = '{}/parms/parm_estimate_{}-{}_irm-{}_{}.nc'.format(
        #     self.dataroot, var_n, var_t, nl, project.lower())
        # logger.info('reading {}'.format(path))
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
        parms.columns = [map_labels.get(x, x) for x in parms.columns]
        parms = parms.reindex(columns=['mip']+sorted(parms.columns.tolist()))
        parms['mip'] = project

        return parms

    def get_irm(self, parms=None, **kw):
        names_a = self.get_names('a')
        names_tau = self.get_names('tau')

        cfg_forcing = {}
        cfg_irm = {}

        if parms is not None:
            cfg_forcing['alpha'] = parms['alpha']
            cfg_forcing['beta'] = parms['beta']
            cfg_irm['asj'] = parms[names_a].values
            cfg_irm['tauj'] = parms[names_tau].values
            cfg_irm['lamb'] = parms['lambda']

        forcing = RfCO2(**cfg_forcing)
        irm = IrmBase(self.nl, **cfg_irm)

        kw2 = [(name, kw[name]) for name in forcing.parms.keys() if name in kw]
        if len(kw2) > 0:
            forcing.parms_update(**dict(kw2))

        kw2 = [(name, kw[name]) for name in irm.parms.keys() if name in kw]
        if len(kw2) > 0:
            irm.parms_update(**dict(kw2))

        t70 = np.log(2.) / np.log(1.01)
        f2x = forcing.x2erf(2.)
        asj = irm.parms['asj']
        tauj = irm.parms['tauj']
        ecs =  f2x / irm.parms['lamb']
        tcr = ecs * (1 - (asj*tauj*(1-np.exp(-t70/tauj))).sum() / t70)

        return forcing, irm, ecs, tcr

    def tcr2ecs(self, df):
        names_a, map_a = self.get_names('a', ret_rename=True)
        names_tau, map_tau = self.get_names('tau', ret_rename=True)
        asj = df[names_a].rename(columns=map_a)
        tauj = df[names_tau].rename(columns=map_tau)
        t70 = np.log(2.) / np.log(1.01)
        ecs = df['tcr'] / \
            (1 - (asj*tauj*(1-np.exp(-t70/tauj))).sum(axis=1) / t70)
        xlamb = df['alpha'] * np.log(2.) / ecs
        df['ecs'] = ecs
        df['lambda'] = xlamb

