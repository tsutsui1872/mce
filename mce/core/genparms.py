"""
API for model parameter generation.
"""

import numpy as np
import pandas as pd
import scipy.stats
from .. import get_logger
from ..util.stat import PcaBase

logger = get_logger(__name__)

class PcaParmsCmip(PcaBase):
    """
    Principal component analysis for climate parameters of CMIP models.
    """
    def __init__(self, names, **kw):
        """
        Parameters
        ----------
        names : list
            Parameters to be used.

        kw['names_log'] : list, optional, default []
            Parameters to which log conversion is applied.

        kw['dataset_excl'] : list, optional, default []
            Datasets not used for PC analysis.
        """
        self.names = names
        self.names_log = kw.get('names_log', [])
        self.dataset_excl = kw.get('dataset_excl', [])

    def _preproc(self, dfin):
        """
        Pre-processes input data for PC analysis.
        Derived parameters are added, and log conversion is applied.

        Parameters
        ----------
        dfin : pandas.DataFrame
            Input data.

        Returns
        -------
        df : pandas.DataFrame
            Normalized input data.
        """
        df = dfin.reindex(columns=self.names)

        if len(self.dataset_excl) > 0:
            df.drop(self.dataset_excl, inplace=True)

        if 'af1' in df and 'af2' in df:
            df['af1'] = dfin['a1'] / dfin['a0']
            df['af2'] = dfin['a2'] / dfin['a0']
        elif 'af2' in df and 'af0' in df:
            df['af2'] = dfin['a2'] / dfin['a1']
            df['af0'] = dfin['a0'] / dfin['a1']
        elif 'af0' in df and 'af1' in df:
            df['af0'] = dfin['a0'] / dfin['a2']
            df['af1'] = dfin['a1'] / dfin['a2']

        names = []
        self.mean_preproc = {}

        for name in df:
            if name in self.names_log:
                df[name] = np.log(df[name])
                names.append('log({})'.format(name))
            else:
                names.append(name)

        df.columns = names

        self.dfin = df
        df = (df - df.mean()) / df.std()

        return df

    def _postproc(self, df):
        """
        Post-processes output data from PC synthesis.
        Log conversion is reverted, and original parameters are calculated
        from derived parameters.

        Parameters
        ----------
        df : pandas.DataFrame
            Synthesized data.

        Returns
        -------
        df : pandas.DataFrame
            Denormalized synthesized data.
            Same name as input, but newly created.
        """
        df = df * self.dfin.std() + self.dfin.mean()

        names = []
        for name in df:
            if name.endswith(')'):
                names.append(
                    name.replace('log', '')[1:-1])
            else:
                names.append(name)

            if name.startswith('log('):
                df[name] = np.exp(df[name])

        df.columns = names

        if 'af1' in df and 'af2' in df:
            df['a0'] = 1. / (df['af1'] + df['af2'] + 1.)
            df['a1'] = df['af1'] * df['a0']
            df['a2'] = df['af2'] * df['a0']
            df.drop(['af1', 'af2'], axis=1, inplace=True)
        elif 'af2' in df and 'af0' in df:
            df['a1'] = 1. / (df['af2'] + df['af0'] + 1.)
            df['a2'] = df['af2'] * df['a1']
            df['a0'] = df['af0'] * df['a1']
            df.drop(['af2', 'af0'], axis=1, inplace=True)
        elif 'af0' in df and 'af1' in df:
            df['a2'] = 1. / (df['af0'] + df['af1'] + 1.)
            df['a0'] = df['af0'] * df['a2']
            df['a1'] = df['af1'] * df['a2']
            df.drop(['af0', 'af1'], axis=1, inplace=True)

        return df

    def analysis(self, df):
        """
        Performs principal component analysis.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data.

        Returns
        -------
        score : pandas.DataFrame
            Score data.
        """
        score = super().analysis(df)

        # arrange eigenvectors in ECS ascending order
        for ev in self.eigvec:
            schk = self.get_score_ref(2, **{ev: [-1., 1.]})
            pchk = self.synthesis(schk)
            if pchk.loc[0, 'ecs'] > pchk.loc[1, 'ecs']:
                logger.info('{} reversed'.format(ev))
                self.eigvec[ev] = -self.eigvec[ev]
                score[ev] = -score[ev]

        return score

    def synthesis(self, score):
        """
        Performs principal component synthesis.

        Parameters
        ----------
        score : pandas.DataFrame
            score data.

        Returns
        -------
        df : pandas.DataFrame
            Synthesized data.
        """
        df = super().synthesis(score)

        self.tcr2ecs(df)
        df['ecs_reg'] = df['co2_beta'] * df['ecs']

        return df

    def get_score_ref(self, length=1, **kw):
        """
        Returns score data for given ratios to square-root of eigenvalues.

        Parameters
        ----------
        length : int, optional, default 1
            Number of data points.

        kw['ev0'], kw['ev1'], ... : float or list of float
            Ratios to square-root of eigenvalues for selected components.
            When lists are given, they are the ranges to be used for np.linspace
            with the length parameter.

        Returns
        -------
        score : pandas.DataFrame
            Score data.
        """
        score = pd.DataFrame(
            np.zeros((length, len(self.eigval))), columns=self.eigval.index)

        if length == 1:
            for ev, x1, in kw.items():
                score.loc[0, ev] = np.sqrt(self.eigval[ev]) * x1
        else:
            for ev, stdrange in kw.items():
                args = (np.sqrt(self.eigval[ev]) * np.array(stdrange)).tolist()
                score[ev] = np.linspace(*args, **{'num': length})

        return score

    def tcr2ecs(self, df):
        """
        Calculates equilibrium climate sensitivity (ECS) from transient
        climate response (TCR).

        Parameters
        ----------
        df : pandas.DataFrame
            Climate parameter data. Two columns 'ecs' and 'lambda' added.
        """
        asj = (
            df[['a{}'.format(x) for x in range(3)]]
            .rename(lambda x: x.replace('a', ''), axis=1)
        )
        tauj = (
            df[['tau{}'.format(x) for x in range(3)]]
            .rename(lambda x: x.replace('tau', ''), axis=1)
        )
        t70 = np.log(2.) / np.log(1.01)
        ecs = df['tcr'] / \
            (1 - (asj*tauj*(1-np.exp(-t70/tauj))).sum(axis=1) / t70)
        xlamb = df['co2_alpha'] * np.log(2.) / ecs
        df['ecs'] = ecs
        df['lambda'] = xlamb

    def genparms(self, variance=None, mean=None, nsize=1000, seed=1):
        """
        Returns statistically simulated climate parameters.

        Parameters
        ----------
        variance : pandas.Series or dict, optional, default None
            Variance of PC components. self.eigval is assumed.

        mean : array_like, optional, default None
            Mean of the distribution. None is treated as zero. 

        nsize : int, optional, default 1000
            Sample size.

        seed : None, int, or other instance for scipy.stats random_state,
            optional, default 1
            If an int, use a new RandomState instance seeded with seed,
            otherwise, other RandomState is used.

        Returns
        -------
        df : pandas.DataFrame
            Simulated parameter data.

        score : pandas.DataFrame
            Simulated score data.
        """
        ev_names = sorted(self.eigval.index.tolist())

        if variance is None:
            score = pd.Series(mean).to_frame(0).sort_index().T
        else:
            if mean is not None:
                mean = [mean[ev] for ev in ev_names]
            rv = scipy.stats.multivariate_normal(
                mean=mean, cov=np.diag([variance[ev] for ev in ev_names]))
            score = pd.DataFrame(rv.rvs(nsize, seed), columns=ev_names)

        df = self.synthesis(score)

        if variance is None:
            return df
        else:
            return df, score

