"""
API for statistical modules of principal component analysis,
probability distribution fitting, and a Metropolis-Hastings (MH) sampler.
"""

import numpy as np
import pandas as pd
import scipy.stats
from collections import namedtuple
from lmfit import Parameters, minimize

class PcaBase(object):
    """
    Utility for principal component (PC) analysis
    """
    def __init__(self, *args, **kw):
        pass

    def _preproc(self, dfin):
        """
        Pre-processes input data for PC analysis.

        Parameters
        ----------
        dfin : pandas.DataFrame
            Input data.

        Returns
        -------
        df : pandas.DataFrame
            Normalized input data.
        """
        self.dfin = dfin
        df = (dfin - dfin.mean()) / dfin.std()
        return df

    def _postproc(self, df):
        """
        Post-processes output data from PC synthesis.

        Parameters
        ----------
        df : pandas.DataFrame
            Synthesized data.

        Returns
        -------
        dfout : pandas.DataFrame
            Denormalized synthesized data.
        """
        dfin = self.dfin
        dfout = df * dfin.std() + dfin.mean()
        return dfout

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
        df = self._preproc(df)
        self.dfstd = df.std()

        U, s, V = np.linalg.svd(
            df / np.sqrt((df*df).sum()), full_matrices=False)
        self.eigval = pd.Series(s*s, ['ev%d' % i for i in range(len(s))])
        self.eigvec = pd.DataFrame(
            V.transpose(), index=df.columns, columns=self.eigval.index)
        score = pd.DataFrame(
            np.dot(V, (df/self.dfstd).T).transpose(),
            index=df.index, columns=self.eigval.index)

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
        df = pd.DataFrame(
            np.dot(self.eigvec, score.T),
            index=self.eigvec.index, columns=score.index).T
        df = self._postproc(df * self.dfstd)
        return df


class StatModel:
    """
    Provide a wrapper function to make a random variable instance
    fitted to given quantile constraints.
    """
    def __init__(self):
        self.map_qkeys = {
            'very_likely__lower': 0.05,
            'likely__lower': 0.17,
            'central': 0.5,
            'likely__upper': 0.83,
            'very_likely__upper': 0.95,
        }

    def get_rv(
            self, dist_type, constraints={}, p_const=[], p_min={}, p_max={},
            **kw):
        """
        Returns a probability distribution for given constraints.

        Parameters
        ----------
        dist_type : str
            Name of a probability distribution function.

        constraints : dict, optional, default {}
            Constraining values at specific quantile points of 0.5,
            (0.17, 0.83), and (0.05, 0.95). Key names are defined as
            'central' for 0.5, 'likely__lower' and 'likely__upper' for
            0.17 and 0.83, and 'very_likely__lower' and 'very_likely__upper'
            for 0.05 and 0.95.

        p_const : list, optional, default []
            Distribution parameter names, for which the parameters are fixed
            to given values.

        p_min/p_max : dict, optional, default {}
            Lower/upper bounds for specific distribution parameters.

        kw : dict, optional
            Location, scale, and shape parameters. When constraints are given,
            the values are treated as first guess for numerical optimization.
            Location ('loc') and scale ('scale') parameters are optional,
            for which defaults are 0 and 1. Shape parameters are required for
            distributions that need them.

        Returns:
        --------
        rv : scipy.stats random variable instance
            Continuous distribution with frozen parameters.
        """

        f_stat = getattr(scipy.stats, dist_type)
        kwargs = {'loc': kw.get('loc', 0.), 'scale': kw.get('scale', 1.)}

        if f_stat.numargs > 0:
            Shapes = namedtuple('Shapes', f_stat.shapes.split(', '))
            pnames = Shapes._fields + ('loc', 'scale')
            args = Shapes(*[kw[k] for k in Shapes._fields])
        else:
            pnames = ('loc', 'scale')
            args = ()

        rv = f_stat(*args, **kwargs)

        if not constraints:
            return rv

        px = Parameters()

        for pname in pnames:
            if pname in kwargs:
                value = kwargs[pname]
            else:
                value = getattr(args, pname)
            px.add(
                pname, value=value, vary=pname not in p_const,
                min=p_min.get(pname, -np.inf), max=p_max.get(pname, +np.inf),
            )

        qpts = [self.map_qkeys[k] for k in constraints]
        qvals = np.array([constraints[k] for k in constraints])

        def residual(px):
            if len(args) > 0:
                tmp_args = [px[k].value for k in Shapes._fields]
            else:
                tmp_args = ()
            tmp_kwargs = {k: px[k].value for k in kwargs}
            rv = f_stat(*tmp_args, **tmp_kwargs)
            return rv.ppf(qpts) - qvals

        ret = minimize(residual, px)
        self.ret_minimize = ret

        if f_stat.numargs > 0:
            args = Shapes(*[ret.params[k].value for k in Shapes._fields])
        else:
            args = ()
        kwargs = {k: ret.params[k].value for k in ['loc', 'scale']}
        rv = f_stat(*args, **kwargs)

        return rv


class SamplingMH:
    """
    A Metropolis-Hastings (MH) independence sampler.
    Same as the original MH algorithm except that the
    candidate proposals do not depend on the current state.

    Parameters
    ----------
    df : pandas.Series or pandas.DataFrame
        Candidate data.

    pdf_target : function, optional, default None
        Target probability density.
        If means and covariance are given by keyword arguments,
        a multivariate normal distribution is used.

    kw : dict, optional
    kw['nmax'] : int, default 2000
        Maximum data points used to build a kernel densitiy function
        as candidate density.

    kw['mean'] : array_like
        Mean of the target distribution. If not given, `pdf_target` is used
        instead of building a multivariate normal distribution.

    kw['cov'] : array_like
        Covariance matrix of the target distribution. If not given,
        `pdf_target` is used instead of building a multivariate normal
        distribution.

    kw['w_cutoff'] : float
        Used to a cut-off weight to avoid inappropriate acceptance in a tail
        region where the ratio of the target to the candidate density may
        become erroneously large in certain circumstances.

    """
    def __init__(self, df, pdf_target=None, **kw):
        self.df = df
        self.nmax = kw.get('nmax', 2000)
        self.pdf_proposal = self._mk_proposal(df.values)
        if 'mean' in kw and 'cov' in kw:
            self.pdf_target = self._mk_target(kw['mean'], kw['cov'])
        else:
            self.pdf_target = pdf_target
        self.w_cutoff = kw.get('w_cutoff', np.inf)

    def _mk_proposal(self, df):
        nmax = self.nmax
        if df.ndim == 2:
            if df.shape[0] > df.shape[1]:
                df = df.transpose()
            df = df[:, :nmax]
        else:
            df = df[:nmax]
        return scipy.stats.gaussian_kde(df)

    def _mk_target(self, mean, cov):
        rv_target = scipy.stats.multivariate_normal(mean=mean, cov=cov)
        self.rv_target = rv_target
        return rv_target.pdf

    def _mk_weights(self, df):
        if df.ndim == 1:
            f_target = self.pdf_target(df)
            f_proposal = self.pdf_proposal(df)
        elif df.ndim == 2:
            if df.shape[0] > df.shape[1]:
                f_target = self.pdf_target(df)
                f_proposal = self.pdf_proposal(df.transpose())
            else:
                f_target = self.pdf_target(df.transpose())
                f_proposal = self.pdf_proposal(df)
        else:
            raise ValueError('invalid shape')

        self.weights = pd.DataFrame(
            [f_target, f_proposal, f_target/f_proposal],
            index=['f_target', 'f_proposal', 'weights'],
            columns = self.df.index
        ).T

        return f_target / f_proposal

    def sampling(self, seed=1):
        """
        Conduct one-dimensional independence chain MH.

        Parameters
        ----------
        seed : int or 1-d array_like, optional, default 1
            Seed for `RandomState` in Numpy.random.

        Returns
        -------
        index : List of int
            Accepted integer-locations in the given Series or DataFrame.
        """
        weights = self._mk_weights(self.df.values)

        np.random.seed(seed)
        random = np.random.uniform(size=len(weights))
        i0, i1 = 0, 1
        index = [i0]
        index_a = [i0]

        while i1 < len(weights):
            if weights[i1] / weights[i0] > random[i1] \
                    and weights[i1] < self.w_cutoff:
                index_a.append(i1)
                i0 = i1

            index.append(i0)
            i1 += 1

        index = [self.df.index[i] for i in index]
        index_a = [self.df.index[i] for i in index_a]

        return index, index_a
