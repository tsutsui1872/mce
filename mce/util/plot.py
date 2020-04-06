"""
Utility for drawing pictures using matplotlib and seaborn
"""

import numpy as np
import unicodedata
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mce import get_logger

logger = get_logger('mce')

try:
    import seaborn as sns
    has_seaborn = True
except ImportError:
    logger.warning('seaborn is not installed')
    has_seaborn = False


def adjust_pageparms(
    nax=1, ncol=1, paper='a4', landscape=True,
    height=3., aspect=1.41, left=0.8, bottom=0.8,
    right=0.3, top=0.5, wspace=0.8, hspace=0.7):
    """
    Adjust page parameters for pdf output

    Parameters
    ----------
    nax : int, default 1
    ncol : int, default 1
    paper : str, default 'a4'
    landscape : bool, default True
    height : float, default 3.
    aspect : float, default 1.41
    left : float, default 0.8
    bottom : float, default 0.8
    right : float, default 0.3
    top : float, default 0.5
    wspace : float, default 0.8
    hspace : float, default 0.7

    Returns
    -------
    space : dict
    """
    width = height * aspect

    if paper == 'a3':
        paperwidth = 297.
        paperheight = 420.
    else: # a4
        paperwidth = 210.
        paperheight = 297.

    if landscape:
        paperwidth, paperheight = paperheight, paperwidth

    nrow = int(nax / ncol)
    if nax % ncol > 0:
        nrow += 1

    mpi = 25.4
    width_total = (left + ncol*width + (ncol-1)*wspace + right) * mpi
    height_total = (top + nrow*height + (nrow-1)*hspace + bottom) * mpi
    adj = max(
        [(width_total-(left+right)*mpi) / (paperwidth-(left+right)*mpi),
         (height_total-(top+bottom)*mpi) / (paperheight-(top+bottom)*mpi)])
    if adj > 1.:
        width /= adj
        height /= adj
        wspace /= adj
        hspace /= adj
        width_total = (width_total-(left+right)*mpi)/adj + (left+right)*mpi
        height_total = (height_total-(top+bottom)*mpi)/adj + (top+bottom)*mpi

    if width_total < paperwidth:
        adj_width = (paperwidth-width_total)/mpi*0.5
        left += adj_width
        right += adj_width
    if height_total < paperheight:
        adj_height = (paperheight-height_total)/mpi*0.5
        top += adj_height
        bottom += adj_height

    space = dict(
        height=height, aspect=aspect,
        left=left, right=right, top=top, bottom=bottom,
        wspace=wspace, hspace=hspace)

    return space


class PlotSpace(object):
    def __init__(self, *args, **kw):
        parms = {
            'height': 3.,
            'aspect': 1.4,
            'left': 0.8,
            'bottom': 0.8,
            'right': 0.3,
            'top': 0.5,
            'wspace': 0.8,
            'hspace': 0.7,
        }
        parms.update([(k, kw[k]) for k in kw if k in parms])
        rects = [
            [parms['left'], parms['bottom'],
             parms['height']*parms['aspect'], parms['height']] ]
        refs = [[-1, -1, -1, -1]]

        self.parms = parms
        self.rects = rects

        self.parms_seaborn = {
            'parms': {
                # context: dict, None,
                #   or one of {paper, notebook, talk, poster}
                # style: dict, None,
                #   or one of {darkgrid, whitegrid, dark, white, ticks}
                # palette: deep, muted, bright, pastel, dark, or colorblind
                'context': 'notebook',
                'style': 'ticks',
                'palette': 'colorblind',
                # 'font': 'Spica Neue Light',
                # 'font': 'Spica Neue',
            },
            'rc': {
                'mathtext.default': 'regular',
                'axes.facecolor': '.9',
            },
            'rc_ticks': {
                'axes.facecolor': 'w',
                # 'axes.facecolor': 'none',
                'axes.linewidth': 0.8,
                'axes.grid': False,
                'grid.linestyle': ':',
                'grid.linewidth': 0.8,
                'xtick.major.width': 0.8,
                'ytick.major.width': 0.8,
                'xtick.direction': 'in',
                'ytick.direction': 'in',
                'xtick.major.size': 5,
                'xtick.minor.size': 3,
                'ytick.major.size': 5,
                'ytick.minor.size': 3,
                'lines.linewidth': 1.0,
            },
        }
        self.parms_savefig = {
            'transparent': False,
            'bbox_inches': 'tight',
            'pad_inches': 0.04,
            'dpi': 180,
        }
        self.legend_data = {}

    def get_rect_panels(self):
        parms = self.parms
        rects = np.array(self.rects)
        left_edge = rects[:,0].min() - parms['left']
        bottom_edge = rects[:,1].min() - parms['bottom']
        right_edge = (rects[:,0] + rects[:,2]).max() + parms['right']
        top_edge = (rects[:,1] + rects[:,3]).max() + parms['top']
        return [
            left_edge, bottom_edge,
            right_edge-left_edge, top_edge-bottom_edge]

    def append(self, direction='right', ref=-1, **kw):
        parms = self.parms
        rects = self.rects

        height = kw.get('height', parms['height'])
        width = height * kw.get('aspect', parms['aspect'])
        wspace = parms['wspace']
        hspace = parms['hspace']

        rect_base = rects[ref]

        if direction == 'right':
            left = rect_base[0] + rect_base[2] + kw.get('xoff', wspace)
            bottom = rect_base[1] + kw.get('yoff', 0.)
        elif direction == 'bottom':
            left = rect_base[0] + kw.get('xoff', 0.)
            bottom = rect_base[1] - height - kw.get('yoff', hspace)
        elif direction == 'left':
            left = rect_base[0] - width - kw.get('xoff', wspace)
            bottom = rect_base[1] + kw.get('yoff', 0.)
        elif direction == 'top':
            left = rect_base[0] + kw.get('xoff', 0.)
            bottom = rect_base[1] + rect_base[3] + kw.get('yoff', hspace)
        elif direction == 'here':
            left = rect_base[0] + kw.get('xoff', 0.)
            bottom = rect_base[1] + kw.get('yoff', 0.)

        rects.append( [left, bottom, width, height] )

    def newfigure(self, **kw):
        if has_seaborn:
            parms_seaborn = self.parms_seaborn
            parms = parms_seaborn['parms']
            opts = dict([(k, kw.get(k, parms[k])) for k in parms])
            rc = parms_seaborn['rc'].copy()
            if opts['style'] == 'ticks':
                rc.update(parms_seaborn['rc_ticks'])
            sns.rcmod.set(rc=rc, **opts)
            self.sns = sns
        else:
            self.sns = None

        paper = self.get_rect_panels()
        fig = plt.figure(figsize=(paper[2], paper[3]))

        for left, bottom, width, height in self.rects:
            rect = [
                (left - paper[0]) / paper[2],
                (bottom - paper[1]) / paper[3],
                width / paper[2],
                height / paper[3] ]
            fig.add_axes(rect)

        self.figure = fig

        return fig

    def update_legend_data(self, ax):
        handles, labels = ax.get_legend_handles_labels()
        data = dict([x[::-1] for x in zip(handles, labels)])
        self.legend_data.update(data)

    def map_func(self, func, func_args=(), func_kw={}, axes=None):
        """
        Apply func across axes
        """
        if axes is None:
            axes = self.figure.axes

        kw = func_kw.copy()

        for ax in axes:
            if isinstance(func, str):
                getattr(ax, func)(*func_args, **kw)
            else:
                kw.update(ax=ax)
                func(*func_args, **kw)

    def axis_labels(self, labels_list, invis_list, axes=None):
        """
        Set axis labels and make tick-lines/labels invisible
        """
        if axes is None:
            axes = self.figure.axes

        for ax, labels, invis in zip(axes, labels_list, invis_list):
            for k, v in labels.items():
                getattr(ax, 'set_{}'.format(k))(v)

            for k in invis:
                plt.setp(getattr(ax, 'get_{}'.format(k))(), visible=False)

    def axis_share(self, axis='both', axes=None):
        """
        Share the range of x-axis and/or y-axis
        """
        assert axis in ['both', 'x', 'y']

        if axes is None:
            axes = self.figure.axes

        if axis in ['both', 'x']:
            xlims = np.array([ax.get_xlim() for ax in axes])
            xlim = [xlims.min(axis=0)[0], xlims.max(axis=0)[1]]
            for ax in axes:
                ax.set_xlim(*xlim)

        if axis in ['both', 'y']:
            ylims = np.array([ax.get_ylim() for ax in axes])
            ylim = [ylims.min(axis=0)[0], ylims.max(axis=0)[1]]
            for ax in axes:
                ax.set_ylim(*ylim)

        if axis == 'both':
            return xlim, ylim
        elif axis == 'x':
            return xlim
        else:
            return ylim

    def despine(self, ax, **kw):
        if has_seaborn:
            sns.despine(ax=ax, **kw)

    def savefig(self, path, close=True, **kw):
        """
        Wrapper for savefig
        """
        if isinstance(path, PdfPages):
            logger.info('page [{}])'.format(path.get_pagecount() + 1))
            path.savefig(self.figure)
        else:
            kw_savefig = self.parms_savefig.copy()
            kw_savefig.update(kw)
            logger.info('saved to {}'.format(path))
            self.figure.savefig(path, **kw_savefig)

        if close:
            plt.close(self.figure)


def wrap_plotspace(n=1, col=1, kw_space={}, kw_seaborn={}):
    """
    Wrapper for creating a figure that contains axes
    """
    p1 = PlotSpace(**kw_space)

    i = 1
    j = 1

    while 1:
        if (i-1)*col + j == n: break

        if j < col:
            p1.append('right')
            j = j+1
        else:
            p1.append('bottom', ref=-col)
            i = i+1
            j = 1

    p1.newfigure(**kw_seaborn)

    return p1


def unicode_character(name):
    """
    Wrapper function for unicodedata.lookup()

    Parameters
    ----------
    name : str
        Name to be looked up by such as
        - greek small letter alpha
        - greek capital letter alpha
        - degree sign
        - subscript two
        - superscript two

        Typical words
        greek: alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota,
            kappa, lamda, mu, nu, xi, omicron, pi, rho, sigma, tau, upsilon,
            phi, chi, psi, omega
        sign: degree, plus, minus, plus-minus, multiplication, division
        numeral: zero, one, two, three, four, five, six, seven, eight, nine
        other: em dash, en dash

    Returns
    -------
    char : unicode
        unicode character for a given name
    """
    greek_words = [
        'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
        'iota', 'kappa', 'lamda', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho',
        'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega' ]
    sign_words = [
        'degree', 'plus', 'minus', 'plus-minus', 'multiplication', 'division' ]
    numeral_words = [
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
        'eight', 'nine' ]

    if name in greek_words:
        name = 'greek small letter {}'.format(name)
    elif name.lower() in greek_words:
        name = 'greek capital letter {}'.format(name)
    elif name in sign_words:
        name = '{} sign'.format(name)
    elif name[0] == '_' and name[1:] in numeral_words:
        name = 'subscript {}'.format(name[1:])
    elif name[0] == '^' and name[1:] in numeral_words:
        name = 'superscript {}'.format(name[1:])

    char = unicodedata.lookup(name)

    return char

