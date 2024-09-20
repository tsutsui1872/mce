import numpy as np
import pandas as pd
import matplotlib as mpl
from cycler import cycler
import unicodedata

from .. import get_logger

logger = get_logger(__name__)

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


def mk_map_colors(*args):
    return {
        k: '#{:02x}{:02x}{:02x}'.format(*v) for k, v
        in (zip(*args) if len(args)==2 else args[0].items())
    }

def get_colors_ipcc_style(color_id):
    if color_id == '6 colors':
        colors = [
            '#{:02x}{:02x}{:02x}'.format(*v)
            for v in [
                (0, 0, 0), (112, 160, 205), (196, 121, 0),
                (178, 178, 178), (0, 52, 102), (0, 79, 0),
            ]
        ]
        colors = cycler(color=colors)

    elif color_id == 'SSPs':
        colors = mk_map_colors({
            'SSP5-8.5': (149, 27, 30),
            'SSP3-7.0': (231, 29, 37),
            'SSP2-4.5': (247, 148, 32),
            'SSP1-2.6': (23, 60, 102),
            'SSP1-1.9': (0, 173, 207),
        })

    elif color_id == 'RCPs':
        colors = mk_map_colors({
            'RCP8.5': (149, 27, 30),
            'RCP6.0': (196, 121, 0),
            'RCP4.5': (84, 146, 205),
            'RCP2.6': (0, 173, 207),
        })
    else:
        logger.error(f'no such colors {color_id}')
        colors = None

    return colors


def plot_quantile_range(myplt, dfin, **kw):
    """
    Visualize central values, likely ranges, and very likely ranges
    of parameters and compare them from difference sources

    Parameters
    ----------
    dfin : DataFrame
        Indexed with (Member, Quantile) or (Group, Member, Quantile)
        Axis names are arbitrary

    kw : dict, optional
        Valid keys are as follows
        'group_order'
        'member_order'
        'map_color'
        'parm_order'
        'map_name_unit'
        'shrink'
        'kw_space', default {'height': 1.5, 'aspect': 2.5, 'wspace': 1.2},
        'col', default 1
        'kw_legend'
    """
    names = ['Group', 'Member', 'Quantile']

    nlevels = dfin.index.nlevels
    if nlevels == 2:
        dfin = pd.concat({'dummy': dfin}).rename_axis(names)
    elif nlevels == 3:
        dfin = dfin.rename_axis(names)
    else:
        logger.error('invalid number of index levels')
        return

    group_order = kw.get(
        'group_order',
        dfin.index.get_level_values('Group').unique().tolist(),
    )
    member_order = kw.get(
        'member_order',
        dfin.index.get_level_values('Member').unique().tolist(),
    )
    map_color = kw.get(
        'map_color',
        {k: f'C{i}' for i, k in enumerate(member_order)},
    )
    parm_order = kw.get('parm_order', dfin.columns.tolist())
    map_name_unit = kw.get('map_name_unit', {})
    shrink = kw.get('shrink', 0.7)

    kw_space = kw.get(
        'kw_space',
        {'height': 1.5, 'aspect': 2.5, 'wspace': 1.2},
    )
    col = kw.get('col', 1)

    nparms = len(parm_order)
    ngroups = len(group_order)
    nmembers = len(member_order)
    colors = [map_color[member] for member in member_order] * ngroups

    yvals = np.arange(ngroups * nmembers)[::-1].reshape((-1, nmembers)) + 0.5
    ym = yvals.mean(axis=1)
    yvals = ((yvals - ym[:, None]) * shrink + ym[:, None]).ravel()

    if 'extend' in kw_space:
        myplt.init_general(**kw_space)
    else:
        myplt.init_regular(nparms, col, kw_space=kw_space)

    midx = pd.MultiIndex.from_product([group_order, member_order])

    for n, pn in enumerate(parm_order):
        df = dfin[pn].unstack('Quantile').reindex(midx)
        ax = myplt(n)
        ax.hlines(
            yvals, df['very_likely__lower'], df['very_likely__upper'],
            color=colors, lw=1., zorder=1,
        )
        ax.hlines(
            yvals, df['likely__lower'], df['likely__upper'],
            color=colors, lw=4., zorder=1,
        )
        ax.scatter(
            df['central'], yvals,
            marker='o', facecolor='w', edgecolors=colors,
        )
        ax.set_yticks(ym)
        if nlevels == 3:
            ax.set_yticklabels(group_order)
            ax.tick_params(axis='y', labelleft=True, left=False)
        else:
            ax.tick_params(axis='y', labelleft=False, left=False)

        ax.set_ylim(0, ngroups*nmembers)
        ax.spines['left'].set_visible(False)

        name, unit = map_name_unit.get(pn, (pn, ''))
        if unit != '':
            ax.set_xlabel(f'{name} ({unit})')
        else:
            ax.set_xlabel(name)

        ax.grid(axis='x')

    handles = [
        mpl.lines.Line2D([0, 1], [0, 0], color=color, lw=1.5)
        for color in colors[:nmembers]
    ] + [
        mpl.patches.Patch(alpha=0, linewidth=0),
        mpl.lines.Line2D([0], [0], ls='None', marker='o', mec='k', mfc='w'),
        mpl.lines.Line2D([0, 1], [0, 0], color='k', lw=3., solid_capstyle='butt'),
        mpl.lines.Line2D([0, 1], [0, 0], color='k', lw=1., ls='-'),
    ]
    labels = member_order + ['', 'Central', 'likely (66%)', 'very likely (90%)']

    kw_legend = kw.get('kw_legend', {
        'loc': 'upper left',
        'bbox_to_anchor': (1.07, 0.98),
    })
    kw_legend['bbox_to_anchor'] = myplt.get_fig_position_relto_axes(
        kw_legend['bbox_to_anchor'],
    )
    myplt.figure.legend(handles, labels, **kw_legend)


def get_cmap_and_norm(
    cmap_base, ncols=None, levels=None, vmid=None, posmidoff=0.,
    undercol=None, overcol=None, posmin=0., posmax=1.,
    ):
    """Make cmap and norm for a contour plot 

    Parameters
    ----------
    cmap_base
        Base color map
    ncols, optional
        Number of contour levels, by default None
    levels, optional
        Contour levels, by default None
    vmid, optional
        _description_, by default None
    posmidoff, optional
        _description_, by default 0.
    undercol, optional
        _description_, by default None
    overcol, optional
        _description_, by default None
    posmin, optional
        _description_, by default 0.
    posmax, optional
        _description_, by default 1.

    Returns
    -------
        _description_
    """

    if ncols is not None:
        pos = np.linspace(posmin, posmax, ncols)
        colors = cmap_base(pos)
        cmap = mpl.colors.ListedColormap(colors)

    elif levels is not None:
        if vmid in levels:
            im = list(levels).index(vmid)
            pos = np.hstack([
                np.linspace(posmin, 0.5-posmidoff, len(levels[:im+1])-1),
                np.linspace(0.5+posmidoff, posmax, len(levels[im:])-1),
            ])
        else:
            pos = np.linspace(posmin, posmax, len(levels)-1)

        if vmid in levels and posmidoff==0.:
            pos = np.hstack([pos[:im], pos[im+1:]])
            levels = list(levels[:im]) + list(levels[im+1:])

        colors = cmap_base(pos)
        cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.BoundaryNorm(levels, cmap.N)

    else:
        cmap = cmap_base

    if undercol is not None:
        cmap.set_under(undercol)
    if overcol is not None:
        cmap.set_over(overcol)

    if levels is not None:
        return cmap, norm
    else:
        return cmap


def wrap_coastlines(ax, dxgrid=20, dygrid=20, resolution='110m', **kw):
    """Wrapper for ax.coastlines()

    Parameters
    ----------
    ax : Axes
        Axes object to be used
    dxgrid : int, optional
        Longitude grid spacing in degree, by default 20
    dygrid : int, optional
        Latitude grid spacing in degree, by default 20
    resolution : str, optional
        Map scale defined in Natural Earth, by default '110m'
        Map scales are at 1:10m, 1:50m, and 1:110m
    kw : dict, optional
        kw['kw_gridlines'] : Additional parameters for gridlines()
        kw['atts_gridlines'] : Additional parameters applied to gridline labels
    """
    kw_gridlines = {
        'draw_labels': False,
        'linewidth': 0.5,
        'color': 'grey',
    }
    kw_gridlines.update(kw.get('kw_gridlines', {}))
    atts_gridlines = {
        'x_inline': False,
        'y_inline': False,
        'top_labels': False,
        'right_labels': False,
        'xlabel_style': {'size': 'small'},
        'ylabel_style': {'size': 'small'},
    }
    atts_gridlines.update(kw.get('atts_gridlines', {}))

    ax.coastlines(resolution, linewidth=0.5, color='black')

    xlims_for_grid = [-180, 180]
    ylims_for_grid = [-90, 90]
    xgridrange = [dxgrid*np.round(val/dxgrid) for val in xlims_for_grid]
    ygridrange = [dygrid*np.round(val/dygrid) for val in ylims_for_grid]
    xgridpts = np.arange(xgridrange[0], xgridrange[1]+dxgrid, dxgrid)
    ygridpts = np.arange(ygridrange[0], ygridrange[1]+dygrid, dygrid)
    # Filter in case the rounding meant we went off-grid!
    xgridpts = xgridpts[ np.logical_and(xgridpts>=-180, xgridpts<=360) ]
    ygridpts = ygridpts[ np.logical_and(ygridpts>= -90, ygridpts<= 90) ]
    # gl.xlocator = mpl.ticker.FixedLocator(xgridpts)
    # gl.ylocator = mpl.ticker.FixedLocator(ygridpts)
    gl = ax.gridlines(
        # crs=ccrs.Geodetic(), # this raises AttributeError: 'Geodetic' object has no attribute 'x_limits'
        xlocs=xgridpts, ylocs=ygridpts,
        **kw_gridlines,
    )
    if kw_gridlines['draw_labels']:
        for k, v in atts_gridlines.items():
            setattr(gl, k, v)
