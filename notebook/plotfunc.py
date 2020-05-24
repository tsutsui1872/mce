# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""
Plotting functions
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mce.util.plot import PlotSpace, unicode_character

def plot_fitting(gcm, forcing, irm, names, px, kw_space={}, kw_seaborn={}):
    """
    Draw N and T time series, and their scatter diagram

    Parameters
    ----------
    gcm : dict
        AOGCM data

    forcing : RfCO2 object

    irm : IrmBase object

    names : dict
        Names for dataset and variables used in the fitting

    px : dict
        Additional parameters: lambda_reg, ecs, tcr, ecs_reg, tcr_gcm

    kw_space : dict, optional
        Arguments to PlotSpace()

    kw_seaborn : dict, optional
        Arguments to newfigure()

    Returns
    -------
    fig : figure object
    """
    opts_space = {
        'height': 2.5,
        'aspect': 1.4,
        'wspace': 0.9,
        'hspace': 0.4,
        'left': 0.8,
        'bottom': 0.8,
        'right': 0.3,
        'top': 0.5,
    }
    opts_space.update(kw_space)

    height = opts_space['height']
    aspect = opts_space['aspect']
    hspace = opts_space['hspace']
    wspace = opts_space['wspace']
    width = height * aspect

    p1 = PlotSpace(**opts_space)
    p1.append('bottom')
    p1.append('right', height=width, aspect=1., yoff=height*2+hspace-width)
    fig = p1.newfigure(**kw_seaborn)
    for ax in fig.axes:
        p1.despine(ax=ax)
        ax.grid(clip_on=False)

    colors_dark = sns.color_palette('dark')

    linestyle = {
        'GCM 4x': {
            'ls': 'None', 'mfc': 'none', 'ms': 4, 'mew': 0.6,
            'color': 'C1', 'marker': 's', 'alpha': 0.8},
        'GCM 1%': {
            'ls': 'None', 'mfc': 'none', 'ms': 4, 'mew': 0.6,
            'color': 'C0', 'marker': '^', 'alpha': 0.8},
        'IRM 4x': {'color': colors_dark[1]},
        'IRM 2x': {'color': colors_dark[1], 'dashes': [3, 1]},
        'IRM 1%': {'color': colors_dark[0]},
        'TCR': {
            'ls': 'None', 'mfc': 'none', 'mec': 'k', 'ms': 10,
            'mew': 1.5, 'marker': '+'},
        'ECS': {
            'ls': 'None', 'mfc': 'none', 'mec': 'k', 'ms': 10,
            'mew': 1.5, 'marker': '_'},
        'regress': {'color': 'k', 'ls': '-.'},
    }
    kw_legend = {'labelspacing': 0, 'borderaxespad': 0, 'frameon': False}
    degc = u'{}C'.format(unicode_character('degree'))
    wpm2 = u'W/m{}'.format(unicode_character('^two'))
    map_labels = {
        'rtnt': 'TOA net radiation ({})'.format(wpm2),
        'tas': 'Surface air-temperature ({})'.format(degc),
        'ts': 'Surface temperature ({})'.format(degc),
    }
    lamb = irm.parms['lamb']

    time_4x = np.arange(150) + 0.5
    time_1p = np.arange(140) + 0.5
    time_4xi = np.arange(151)
    time_1pi = np.arange(141)
    tp70 = np.log(2.) / np.log(1.01)
    f4x = forcing.x2erf(4.)
    f2x = forcing.x2erf(2.)
    f1p = forcing.xl2erf(time_1pi * np.log(1.01))
    irm_4x_n = irm.response_ideal(time_4xi, 'step', 'flux') * f4x
    irm_4x_t = irm.response_ideal(time_4xi, 'step', 'tres') * f4x / lamb
    irm_1p_t = irm.response(time_1pi, f1p)
    irm_1p_n = f1p - lamb * irm_1p_t

    ax = fig.axes[0]
    ax.plot(time_4x, gcm['4x_n'], label='GCM 4x', **linestyle['GCM 4x'])
    ax.plot(time_1p, gcm['1p_n'], label='GCM 1%', **linestyle['GCM 1%'])
    ax.plot(time_4xi, irm_4x_n, label='IRM 4x', **linestyle['IRM 4x'])
    ax.plot(time_1pi, irm_1p_n, label='IRM 1%', **linestyle['IRM 1%'])
    ax.legend(**kw_legend)
    ax.set_ylabel(map_labels[names['var_n']])

    ax = fig.axes[1]
    ax.plot(time_4x, gcm['4x_t'], label='GCM 4x', **linestyle['GCM 4x'])
    ax.plot(time_1p, gcm['1p_t'], label='GCM 1%', **linestyle['GCM 1%'])
    ax.plot(time_4xi, irm_4x_t, label='IRM 4x', **linestyle['IRM 4x'])
    ax.plot(time_1pi, irm_1p_t, label='IRM 1%', **linestyle['IRM 1%'])
    if 'tcr' in px:
        ax.plot(tp70, px['tcr'], label='TCR', **linestyle['TCR'])
    ax.legend(**kw_legend)
    ax.set_xlabel('Year')
    ax.set_ylabel(map_labels[names['var_t']])

    ax = fig.axes[2]
    ax.plot(
        gcm['4x_t'], gcm['4x_n'], label='GCM 4x',
        clip_on=False, **linestyle['GCM 4x'])
    ax.plot(
        gcm['1p_t'], gcm['1p_n'], label='GCM 1%',
        clip_on=False, **linestyle['GCM 1%'])
    ax.plot(
        [0., f4x/lamb], [f4x, 0.], label='IRM 4x',
        clip_on=False, **linestyle['IRM 4x'])
    ax.plot(
        irm_1p_t, irm_1p_n, label='IRM 1%',
        clip_on=False, **linestyle['IRM 1%'])
    if 'lambda_reg' in px and 'ecs_reg' in px:
        ax.plot(
            [0., 2*px['ecs_reg']],
            [px['lambda_reg']*2*px['ecs_reg'], 0.],
            label='Regress', clip_on=False, **linestyle['regress'])
    ax.plot(
        [0., f2x/lamb], [f2x, 0.], label='IRM 2x',
        clip_on=False, **linestyle['IRM 2x'])
    ax.legend(**kw_legend)
    ax.set_xlabel(map_labels[names['var_t']])
    ax.set_ylabel(map_labels[names['var_n']])

    tauj = irm.parms['tauj'].tolist()
    asj = irm.parms['asj'].tolist()
    nl = len(tauj)
    
    result_text = [
        'IRM-{} fitted to {} and {} of {}'
        .format(len(tauj), names['var_n'], names['var_t'], names['dataset']),
        u'alpha: {:.2f} {}, beta: {:.2f}'.format(
            forcing.parms['alpha'], wpm2, forcing.parms['beta']),
        '{}: {} y'
        .format(
            ', '.join(['tau{}'.format(i) for i in range(nl)]),
            ', '.join(['{:.3g}'.format(x) for x in tauj])),
        '{}: {}'
        .format(
            ', '.join(['a{}'.format(i) for i in range(nl)]),
            ', '.join(['{:.2f}'.format(x) for x in asj])),
    ]
    if 'lambda_reg' in px:
        result_text.append(
            u'lambda, lambda(reg): {:.2f}, {:.2f} {}/{}'
            .format(lamb, px['lambda_reg'], wpm2, degc))
    else:
        result_text.append(
            u'lambda: {:.2f} {}/{}'.format(lamb, wpm2, degc))
    if 'ecs' in px and 'tcr' in px:
        result_text.append(
            u'ecs, tcr: {:.2f}, {:.2f} {}, rwf: {:.2f}'
            .format(px['ecs'], px['tcr'], degc, px['tcr']/px['ecs']))
    if 'ecs_reg' in px and 'tcr_gcm' in px:
        result_text.append(
            u'ecs(reg), tcr(gcm): {:.2f}, {:.2f} {}, rwf: {:.2f}'
            .format(px['ecs_reg'], px['tcr_gcm'], degc,
                    px['tcr_gcm']/px['ecs_reg']))
    # xpos = 0.58
    # ypos = 0.27
    xpos = (opts_space['left'] + width + wspace + 0.02) / (
        opts_space['left'] + 2*width + wspace + opts_space['right'])
    ypos = (opts_space['bottom'] + 1.) / (
        opts_space['bottom'] + 2*height + hspace + opts_space['top'])
    fig.text(xpos, ypos, '\n'.join(result_text), ha='left', va='top', size=10)

    for i, ax in enumerate(fig.axes):
        ax.text(
            -0.12, 1., chr(97+i), ha='right', va='top', size=17,
            transform=ax.transAxes)

    return p1


def plot_tcr_ecs(parms, ecs_conv=True, **kw):
    """
    Draw TCR-ECS scatter diagrams

    Parameters
    ----------
    parms : pandas.DataFrame
        Thermal response parameters of CMIP models

    ecs_conv : bool, optional, default True
        If True, it draws two panels with conventional and new ECS estimates;
        otherwise one panel for the latter

    kw : dict, optional, default {}
        Used to modify labels

    Returns
    -------
    p1 : PlotSpace object
    """
    degc = u'{}C'.format(unicode_character('degree'))
    xlabels = kw.get(
        'xlabels',
        [u'Equilibrium climate sensitivity by regression ({})'.format(degc),
         u'Equilibrium climate sensitivity by emulator ({})'.format(degc)])
    ylabel = kw.get(
        'ylabel', u'Transient climate response ({})'.format(degc))

    height = 4.
    aspect = 1.
    aspect2 = 7.
    p1 = PlotSpace(height=height, aspect=aspect, wspace=0.2, hspace=0.2)

    if ecs_conv:
        p1.append('right')
        p1.append('top', ref=0, height=height/aspect2, aspect=aspect2, yoff=0.)
        p1.append('top', ref=1, height=height/aspect2, aspect=aspect2, yoff=0.)
        p1.append('right', ref=1, aspect=1./aspect2, xoff=0.)
    else:
        p1.append('top', ref=0, height=height/aspect2, aspect=aspect2, yoff=0.)
        p1.append('right', ref=0, aspect=1./aspect2, xoff=0.)

    fig = p1.newfigure()
    for i, ax in enumerate(fig.axes):
        if i > (ecs_conv and 1 or 0):
            p1.despine(ax, left=True, bottom=True)
        else:
            p1.despine(ax)

    df = parms.groupby('mip')
    mipnames = list(df.groups)

    xvars = ['ecs_reg', 'ecs']
    yvar = 'tcr'
    label_refline = '0.6-to-1'

    if not ecs_conv:
        del xvars[0]
        del xlabels[0]

    label_list = []
    invis_list = []

    for i, xvar in enumerate(xvars):
        ax = fig.axes[i]

        for j, mip in enumerate(mipnames):
            color = 'C{}'.format(j)
            sns.regplot(
                x=xvar, y=yvar, data=df.get_group(mip), color=color, ax=ax,
                scatter_kws={'edgecolor': 'w', 's': 50}, label=mip)
            p1.update_legend_data(ax)

        if i == 0:
            label_list.append({'xlabel': xlabels[0], 'ylabel': ylabel})
            invis_list.append([])
        else:
            label_list.append({'xlabel': xlabels[1], 'ylabel': ''})
            invis_list.append(['yticklabels'])

    nb = len(xvars)
    xlim, ylim = p1.axis_share(axes=fig.axes[:nb])

    for ax in fig.axes[:nb]:
        ax.grid(True)
        ax.plot(
            np.array(xlim), 0.6*np.array(xlim), ls='--', color='0.2',
            label=label_refline)
        p1.update_legend_data(ax)

    box_kws = {
        'saturation': 1,
        'width': 0.5, 'showmeans': True,
        'meanprops': {'markerfacecolor': 'none', 'markeredgecolor': 'k'} }

    for i, xvar in enumerate(xvars):
        ax = fig.axes[nb+i]
        ax.set_xlim(*xlim)
        sns.boxplot(
            x=xvar, y='mip', data=parms, orient='horizontal', ax=ax, **box_kws)
        label_list.append({'xlabel': '', 'ylabel': ''})
        if i == 0:
            invis_list.append(['xticklabels', 'xticklines', 'yticklines'])
        else:
            invis_list.append(
                ['xticklabels', 'yticklabels', 'xticklines', 'yticklines'])

    ax = fig.axes[2*nb]
    ax.set_ylim(*ylim)
    sns.boxplot(
        x='mip', y='tcr', data=parms, orient='vertical', ax=ax, **box_kws)
    label_list.append({'xlabel': '', 'ylabel': ''})
    invis_list.append(['yticklabels', 'xticklines', 'yticklines'])

    p1.axis_labels(label_list, invis_list)
    # plt.setp(ax.get_xticklabels(), rotation=30)
    plt.setp(ax.get_xticklabels(), rotation=60)

    labels = mipnames + [label_refline]
    handles = [p1.legend_data[k] for k in labels]
    ax = fig.axes[0]
    ax.legend(handles, labels, labelspacing=0.)

    if ecs_conv:
        for i, ax in enumerate(fig.axes[:2]):
            ax.text(
                0.98, 0.02, chr(97+i), ha='right', va='bottom', size=17,
                transform=ax.transAxes)

    return p1


def plot_parms_rel(df, **kw):
    """
    """
    # bins_fb = kw.get('bins_fb', [0.5, 0.8, 1.1, 1.4, 1.7])
    bins_fb = kw.get('bins_fb', [-np.inf, 0.8, 1.1, 1.4, np.inf])
    df = df.copy()
    df['1/lambda'] = 1. / df['lambda']
    df['rwf'] = df['tcr'] / df['ecs']

    # derived parameters for yet to be realized warming fractions
    tp2x = np.log(2.) / np.log(1.01)
    df['yrwf0'] = df['tau0']/tp2x*(1-np.exp(-tp2x/df['tau0']))
    df['yrwf1'] = df['tau1']/tp2x*(1-np.exp(-tp2x/df['tau1']))
    df['yrwf2'] = df['tau2']/tp2x*(1-np.exp(-tp2x/df['tau2']))
    df['a0*yrwf0'] = df['a0']*df['yrwf0']
    df['a1*yrwf1'] = df['a1']*df['yrwf1']
    df['a2*yrwf2'] = df['a2']*df['yrwf2']

    # categorize with 1/lambda
    # if df['1/lambda'].min() < 0.5 or df['1/lambda'].max() > 1.7:
    #     raise ValueError
    # bincat = pd.cut(df['1/lambda'], bins=np.linspace(0.5, 1.7, 5))
    bincat = pd.cut(df['1/lambda'], bins=bins_fb)

    # CMIP5 mean
    dfm = df.groupby('mip').mean().loc['CMIP5']

    # deviation from CMIP5 mean
    dfa = df.drop('mip', axis=1).sub(dfm)
    dfa['mip'] = df['mip']
    dfa['class'] = bincat

    # fractional difference from CMIP5 mean
    dfb = df.drop('mip', axis=1).div(dfm) - 1
    dfb['mip'] = df['mip']
    
    height = 3.
    aspect1 = 0.65
    aspect2 = 1.
    wspace = 0.4
    hspace = 0.4
    hspace2 = hspace*1.8
    height2  = (height*2 + hspace2 - hspace*2) / 3.
    p1 = PlotSpace(height=height, aspect=aspect1, wspace=wspace, hspace=hspace)
    p1.append(
        'right', height=height2, aspect=aspect2, xoff=2*wspace,
        yoff=height-height2)
    p1.append('right', height=height2, aspect=aspect2)
    p1.append('right', height=height2, aspect=aspect2)
    p1.append('bottom', -2, height=height2, aspect=aspect2)
    p1.append('right', height=height2, aspect=aspect2)
    p1.append('bottom', height=height2, aspect=aspect2)
    p1.append('bottom', 0, aspect=aspect1*(15./12.), yoff=hspace2)
    p1.append('right', aspect=aspect1*(10./12.))
    fig = p1.newfigure()
    for ax in fig.axes:
        p1.despine(ax)

    yvars = ['alpha', 'beta', '1/lambda', 'rwf']
    box_kws = {
        'width': 0.65, 'showmeans': True,
        'meanprops': {'markerfacecolor': 'none', 'markeredgecolor': 'k'} }

    df1 = dfb[yvars+['mip']].melt(id_vars='mip')
    k = 0
    ax = fig.axes[k]
    sns.boxplot(
        x='variable', y='value', hue='mip', data=df1, ax=ax, **box_kws)
    k = k+1

    df_corr = df[yvars].corr()
    text_minus = unicode_character('minus')

    for i in range(len(yvars)):
        for j in range(i+1, len(yvars)):
            ax = fig.axes[k]
            if k == 6:
                kw = {}
            else:
                kw = {'legend': False}
            sns.scatterplot(
                x=yvars[j], y=yvars[i], hue='mip', data=df, ax=ax, **kw)
            ax.axhline(dfm[yvars[i]], color='0.4', lw=0.6, alpha=0.5)
            ax.axvline(dfm[yvars[j]], color='0.4', lw=0.6, alpha=0.5)
            tx1 = 'Corr: {:.3f}'.format(df_corr.loc[yvars[j], yvars[i]])
            tx1 = tx1.replace('-', text_minus)
            ax.text(
                1., 1.01, tx1, ha='right', va='bottom', size=10,
                transform=ax.transAxes)
            k = k+1

    box_kws = {'width': 0.65, 'showfliers': False, 'palette': 'YlOrBr'}
                
    ax = fig.axes[k]
    xvars1 = ['a0*yrwf0', 'a1*yrwf1', 'a2*yrwf2']
    df1 = dfa[xvars1+['class']].melt(id_vars=['class'])
    sns.boxplot(
        x='variable', y='value', hue='class', data=df1, ax=ax, **box_kws)

    ax = fig.axes[k+1]
    xvars2 = ['a2', 'yrwf2']
    df1 = dfa[xvars2+['class']].melt(id_vars=['class'])
    sns.boxplot(
        x='variable', y='value', hue='class', data=df1, ax=ax, **box_kws)

    p1.axis_share(axis='y', axes=fig.axes[7:])

    wpsqm = u'W/m{}'.format(unicode_character('^two'))
    degc = u'{}C'.format(unicode_character('degree'))
    gsl_alpha = unicode_character('alpha')
    gsl_beta = unicode_character('beta')
    gsl_lambda = unicode_character('lamda')
    map_labels = {
        'alpha': u'{} ({})'.format(gsl_alpha, wpsqm),
        'beta': gsl_beta,
        '1/lambda': u'1/{} ({}/({}))'.format(gsl_lambda, degc, wpsqm),
        'rwf': 'RWF',
    }
    map_labels_wo_units = {
        'alpha': r'$\alpha$',
        'beta': r'$\beta$',
        '1/lambda': r'$1/\lambda$',
        'rwf': 'RWF',
        'a0*yrwf0': r'$A_{0} \kappa_{0}$',
        'a1*yrwf1': r'$A_{1} \kappa_{1}$',
        'a2*yrwf2': r'$A_{2} \kappa_{2}$',
        'a2': r'$A_{2}$',
        'yrwf2': r'$\kappa_{2}$',
    }
    p1.axis_labels(
        [{'xlabel': '',
          'ylabel': 'Fractional difference from CMIP5 mean'},
         {'xlabel': map_labels.get(yvars[1], yvars[1]),
          'ylabel': map_labels.get(yvars[0], yvars[0])},
         {'xlabel': '', 'ylabel': ''},
         {'xlabel': '', 'ylabel': ''},
         {'xlabel': map_labels.get(yvars[2], yvars[2]),
          'ylabel': map_labels.get(yvars[1], yvars[1])},
         {'xlabel': '', 'ylabel': ''},
         {'xlabel': map_labels.get(yvars[3], yvars[3]),
          'ylabel': map_labels.get(yvars[2], yvars[2])},
         {'xlabel': '', 'ylabel': 'Value relative to CMIP5 mean'},
         {'xlabel': '', 'ylabel': ''}],
        [[],
         [], ['xticklabels', 'yticklabels'], ['xticklabels', 'yticklabels'],
         [], ['xticklabels', 'yticklabels'],
         [],
         [], ['yticklabels']]
    )

    ax = fig.axes[0]
    ax.set_xticklabels([map_labels_wo_units.get(x, x) for x in yvars])
    ax.legend(handlelength=1., labelspacing=0.1)

    ax = fig.axes[6]
    handles, labels = ax.get_legend_handles_labels()
    legend_data = dict([x[::-1] for x in zip(handles, labels)])
    labels = ['CMIP5', 'CMIP6']
    handles = [legend_data[x] for x in labels]
    ax.legend(
        handles, labels, handlelength=1., labelspacing=0.1,
        bbox_to_anchor=(-0.4, 0.5), loc='center right')

    ax = fig.axes[7]
    ax.set_xticklabels([map_labels_wo_units.get(x, x) for x in xvars1])
    ax.legend(handlelength=1., labelspacing=0.1)

    ax = fig.axes[8]
    ax.set_xticklabels([map_labels_wo_units.get(x, x) for x in xvars2])
    ax.legend_.remove()

    for i, ax in enumerate(fig.axes):
        ax.text(
            0.02, 1.01, chr(97+i), ha='left', va='bottom', size=15,
            transform=ax.transAxes)

    return p1


def plot_rwf_ramp(asj, tauj):
    tp = np.arange(140) + 0.5
    nl = len(tauj)
    df = (asj * tauj).reshape((nl, 1)) / tp \
        * (1 - np.exp(-tp/tauj.reshape((nl, 1))))
    names_tau = ['tau{}'.format(i) for i in range(nl)]
    df = pd.DataFrame(df, names_tau).T

    tp2x = np.log(2.) / np.log(1.01)
    yrwf = asj * tauj / tp2x * (1. - np.exp(-tp2x/tauj))

    p1 = PlotSpace()
    fig = p1.newfigure()
    ax = fig.axes[0]

    ret = df.plot(kind='area', ylim=[0, 1], legend=False, alpha=0.4, ax=ax)
    ax.plot(
        tp2x, yrwf.sum(), ls='None', marker='x', color='k', label='2x point')

    # handles, labels = ax.get_legend_handles_labels()
    # map_labels = {
    #     'tau0': r'$\tau_{0}$', 'tau1': r'$\tau_{1}$', 'tau2': r'$\tau_{2}$'}
    # labels = [map_labels.get(x, x) for x in labels]
    # ax.legend(handles, labels)
    ax.legend()

    ax.set_xlabel('Year')
    ax.set_ylabel('Yet to be realized warming fraction')
    sns.despine(ax=ax)
    ax.grid()

    return p1

