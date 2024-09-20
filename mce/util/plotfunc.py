import numpy as np
import pandas as pd
import seaborn as sns
from .plot_base import PlotBase
from .plot_util import unicode_character

def plot_fitting(myplt, gcm, forcing, irm, names, px):
    """Draw N and T time series, and their scatter diagram

    Parameters
    ----------
    myplt
        Plotting module object
    gcm
        GCM data
    forcing
        Forcing module object
    irm
        Climate module object
    names
        Names for dataset and variables used in the fitting
    px
        Additional parameters: lambda_reg, ecs, tcr, ecs_reg, tcr_gcm
    """
    height = 2.5
    aspect = 1.4
    wspace = 0.9
    hspace = 0.4
    width = height * aspect

    myplt.init_general(
        height=height, aspect=aspect, wspace=wspace, hspace=hspace,
        extend=[
            ('bottom', -1, {}),
            (
                'right', -1,
                {'height': width, 'aspect': 1., 'yoff': height*2+hspace-width},
            ),
        ],
    )
    for ax in myplt.figure.axes:
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
    lamb = irm.parms.lamb

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

    ax = myplt(0)
    ax.plot(time_4x, gcm['4x_n'], label='GCM 4x', **linestyle['GCM 4x'])
    ax.plot(time_1p, gcm['1p_n'], label='GCM 1%', **linestyle['GCM 1%'])
    ax.plot(time_4xi, irm_4x_n, label='IRM 4x', **linestyle['IRM 4x'])
    ax.plot(time_1pi, irm_1p_n, label='IRM 1%', **linestyle['IRM 1%'])
    ax.legend(**kw_legend)
    ax.set_ylabel(map_labels[names['var_n']])

    ax = myplt(1)
    ax.plot(time_4x, gcm['4x_t'], label='GCM 4x', **linestyle['GCM 4x'])
    ax.plot(time_1p, gcm['1p_t'], label='GCM 1%', **linestyle['GCM 1%'])
    ax.plot(time_4xi, irm_4x_t, label='IRM 4x', **linestyle['IRM 4x'])
    ax.plot(time_1pi, irm_1p_t, label='IRM 1%', **linestyle['IRM 1%'])
    if 'tcr' in px:
        ax.plot(tp70, px['tcr'], label='TCR', **linestyle['TCR'])
    ax.legend(**kw_legend)
    ax.set_xlabel('Year')
    ax.set_ylabel(map_labels[names['var_t']])

    ax = myplt(2)
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

    tauj = irm.parms.tauj.tolist()
    asj = irm.parms.asj.tolist()
    nl = len(tauj)
    
    result_text = [
        'IRM-{} fitted to {} and {} of {}'
        .format(len(tauj), names['var_n'], names['var_t'], names['dataset']),
        u'alpha: {:.2f} {}, beta: {:.2f}'.format(
            forcing.parms.alpha, wpm2, forcing.parms.beta),
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
    space = myplt.plot_space.kw_space
    xpos = (space['left'] + width + wspace + 0.02) / (
        space['left'] + 2*width + wspace + space['right'])
    ypos = (space['bottom'] + 1.) / (
        space['bottom'] + 2*height + hspace + space['top'])
    myplt.figure.text(
        xpos, ypos, '\n'.join(result_text), ha='left', va='top', size=10,
    )

    myplt.panel_label(
        xy=(-0.12, 1.), xytext=(0., 0.), 
        ha='right', va='top', size=17,
    )


def plot_tcr_ecs(myplt, parms, ecs_conv=True, **kw):
    """Draw TCR-ECS scatter diagrams

    Parameters
    ----------
    myplt
        Plotting module object
    parms
        Thermal response parameters calibrated to CMIP models
    ecs_conv, optional
        If True, it draws two panels with conventional and new ECS estimates;
        otherwise one panel for the latter
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

    if ecs_conv:
        ext = [
            ('right', -1, {}),
            (
                'top', 0,
                {'height': height/aspect2, 'aspect': aspect2, 'yoff': 0.},
            ),
            (
                'top', 1,
                {'height': height/aspect2, 'aspect': aspect2, 'yoff': 0.},
            ),
            ('right', 1, {'aspect': 1./aspect2, 'xoff': 0.}),
        ]
    else:
        ext = [
            (
                'top', 0,
                {'height': height/aspect2, 'aspect': aspect2, 'yoff': 0.},
            ),
            ('right', 0, {'aspect': 1./aspect2, 'xoff': 0.}),
        ]
    myplt.init_general(
        extend=ext, height=height, aspect=aspect, wspace=0.2, hspace=0.2,
    )
    legend_data = {}

    for i, ax in enumerate(myplt.figure.axes):
        if i > (ecs_conv and 1 or 0):
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

    df = parms.groupby('mip')
    mipnames = list(df.groups)

    xvars = ['ecs_reg', 'ecs']
    yvar = 'tcr'
    label_refline = '0.6-to-1'

    if not ecs_conv:
        del xvars[0]
        del xlabels[0]

    for i, xvar in enumerate(xvars):
        ax = myplt(i)

        for j, mip in enumerate(mipnames):
            color = 'C{}'.format(j)
            sns.regplot(
                x=xvar, y=yvar, data=df.get_group(mip), color=color, ax=ax,
                scatter_kws={'edgecolor': 'w', 's': 50}, label=mip,
            )
            h, l = ax.get_legend_handles_labels()
            legend_data.update(dict([x[::-1] for x in zip(h, l)]))

        if i == 0:
            ax.set_xlabel(xlabels[0])
            ax.set_ylabel(ylabel)
        else:
            ax.set_xlabel(xlabels[1])
            ax.set_ylabel(None)
            ax.tick_params(axis='y', labelleft=False)

    nb = len(xvars)
    myplt.axis_share(axis='both', axes=list(range(nb)))
    xlim = myplt(0).get_xlim()
    ylim = myplt(0).get_ylim()

    for n in range(nb):
        ax = myplt(n)
        ax.grid(True)
        ax.plot(
            np.array(xlim), 0.6*np.array(xlim), ls='--', color='0.2',
            label=label_refline,
        )
        h, l = ax.get_legend_handles_labels()
        legend_data.update(dict([x[::-1] for x in zip(h, l)]))

    box_kws = {
        'saturation': 1,
        'width': 0.5, 'showmeans': True,
        'meanprops': {'markerfacecolor': 'none', 'markeredgecolor': 'k'},
    }

    for i, xvar in enumerate(xvars):
        ax = myplt(nb+i)
        ax.set_xlim(*xlim)
        sns.boxplot(
            x=xvar, y='mip', data=parms, orient='horizontal',
            hue='mip', palette=['C0', 'C1'],
            ax=ax, **box_kws,
        )
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        if i == 0:
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.tick_params(axis='y', left=False)
        else:
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.tick_params(axis='y', left=False, labelleft=False)

    ax = myplt(2*nb)
    ax.set_ylim(*ylim)
    sns.boxplot(
        x='mip', y='tcr', data=parms, orient='vertical',
        hue='mip', palette=['C0', 'C1'],
        ax=ax, **box_kws,
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.tick_params(axis='x', bottom=False, labelrotation=60)
    ax.tick_params(axis='y', left=False, labelleft=False)

    labels = mipnames + [label_refline]
    handles = [legend_data[k] for k in labels]
    ax = myplt(0)
    ax.legend(handles, labels, labelspacing=0.)

    if ecs_conv:
        myplt.panel_label(
            xy=(0.98, 0.02), xytext=(0., 0.),
            ha='right', va='bottom', size=17, axes=[0, 1],
        )


def plot_parms_rel(myplt, df, **kw):
    """Draw distributions of parameter values

    Parameters
    ----------
    myplt
        Plotting module object
    df
        Thermal response parameters calibrated to CMIP models
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
    myplt = PlotBase()
    ext = [
        (
            'right', -1,
            {
                'height': height2, 'aspect': aspect2,
                'xoff': 2 * wspace, 'yoff': height - height2,
            },
        ),
        ('right', -1, {'height': height2, 'aspect': aspect2}),
        ('right', -1, {'height': height2, 'aspect': aspect2}),
        ('bottom', -2, {'height': height2, 'aspect': aspect2}),
        ('right', -1, {'height': height2, 'aspect': aspect2}),
        ('bottom', -1, {'height': height2, 'aspect': aspect2}),
        ('bottom', 0, {'aspect': aspect1*(15./12.), 'yoff': hspace2}),
        ('right', -1, {'aspect': aspect1*(10./12.)}),
    ]
    myplt.init_general(
        height=height, aspect=aspect1, wspace=wspace, hspace=hspace,
        extend=ext,
    )

    yvars = ['alpha', 'beta', '1/lambda', 'rwf']
    box_kws = {
        'width': 0.65, 'showmeans': True,
        'meanprops': {'markerfacecolor': 'none', 'markeredgecolor': 'k'} }

    df1 = dfb[yvars+['mip']].melt(id_vars='mip')
    k = 0
    ax = myplt(k)
    sns.boxplot(
        x='variable', y='value', hue='mip', data=df1, ax=ax, **box_kws)
    k = k+1

    df_corr = df[yvars].corr()
    text_minus = unicode_character('minus')

    for i in range(len(yvars)):
        for j in range(i+1, len(yvars)):
            ax = myplt(k)
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
                
    ax = myplt(k)
    xvars1 = ['a0*yrwf0', 'a1*yrwf1', 'a2*yrwf2']
    df1 = dfa[xvars1+['class']].melt(id_vars=['class'])
    sns.boxplot(
        x='variable', y='value', hue='class', data=df1, ax=ax, **box_kws)

    ax = myplt(k+1)
    xvars2 = ['a2', 'yrwf2']
    df1 = dfa[xvars2+['class']].melt(id_vars=['class'])
    sns.boxplot(
        x='variable', y='value', hue='class', data=df1, ax=ax, **box_kws)

    myplt.axis_share(axis='y', axes=list(range(7, len(myplt.figure.axes))))

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
    ax = myplt(0)
    ax.set(
        xlabel=None,
        ylabel='Fractional difference from CMIP5 mean',
    )
    ax = myplt(1)
    ax.set(
        xlabel=map_labels.get(yvars[1], yvars[1]),
        ylabel=map_labels.get(yvars[0], yvars[0]),
    )
    ax = myplt(2)
    ax.set(xlabel=None, ylabel=None)
    ax.tick_params(axis='both', labelleft=False, labelbottom=False)
    ax = myplt(3)
    ax.set(xlabel=None, ylabel=None)
    ax.tick_params(axis='both', labelleft=False, labelbottom=False)
    ax = myplt(4)
    ax.set(
        xlabel=map_labels.get(yvars[2], yvars[2]),
        ylabel=map_labels.get(yvars[1], yvars[1]),
    )
    ax = myplt(5)
    ax.set(xlabel=None, ylabel=None)
    ax.tick_params(axis='both', labelleft=False, labelbottom=False)
    ax = myplt(6)
    ax.set(
        xlabel=map_labels.get(yvars[3], yvars[3]),
        ylabel=map_labels.get(yvars[2], yvars[2]),
    )
    ax = myplt(7)
    ax.set(xlabel=None, ylabel='Value relative to CMIP5 mean')
    ax = myplt(8)
    ax.set(xlabel=None, ylabel=None)
    ax.tick_params(axis='y', labelleft=False)

    ax = myplt(0)
    ax.set_xticks(ax.get_xticks()) # to avoid warning from set_xticklabels()
    ax.set_xticklabels([map_labels_wo_units.get(x, x) for x in yvars])
    ax.legend(handlelength=1., labelspacing=0.1)

    ax = myplt(6)
    handles, labels = ax.get_legend_handles_labels()
    legend_data = dict([x[::-1] for x in zip(handles, labels)])
    labels = ['CMIP5', 'CMIP6']
    handles = [legend_data[x] for x in labels]
    ax.legend(
        handles, labels, handlelength=1., labelspacing=0.1,
        bbox_to_anchor=(-0.4, 0.5), loc='center right')

    ax = myplt(7)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([map_labels_wo_units.get(x, x) for x in xvars1])
    ax.legend(handlelength=1., labelspacing=0.1)

    ax = myplt(8)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([map_labels_wo_units.get(x, x) for x in xvars2])
    ax.legend_.remove()

    myplt.panel_label(
        xy=(0.02, 1.01), xytext=(0., 0.),
        ha='left', va='bottom', size=15,
    )
