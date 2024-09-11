import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

from .. import get_logger

logger = get_logger(__name__)

class PlotSpace:
    """Create a figure object that contains multiple Axes
    """
    def __init__(self, **kw):
        """Define the default Axes size with the following parameters
            'height': height, default 3.
            'aspect': aspect ratio, default 1.4
            'left': left margin, default 0.8
            'bottom': bottom margin, default 0.8
            'right': right margin, default 0.3
            'top': top margin, default 0.5
            'wspace': width spacing for multiple Axes, default 0.8
            'hspace': height spacing for multiple Axes, default 0.7
        The size is specified in figure units
        """
        kw_space = {
            'height': 3.,
            'aspect': 1.4,
            'left': 0.8,
            'bottom': 0.8,
            'right': 0.3,
            'top': 0.5,
            'wspace': 0.8,
            'hspace': 0.7,
        }
        kw_space.update(kw)
        self.kw_space = kw_space

        self.rects = [
            [
                kw_space['left'],
                kw_space['bottom'],
                kw_space['height'] * kw_space['aspect'],
                kw_space['height'],
            ],
        ]
        self._figure = None

    def append(self, direction='right', ref=-1, **kw):
        """Append a new Axes

        Parameters
        ----------
        direction, optional
            Direction to which the new Axes is placed.
            'right' (default), 'bottom', 'left', 'top', and 'here' are possible.
        ref, optional
            Index of reference Axes for placing direction, by default -1

        Axes size and spacing is adjusted by keyword arguments
        'height', 'aspect', 'xoff', and 'yoff'
        'xoff' and 'yoff' are used to change 'wspace' and 'hspace'
        or X and Y offset, depending on 'direction' parameter
        """
        kw_space = self.kw_space
        rects = self.rects

        height = kw.get('height', kw_space['height'])
        width = height * kw.get('aspect', kw_space['aspect'])
        # wspace = kw_space['wspace']
        wspace = kw.get('xoff', kw_space['wspace'])
        # hspace = kw_space['hspace']
        hspace = kw.get('yoff', kw_space['hspace'])
        xoff = kw.get('xoff', 0.)
        yoff = kw.get('yoff', 0.)

        left_ref, bottom_ref, width_ref, height_ref = rects[ref]

        if direction == 'right':
            # left = left_ref + width_ref + kw.get('xoff', wspace)
            left = left_ref + width_ref + wspace
            # bottom = bottom_ref + kw.get('yoff', 0.)
            bottom = bottom_ref + yoff
        elif direction == 'bottom':
            # left = left_ref + kw.get('xoff', 0.)
            left = left_ref + xoff
            # bottom = bottom_ref - height - kw.get('yoff', hspace)
            bottom = bottom_ref - height - hspace
        elif direction == 'left':
            # left = left_ref - width - kw.get('xoff', wspace)
            left = left_ref - width - wspace
            # bottom = bottom_ref + kw.get('yoff', 0.)
            bottom = bottom_ref + yoff
        elif direction == 'top':
            # left = left_ref + kw.get('xoff', 0.)
            left = left_ref + xoff
            # bottom = bottom_ref + height_ref + kw.get('yoff', hspace)
            bottom = bottom_ref + height_ref + hspace
        elif direction == 'here':
            # left = left_ref + kw.get('xoff', 0.)
            left = left_ref + xoff
            # bottom = bottom_ref + kw.get('yoff', 0.)
            bottom = bottom_ref + yoff

        rects.append( [left, bottom, width, height] )

    def get_rect_panels(self):
        """Calculate the position and size of the rectangle space
        that contains multiple Axes

        Returns
        -------
            [left, bottom, width, height] in figure units
        """
        kw_space = self.kw_space
        rects = np.array([rect[:4] for rect in self.rects])

        left = rects[:, 0].min() - kw_space['left']
        bottom = rects[:, 1].min() - kw_space['bottom']
        right = (rects[:, 0] + rects[:, 2]).max() + kw_space['right']
        top = (rects[:, 1] + rects[:, 3]).max() + kw_space['top']

        return [left, bottom, right-left, top-bottom]

    def newfigure(self):
        """Create a figure and add multiple Axes

        Returns
        -------
            Figure object
        """
        fig_left, fig_bottom, fig_width, fig_height = (
            self.get_rect_panels()
        )
        figure = plt.figure(figsize=(fig_width, fig_height))

        for rect in self.rects:
            left, bottom, width, height = rect[:4]
            projection = rect[4] if len(rect) > 4 else None

            figure.add_axes(
                [
                    (left - fig_left) / fig_width,
                    (bottom - fig_bottom) / fig_height,
                    width / fig_width,
                    height / fig_height,
                ],
                projection=projection,
            )

        self._figure = figure

        return figure
    
    def __str__(self):
        """Retrieve Figure and Axes boundaries in inches

        Returns
        -------
            Formatted text that describes the boundaries
        """
        bbox = self._figure.bbox_inches
        ret = [
            'figure: {:g} x {:g} at ({:g}, {:g})'
            .format(bbox.width, bbox.height, bbox.x0, bbox.y0)
        ]

        for i, rect in enumerate(self.rects):
            left, bottom, width, height = rect[:4]
            ret.append(
                'ax[{}]: {:g} x {:g} at ({:g}, {:g})'
                .format(i, width, height, left, bottom)
            )

        return '\n'.join(ret)

    def get_extents(self, *args):
        """Get the boundary points of multiple Axes
        Axes can be selected by indexes with positional arguments

        Returns
        -------
            Points array as extents (xmin, ymin, xmax, ymax)
        """
        if len(args) == 0:
            axes = self._figure.axes
        else:
            axes = [
                ax for i, ax in enumerate(self._figure.axes)
                if i in args
            ]
        pos = np.array([ax._position.extents for ax in axes])
        return np.array([
            pos[:, 0].min(), pos[:, 1].min(),
            pos[:, 2].max(), pos[:, 3].max(),
        ])
        

class PlotBase:
    """Custom base class for graph drawing
    """
    def __init__(self, **kw):
        """Define matplotlib and some function parameters
        """
        self.plot_space = None

        self.palettes = {
            # seaborn.color_palette('colorblind', as_cmap=True)
            'seaborn_colorblind': [
                '#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC',
                '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9',
            ],
            'ipcc_wg1': [
                '#{:02x}{:02x}{:02x}'.format(*v)
                for v in [
                    (0, 0, 0), (112, 160, 205), (196, 121, 0),
                    (178, 178, 178), (0, 52, 102), (0, 79, 0),
                ]
            ],
        }
        palette = kw.get('palette', 'seaborn_colorblind')
        colors = cycler(color=self.palettes[palette])

        self.rc = {
            'axes.prop_cycle': colors,

            'font.family': 'sans-serif',
            'font.sans-serif': 'DejaVu Sans',
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 12,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11,

            # 'axes.linewidth': 1.25,
            'axes.linewidth': 0.8,

            'axes.grid': False,
            # 'grid.linewidth': 1,
            'grid.linewidth': 0.8,
            'grid.linestyle': ':',

            # 'lines.linewidth': 1.5,
            'lines.linewidth': 1.0,
            'lines.markersize': 6,
            'patch.linewidth': 1,

            # 'xtick.major.width': 1.25,
            # 'ytick.major.width': 1.25,
            # 'xtick.minor.width': 1,
            # 'ytick.minor.width': 1,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.minor.width': 0.8,
            'ytick.minor.width': 0.8,

            # 'xtick.major.size': 6,
            # 'ytick.major.size': 6,
            # 'xtick.minor.size': 4,
            # 'ytick.minor.size': 4,
            'xtick.major.size': 5,
            'ytick.major.size': 5,
            'xtick.minor.size': 3,
            'ytick.minor.size': 3,

            'axes.facecolor': 'w',
            'xtick.direction': 'in',
            'ytick.direction': 'in',

            'mathtext.default': 'regular',
        }
        self.kw_savefig = {
            'transparent': False,
            'bbox_inches': 'tight',
            'pad_inches': 0.04,
            'dpi': 180,
        }
        self.rc.update(kw.get('rc', {}))
        self.update_rc()
        self.kw_savefig.update(kw.get('kw_savefig', {}))

    def clear(self):
        """Clear the existing figure
        """
        if self.plot_space is None:
            return
        self.figure.clear()
        self.plot_space = None
    
    def update_rc(self, **kw):
        """Update matplotlib rc parameters
        """
        rc = self.rc.copy()
        rc.update(kw)
        mpl.rcParams.update(rc)
        
    def despine(self, left=False, bottom=False, right=True, top=True):
        """Remove unnecessary spines

        Parameters
        ----------
        left, optional
            Remove or not the left spine, by default False
        bottom, optional
            Remove or not the bottom spine, by default False
        right, optional
            Remove or not the right spine, by default True
        top, optional
            Remove or not the top spine, by default True
        """
        axes = self.figure.axes
        for ax in axes:
            for side in ['left', 'bottom', 'right', 'top']:
                is_visible = not locals()[side]
                ax.spines[side].set_visible(is_visible)

            if left and not right:
                ax.yaxis.set_ticks_position('right')

            if bottom and not top:
                ax.xaxis.set_ticks_position('top')

    def init_general(self, extend=[], projection=None, **kw_space):
        """PlotSpace() wrapper for general configurations

        Parameters
        ----------
        extend, optional
            Specification of the second and subsequent Axes, by default []
            Default Axes size and spacing, applied to the first Axes, are
            specified by the keyword arguments
        projection, optional
            Projection to be applied to the specified or the whole Axes,
            by default None
        """
        self.clear()

        plot_space = PlotSpace(**kw_space)

        for direction, ref, kw in extend:
            plot_space.append(direction, ref, **kw)

        if isinstance(projection, dict):
            for k, v in projection.items():
                plot_space.rects[k].append(v)
        elif projection is not None:
            for rect in plot_space.rects:
                rect.append(projection)

        self.figure = plot_space.newfigure()
        self.plot_space = plot_space

        self.despine()

    def init_regular(self, n=1, col=1, kw_space={}):
        """PlotSpace() wrapper for the typical configuration,
        where multiple Axes with the same size are regularly aligned 

        Parameters
        ----------
        n, optional
            Total number of Axes, by default 1
        col, optional
            Number of horizontally aligned Axes, by default 1
        kw_space, optional
            Parameters passed to PlotSpace(), by default {}
        """
        self.clear()
        plot_space = PlotSpace(**kw_space)

        i = 1
        j = 1

        while 1:
            if (i-1)*col + j == n: break

            if j < col:
                plot_space.append('right')
                j += 1
            else:
                plot_space.append('bottom', ref=-col)
                i += 1
                j = 1

        self.figure = plot_space.newfigure()
        self.plot_space = plot_space

        self.despine()

    def close(self):
        """Close the existing figure
        """
        plt.close(self.figure)

    def get_axes(self, *args, **kw):
        """Get Axes corresponding to given indexes or the whole Axes

        Returns
        -------
            Axes or the list of Axes
        """
        squeeze = kw.get('squeeze', True)

        if len(args) == 0:
            axes = self.figure.axes
        else:
            axes = [self.figure.axes[i] for i in args]

        if squeeze and len(axes) == 1:
            axes = axes[0]

        return axes

    def __call__(self, *args, **kw):
        """Mapped to get_axes()

        Returns
        -------
            Axes or the list of Axes
        """
        return self.get_axes(*args, **kw)

    def savefig(self, path, **kw):
        """Wrapper for figure.savefig()
        Default parameters defined by self.kw_savefig and updates
        specified by the keyword arguments are passed

        Parameters
        ----------
        path
            Output file path with supported file name extension,
            such as 'png', 'svg', etc.
        """
        kw_savefig = self.kw_savefig.copy()
        kw_savefig.update(kw)
        logger.info(f'saved to {path}')
        self.figure.savefig(path, **kw_savefig)

    def panel_label(self, axes=[], **kw):
        """Add panel labels

        Parameters
        ----------
        axes, optional
            Axes indexes to be passed to get_axes(), by default []

        Keyword arguments
            'xy', default (0., 1.)
            'xytext', default (8, 3)
            'ha', default 'left'
            'va', default 'bottom'
            'size', default 'x-large'
            'mktext', default lambda n: chr(97+n)
            'textprop', default {}
        """
        xy = kw.get('xy', (0., 1.))
        xytext = kw.get('xytext', (8, 3))
        ha = kw.get('ha', 'left')
        va = kw.get('va', 'bottom')
        size = kw.get('size', 'x-large')
        mktext = kw.get('mktext', lambda n: chr(97+n))
        textprop = kw.get('textprop', {})

        axes = self.get_axes(*axes, **{'squeeze': False})

        for n, ax in enumerate(axes):
            ax.annotate(
                mktext(n), xy=xy, xytext=xytext,
                xycoords='axes fraction', textcoords='offset points',
                ha=ha, va=va, fontsize=size, **textprop,
            )

    def axis_share(self, axis, axes=[], **kw):
        """Align axis ranges

        Parameters
        ----------
        axis
            Axis to be aligned, 'x', 'y', or 'both'
        axes, optional
            Axes indexes to be passed to get_axes(), by default []
        """
        axes = self.get_axes(*axes, **{'squeeze': False})

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

    def make_twin(self, axhost, spine, **kw):
        """Create a twin Axes sharing the x- or y-axis

        Parameters
        ----------
        axhost
            Host Axes
        spine
            Spine to be visible, which determines the shared axis

        Returns
        -------
            Newly created Axes
        """
        offset = kw.get('offset')

        if not isinstance(axhost, mpl.axes.Axes):
            axhost = self.get_axes(axhost)

        if spine in ['left', 'right']:
            ax = axhost.twinx()
            axis = ax.yaxis
        elif spine in ['bottom', 'top']:
            ax = axhost.twiny()
            axis = ax.xaxis

        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for k, sp in ax.spines.items():
            sp.set_visible(k == spine)

        if offset is not None:
            ax.spines[spine].set_position(('outward', offset))

        axis.set_visible(True)
        axis.set_ticks_position(spine)
        axis.set_label_position(spine)

        return ax

    def get_fig_position_relto_axes(self, p, *args):
        """Get a figure position relative to multiple axes

        Parameters
        ----------
        p
            Position relative to multiple axes

        Returns
        -------
            Figure position
        """
        axes = self.get_axes(*args, **{'squeeze': False})

        bounds = np.vstack([
            (ax.transAxes + ax.figure.transFigure.inverted())
            .transform([(0., 0.), (1., 1.)]).ravel()
            for ax in axes
        ])

        x0 = bounds[:, 0].min()
        y0 = bounds[:, 1].min()
        x1 = bounds[:, 2].max()
        y1 = bounds[:, 3].max()

        return (
            x0 + (x1 - x0) * p[0],
            y0 + (y1 - y0) * p[1],
        )
