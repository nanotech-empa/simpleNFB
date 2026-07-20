'''
DAT_browser.py
--------------
spectrumBrowser widget for Nanonis DAT spectroscopy data in Jupyter notebooks.
Renders with matplotlib on the ipympl (widget) backend.
'''

import bisect
import os
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ipywidgets as widgets
import matplotlib as mpl
import numpy as np
from IPython import display
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from scipy.interpolate import interp1d
from scipy.signal import medfilt, savgol_filter

from spmpy import Spm
from .base_browser import BaseBrowser
from .process_utils import (rebin_intensity_nm_to_ev, smooth_data, group_average,
                             relative_position, despike_z_score, moving_average)
from .widget_helpers import HBox, VBox, Btn_Widget, Text_Widget

# Colormap catalogue from the matplotlib registry ('_r' variants omitted —
# reversal is a name suffix, not a separate map). Qualitative = small discrete
# ListedColormaps (tab10, Set1, …); everything else is treated as sequential.
_ALL_NAMES  = sorted((n for n in mpl.colormaps if not n.endswith('_r')),
                     key=str.lower)
_QUAL_CMAPS = [n for n in _ALL_NAMES
               if isinstance(mpl.colormaps[n], ListedColormap)
               and mpl.colormaps[n].N <= 20]
_SEQ_CMAPS  = [n for n in _ALL_NAMES if n not in _QUAL_CMAPS]
_ALL_CMAPS  = _QUAL_CMAPS + _SEQ_CMAPS

# plotly marker names kept in the UI for continuity → matplotlib marker codes
_MARKER_MAP = {'circle': 'o', 'star': '*', 'square': 's',
               'triangle-up': '^', 'x': 'x'}

# F6: Spm.__init__ mutates the *global* yaml.SafeLoader via add_implicit_resolver
# on every call. Under ThreadPoolExecutor that is a data race and the resolver
# list grows unbounded. Serialise object *construction* only; get_channel reads
# stay parallel.
_SPM_INIT_LOCK = threading.Lock()


def _make_spm(path):
    """Construct an Spm under the global init lock (F6)."""
    with _SPM_INIT_LOCK:
        return Spm(path)


class fileBrowser(BaseBrowser):
    '''
    Interactive browser for Nanonis DAT spectroscopy data.

    Public attributes:
        figure    – matplotlib Figure (self.canvas is the ipympl widget)
        ax        – main Axes
        spec      – list of Spm objects for loaded files
        spec_data – list of ndarrays (raw / processed y-data)
        spec_x    – list of ndarrays (x-data per spectrum)

    Key methods:
        update_axes()       – refresh plot from current spec_data / spec_x
        update_image_data() – reload and process channel data
        save_figure(a)      – save figure to browser_outputs/
        save_data(a)        – save CSV to browser_outputs/
    '''

    _TEMPLATES_SUBDIR: str = 'dat'

    _INFO_BUILDERS: dict = {
        'STML':               '_info_stml',
        'bias spectroscopy':  '_info_bias_spec',
        'THz amplitude sweep':'_info_thz',
        'Z spectroscopy':     '_info_z_spec',
        'History Data':       '_info_history',
    }

    def __init__(self, width: int = 700, height: int = 550, fontsize: int = 8,
                 titlesize: int = 5, cmap: str = 'greys', home_directory: str = './',
                 sxmBrowser=None) -> None:
        # --- state ---
        self.img           = None
        self._make_figure(width, height)             # figure / canvas / ax
        # Constrained layout = matplotlib's equivalent of plotly automargin:
        # margins are recomputed from rendered decoration extents (multi-line
        # title, tick/axis labels, colorbar, secondary axis) on every draw,
        # inside the fixed figWidth×figHeight canvas. No manual margin math.
        self.figure.set_layout_engine('constrained')
        # Permanent 2-cell grid (docs: "Manual placement of colorbars", cax=):
        # ax lives in cell 0; the condensed/2D colorbar lives in a PERSISTENT
        # cax in cell 1. Renders only toggle cax visibility and re-shape the
        # grid — no axes are ever created/removed, so plot space cannot leak.
        # (Verified cycle-stable under constrained layout, mpl 3.10.)
        _gs = self.figure.add_gridspec(1, 2, width_ratios=[width / self._DPI, 1.5])
        self.ax.set_subplotspec(_gs[0])
        self._cax = self.figure.add_subplot(_gs[1])
        self._cax.set_visible(False)
        self._grid_state = None
        self._fig_width, self._fig_height = width, height
        self._sec_ax = self._cbar = self._mesh = None
        self._lines: list  = []
        self._spm_cache: dict = {}                   # F6: path -> (mtime, Spm)
        self.sxmBrowser    = sxmBrowser
        self.fontsize      = fontsize
        self.titlesize     = titlesize
        cmap_lower = cmap.lower()
        self.colorMap = next((n for n in _ALL_CMAPS if n.lower() == cmap_lower), 'Greys')
        self.spec_x        = [np.linspace(-2, 2, 64)]
        self.spec_data     = [np.zeros(64)]
        self.spec_info     = [{'x_unit': 'N', 'y_unit': 'a.u.', 'x_label': 'Index'}]
        self.spec_label    = ''
        self.plasmonInfo   = {'file': None, 'spm': None, 'interp': None}
        self.labels        = []
        self.legendFontsize = [8, 6, 4, 3]
        self.errors        = []
        self.spec_index    = [0]
        self.default_channel = {
            'STML':             ['Wavelength', 'Intensity'],
            'bias spectroscopy':['V', 'dIdV'],
            'Z spectroscopy':   ['zrel', 'I'],
            'History Data':     ['Index', 'I'],
        }
        self.loaded_experiments = None
        self.current_experiment = None
        self.active_dir    = Path(home_directory)
        self.sxm_files     = []
        self.dat_files     = []
        self.directories   = [self.active_dir]
        self._scan_cache: dict = {}
        self._current_selection: tuple = ()  # §7: selection-diff guard
        self._auto_condensed: bool = False
        self._redraw_handle = None
        self._data_dirty:   bool = False

        # Initial placeholder trace
        line, = self.ax.plot(self.spec_x[0], self.spec_data[0])
        self._lines = [line]

        self._build_widgets()
        self._build_layout()
        self._connect_observers()
        self._apply_figure_layout()   # apply default Figure Settings

        # sxmBrowser coupling must happen after widgets are built
        if sxmBrowser is None:
            self.referenceLocBtn.disabled = True

        self.display()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        """Instantiate all ipywidgets; no observers set here."""
        L, FL, FLB, FLH = self._layout_helpers()
        self._build_common_file_widgets()

        # selections
        self.selectionList = widgets.SelectMultiple(
            options=self.dat_files, value=[], description='', rows=30, layout=FL(98),
            continuous_update=False)
        self.filterSelection = widgets.SelectMultiple(
            options=['all', 'dIdV', 'Z-Spectroscopy', 'stml', 'History'],
            value=['all'], description='', rows=5, layout=FL(98))
        self.newFilterText = widgets.Text(
            description='', tooltip='User-defined string for file filtering', layout=FL(50))
        self.addFilterBtn = widgets.Button(
            description='+', tooltip='Add new filter to selection', layout=FLB(24))

        self.channelXSelect = widgets.Dropdown(
            options=['Index'], value=None, description='X',
            layout=FL(98), style={'description_width': '30px'})
        self.channelYSelect = widgets.SelectMultiple(
            options=[None], value=[None], description='Y',
            rows=5, layout=FL(98), style={'description_width': '30px'})

        self.refreshBtn = Btn_Widget('', icon='refresh', tooltip='Reload file list',
                                     layout=L(30))

        self.saveNote     = Text_Widget('', description='note',
                                        layout=FL(98), style={'description_width': '30px'})

        # image-control buttons
        self.flattenBtn      = widgets.ToggleButton(description='', value=False, layout=L(30),
                                                    icon='barcode',
                                                    tooltip='Normalize each curve to max=1')
        self.fixZeroBtn      = widgets.ToggleButton(description='', value=False, layout=L(30),
                                                    icon='neuter', tooltip='Subtract local baseline')
        self.referenceLocBtn = Btn_Widget('', layout=L(30), icon='map-marker',
                                          tooltip='Plot tip location on image browser')
        self.saveBtn         = widgets.ToggleButton(
            value=True, description='', layout=L(30), icon='file-image-o',
            tooltip='Save to file when copying (toggle off to copy to clipboard only)')
        self.copyBtn         = Btn_Widget('', layout=L(30), icon='clipboard',
                                          tooltip='Copy figure to clipboard')
        self.csvBtn          = Btn_Widget('', layout=L(30), icon='list-ul',
                                          tooltip='Save data to browser_outputs/ as .csv')

        # colormap / marker
        _init_seq = self.colorMap in _SEQ_CMAPS or self.colorMap not in _QUAL_CMAPS
        self.cmapCategory = widgets.ToggleButtons(
            options=['Discrete', 'Sequential'], value='Sequential' if _init_seq else 'Discrete',
            description='', layout=FL(98), style={'button_width': 'auto'})
        _init_opts = _SEQ_CMAPS if _init_seq else _QUAL_CMAPS
        _init_val  = self.colorMap if self.colorMap in _init_opts else _init_opts[0]
        self.cmapSelection = widgets.Dropdown(
            description='colormap:', options=_init_opts, value=_init_val,
            layout=FL(98), style={'description_width': '80px'})
        self.markerSelection = widgets.Dropdown(
            description='marker:',
            options=['N', 'circle', 'star', 'square', 'triangle-up', 'x'],
            value='circle', layout=FL(98), style={'description_width': '80px'})

        # settings panel toggle
        self.settingsBtn = widgets.ToggleButton(description='', icon='gear', value=True,
                                                tooltip='Display settings panel', layout=L(30))
        self.codeBtn = Btn_Widget('', icon='file-code-o', tooltip='Export code snippet to new cell',
                                  layout=L(30))

        # --- settings panel widgets ---
        # title settings
        self.titleLabel      = widgets.Label(value='Figure Title Settings', layout=FLB(98))
        self.titleToggle     = widgets.ToggleButton(value=True, description='Show Title',
                                                     tooltip='Toggle figure title', layout=FLB(98))
        self.setpointToggle  = widgets.ToggleButton(value=True, description='Setpoint', layout=FLB(98))
        self.feedbackToggle  = widgets.ToggleButton(value=True, description='Feedback', layout=FLB(98))
        self.locationToggle  = widgets.ToggleButton(value=True, description='file location', layout=FLB(98))
        self.depthSelection  = widgets.Dropdown(value='full', options=['full', 1, 2, 3, 4, 5],
                                                description='Depth:', layout=FLB(98),
                                                style={'description_width': 'initial'})
        self.nameToggle      = widgets.ToggleButton(value=True, description='Filename', layout=FLB(98))
        self.dateToggle      = widgets.ToggleButton(value=True, description='Date', layout=FLB(98))

        # legend settings
        self.legendModeToggle = widgets.ToggleButtons(
            options=['Parameter', 'Custom', 'Condensed'], value='Parameter',
            description='', layout=FL(98), style={'button_width': 'auto'})
        self.legendToggle = widgets.ToggleButton(value=True, description='legend', layout=FLB(98))
        self.parameterLegendList = widgets.Dropdown(
            options=['Filename', 'Z (m)', 'Current [A]', 'Bias [V]',
                     'Exposure Time [ms]', 'Center Wavelength [nm]', 'Selected Grating'],
            value='Filename', description='Parameter:', layout=FL(98),
            style={'description_width': '80px'})
        # Custom mode widgets — hidden initially (Parameter is default)
        self.legendText   = widgets.Select(value='', options=[''], rows=5,
                                           layout=widgets.Layout(display='none', width='98%'))
        self.legendEntry  = widgets.Text(description='label:', tooltip='New legend text',
                                          layout=widgets.Layout(display='none', width='98%'),
                                          style={'description_width': '40px'},
                                          continuous_update=False)
        self.legendUpdate = widgets.Button(description='Update',
                                            layout=widgets.Layout(display='none', width='98%'))

        # filter settings
        self.offsetToggle    = widgets.ToggleButton(value=False, description='Offset', layout=FLB(50))
        self.offsetSize      = widgets.FloatText(value=0.1e-12, description='amt:',
                                                  step=.1e-12, readout_format='.1e',
                                                  layout=FL(48), style={'description_width': '28px'})
        self.svgToggle  = widgets.ToggleButton(value=False, description='Savitsky-Golay', layout=FLB(44))
        self.svgSize    = widgets.BoundedIntText(description='w:', value=3, min=3, max=101, step=2,
                                                  layout=FL(28), style={'description_width': '18px'})
        self.svgOrder   = widgets.BoundedIntText(description='o:', value=1, min=1, max=5, step=1,
                                                  layout=FL(26), style={'description_width': '18px'})
        self.medFiltBtn  = widgets.ToggleButton(description='Median', value=False, layout=FLB(74))
        self.medFiltSize = widgets.BoundedIntText(description='', value=3, min=3, max=21, step=2,
                                                   layout=FLB(24))
        self.despikeBtn       = widgets.ToggleButton(description='Despike', value=False, layout=FLB(44),
                                                      tooltip='Modified Z-score spike removal')
        self.despikeWindow    = widgets.BoundedIntText(description='w:', value=10, min=2, max=100,
                                                        step=1, layout=FL(28),
                                                        style={'description_width': '18px'})
        self.despikeThreshold = widgets.FloatText(description='t:', value=3.0, step=0.5,
                                                   layout=FL(26),
                                                   style={'description_width': '14px'})
        self.movAvgBtn  = widgets.ToggleButton(description='Mov. Avg', value=False, layout=FLB(74),
                                                tooltip='Boxcar moving average')
        self.movAvgSize = widgets.BoundedIntText(description='', value=5, min=1, max=101, step=2,
                                                  layout=FLB(24))
        self.thresholdToggle = widgets.ToggleButton(value=False, description='Threshold', layout=FLB(50))
        self.thresholdValue  = widgets.FloatText(value=100, description='val:',
                                                  layout=FL(48), style={'description_width': '28px'})
        self.averageToggle   = widgets.ToggleButton(value=False, description='Average', layout=FLB(50))
        self.groupSize       = widgets.BoundedIntText(description='grp:', value=3, min=3, max=20,
                                                       step=1, layout=FL(48),
                                                       style={'description_width': '28px'})

        # STML mode settings
        self.stmlToggle           = widgets.ToggleButton(value=False, description='STML Mode', layout=FLB(98))
        self.normalizeTimeBtn     = widgets.ToggleButton(value=True, description='Norm. Time', layout=FLB(98))
        self.normalizeCurrentBtn  = widgets.ToggleButton(value=True, description='Norm. Current', layout=FLB(98))
        self.normalizeEnergyBtn   = widgets.ToggleButton(value=True, description='Norm. Energy', layout=FLB(98))
        self.normalizePlasmonBtn  = widgets.ToggleButton(value=False, description='Norm. Plasmon', layout=FLB(98))
        self.plasmonReference     = widgets.Dropdown(options=['None'], value='None',
                                                     description='Plasmon:', layout=FL(98),
                                                     style={'description_width': '60px'})

        # axes controls
        self.xLimitsBtn = widgets.Button(description='Update X', layout=FLB(74))
        self.xLimitsMin = widgets.FloatText(value=-1, description='Min',
                                             layout=FL(98), style={'description_width': '25px'})
        self.xLimitsMax = widgets.FloatText(value=1, description='Max',
                                             layout=FL(98), style={'description_width': '25px'})
        self.yLimitsBtn = widgets.Button(description='Update Y', layout=FLB(74))
        self.yLimitsMin = widgets.FloatText(value=-1, description='Min',
                                             layout=FL(98), style={'description_width': '25px'})
        self.yLimitsMax = widgets.FloatText(value=1, description='Max',
                                             layout=FL(98), style={'description_width': '25px'})
        self.xLimitLock = widgets.ToggleButton(value=False, description='', icon='lock',
                                                layout=FLB(24))
        self.yLimitLock = widgets.ToggleButton(value=False, description='', icon='lock',
                                                layout=FLB(24))

        # 1D plot controls
        self.xLabel1D = widgets.Text(
            description='X label:', value='', placeholder='auto',
            layout=FL(98), style={'description_width': '52px'})
        self.yLabel1D = widgets.Text(
            description='Y label:', value='', placeholder='auto',
            layout=FL(98), style={'description_width': '52px'})
        self.xScaleMode = widgets.ToggleButtons(
            options=['Linear', 'Log', 'Custom'], value='Linear',
            description='', layout=FL(98), style={'button_width': 'auto'})
        self.yScaleMode = widgets.ToggleButtons(
            options=['Linear', 'Log', 'Custom'], value='Linear',
            description='', layout=FL(98), style={'button_width': 'auto'})
        # Formula fields hidden until Custom is selected
        self.xCustomFormula = widgets.Text(
            description='f(x):', value='x', placeholder='e.g. x * 1e3',
            layout=widgets.Layout(display='none', width='98%'),
            style={'description_width': '40px'})
        self.yCustomFormula = widgets.Text(
            description='f(y):', value='y', placeholder='e.g. np.abs(y)',
            layout=widgets.Layout(display='none', width='98%'),
            style={'description_width': '40px'})

        # 2D plot settings
        _p_opts = ['Index', 'Position (nm)', 'Z (m)', 'Current [A]', 'Bias [V]',
                   'Exposure Time [ms]', 'Center Wavelength [nm]', 'Selected Grating']
        self.plot2DToggle   = widgets.ToggleButton(value=False, description='2D View',
                                                    layout=FLB(98))
        self.plot2DYParam   = widgets.Dropdown(options=_p_opts, value='Index',
                                                description='param:', layout=FL(98),
                                                style={'description_width': '44px'})
        self.plot2DYLabel   = widgets.Text(description='Y lbl:', value='', placeholder='auto',
                                            layout=FL(98), style={'description_width': '44px'})
        self.plot2DYMin     = widgets.FloatText(description='min:', value=0,
                                                 layout=FL(48), style={'description_width': '28px'})
        self.plot2DYMax     = widgets.FloatText(description='max:', value=0,
                                                 layout=FL(48), style={'description_width': '28px'})
        self.plot2DXLabel   = widgets.Text(description='X lbl:', value='', placeholder='auto',
                                            layout=FL(98), style={'description_width': '44px'})
        self.plot2DXMin     = widgets.FloatText(description='min:', value=0,
                                                 layout=FL(48), style={'description_width': '28px'})
        self.plot2DXMax     = widgets.FloatText(description='max:', value=0,
                                                 layout=FL(48), style={'description_width': '28px'})
        self.plot2DUpdateBtn = widgets.Button(description='Update 2D', layout=FL(98))
        self.plot2DClimMode  = widgets.ToggleButtons(
            options=['Visible', 'Full', 'Custom'], value='Visible',
            description='', layout=FL(98), style={'button_width': 'auto'})
        self.plot2DVMin = widgets.FloatText(description='min:', value=0, disabled=True,
                                             layout=FL(48), style={'description_width': '28px'})
        self.plot2DVMax = widgets.FloatText(description='max:', value=0, disabled=True,
                                             layout=FL(48), style={'description_width': '28px'})
        # Header inspector: key list + read-only value
        self.headerKeySelect = widgets.Select(options=[], rows=8, layout=FL(98))
        self.headerValueText = widgets.Textarea(
            value='', disabled=True, rows=6, layout=FL(98))

        self._build_figure_settings_widgets(self._fig_width, self._fig_height)

        # Legend cell size (INCHES, same unit as W/H above), added to the
        # figure on the placement side while the legend/colorbar cell is in
        # use: width for Right placement, height for Bottom placement.
        self.figCdnLegendW = widgets.BoundedFloatText(
            value=1.5, min=0, max=6, step=0.1, description='Cdn legend (in):',
            tooltip='Inches added to the figure on the placement side while '
                    'the Condensed legend / 2D colorbar is shown',
            layout=widgets.Layout(width='98%'), style={'description_width': '96px'})
        self.figCdnLegendLoc = widgets.ToggleButtons(
            options=['Right', 'Bottom'], value='Right', description='Cdn loc:',
            tooltip='Condensed legend / colorbar placement',
            layout=widgets.Layout(width='98%'),
            style={'description_width': '96px', 'button_width': 'auto'})

    def _figure_settings_extras(self) -> list:
        """DAT contribution to the Figure Settings tab (see BaseBrowser)."""
        return [self.figCdnLegendW, self.figCdnLegendLoc]

    def _build_layout(self) -> None:
        """Assemble widgets into HBox/VBox/Accordion containers."""
        _, FL, FLB, FLH = self._layout_helpers()

        self.h_new_filter_layout = HBox(children=[
            widgets.Label('New', layout=FLB(24)), self.newFilterText, self.addFilterBtn])
        self.v_filter_layout = VBox(children=[
            widgets.Label('Filter', layout=FL(50)),
            self.filterSelection, self.h_new_filter_layout])
        self.h_process_layout = HBox(children=[
            self.flattenBtn, self.fixZeroBtn, self.referenceLocBtn],
            layout=FL(98))
        self.h_selection_btn_layout = HBox(children=[
            self.refreshBtn, self.csvBtn, self.saveBtn, self.copyBtn,
            self.codeBtn, self.settingsBtn],
            layout=FL(98))
        self.v_channel_layout = VBox(children=[
            self.channelXSelect, self.channelYSelect, self.saveNote], layout=FL(48))
        self.v_file_select_layout = VBox(children=[
            widgets.Label('Folder', layout=FL(98)),
            self.directorySelection,
            widgets.Label('Files', layout=FL(50)),
            self.selectionList,
            self.v_filter_layout],
            layout=FL(98))
        self.v_btn_layout = VBox(children=[
            self.h_selection_btn_layout, self.h_process_layout,
            self.cmapCategory, self.cmapSelection, self.markerSelection], layout=FL(48))
        self.h_user_layout = HBox(children=[
            self.v_channel_layout, self.v_btn_layout], layout=FLB(98))
        self.v_settings_layout = widgets.Accordion(children=[
            VBox(children=[
                self.legendModeToggle,
                self.legendToggle,
                self.parameterLegendList,
                self.legendText,
                self.legendEntry,
                self.legendUpdate,
            ], layout=FLH(98)),
            VBox(children=[
                self.titleToggle, self.nameToggle, self.setpointToggle,
                self.feedbackToggle, self.locationToggle, self.depthSelection,
                self.dateToggle], layout=FLH(98)),
            VBox(children=[
                HBox(children=[self.offsetToggle,    self.offsetSize],                     layout=FL(98)),
                HBox(children=[self.svgToggle,       self.svgSize,    self.svgOrder],      layout=FL(98)),
                HBox(children=[self.medFiltBtn,      self.medFiltSize],                    layout=FL(98)),
                HBox(children=[self.despikeBtn,      self.despikeWindow,
                               self.despikeThreshold],                                     layout=FL(98)),
                HBox(children=[self.movAvgBtn,       self.movAvgSize],                     layout=FL(98)),
                HBox(children=[self.thresholdToggle, self.thresholdValue],                 layout=FL(98)),
                HBox(children=[self.averageToggle,   self.groupSize],                      layout=FL(98)),
            ], layout=FLH(98)),
            VBox(children=[
                self.stmlToggle, self.normalizeTimeBtn, self.normalizeCurrentBtn,
                self.normalizeEnergyBtn, self.normalizePlasmonBtn,
                self.plasmonReference], layout=FLH(98)),
            VBox(children=[
                self.plot2DToggle,
                widgets.Label('── Y axis ──', layout=FL(98)),
                self.plot2DYParam,
                self.plot2DYLabel,
                HBox(children=[self.plot2DYMin, self.plot2DYMax], layout=FL(98)),
                widgets.Label('── X axis ──', layout=FL(98)),
                self.plot2DXLabel,
                HBox(children=[self.plot2DXMin, self.plot2DXMax], layout=FL(98)),
                widgets.Label('── Color scale ──', layout=FL(98)),
                self.plot2DClimMode,
                HBox(children=[self.plot2DVMin, self.plot2DVMax], layout=FL(98)),
                self.plot2DUpdateBtn,
            ], layout=FLH(98)),
            VBox(children=[
                widgets.Label('── Y axis ──', layout=FL(98)),
                self.yLabel1D,
                self.yScaleMode,
                self.yCustomFormula,
                self.yLimitsMin, self.yLimitsMax,
                HBox(children=[self.yLimitsBtn, self.yLimitLock], layout=FL(98)),
                widgets.Label('── X axis ──', layout=FL(98)),
                self.xLabel1D,
                self.xScaleMode,
                self.xCustomFormula,
                self.xLimitsMin, self.xLimitsMax,
                HBox(children=[self.xLimitsBtn, self.xLimitLock], layout=FL(98)),
            ], layout=FLH(98)),
            VBox(children=[
                widgets.Label('Header key', layout=FL(98)),
                self.headerKeySelect,
                widgets.Label('Value', layout=FL(98)),
                self.headerValueText,
            ], layout=FLH(98)),
            self._figure_settings_tab(),
        ], layout=FLH(98),
        titles=['Legend Settings', 'Title Settings', 'Filter Settings',
                'STML Mode', '2D Plot', '1D Plot', 'Header', 'Figure Settings'])

        # The ipympl Canvas is itself a widget — embed it directly.
        self.v_image_layout = VBox(children=[
            self.canvas, self.h_user_layout],
            layout=widgets.Layout(display='flex', flex_flow='column', justify_content='center', align_items='center', width='100%' ))
        self._build_main_layout(self.v_file_select_layout, self.v_image_layout, 5)

        # Output panel for code export — embedded in layout so it works in VS Code too
        self._code_out = widgets.Output(
            layout=widgets.Layout(width='100%', display='none'))
        self.h_main_layout.children = tuple(self.h_main_layout.children) + (self._code_out,)

    def _connect_observers(self) -> None:
        """Wire all observe() and on_click() callbacks."""
        # Display-only — patch existing artists in-place; no rebuild
        for w in (self.legendToggle, self.cmapSelection):
            w.observe(self._update_display, names='value')
        # parameterLegendList change may affect condensed colorbar labels → tier 1
        self.parameterLegendList.observe(self._schedule_redraw, names='value')

        # Tier 1 — filter/transform pipeline; debounced full rebuild
        for w in (self.plot2DToggle, self.plot2DYParam,
                  self.plot2DYLabel, self.plot2DXLabel,
                  self.offsetToggle, self.offsetSize,
                  self.medFiltBtn, self.medFiltSize,
                  self.despikeBtn, self.despikeWindow, self.despikeThreshold,
                  self.movAvgBtn, self.movAvgSize,
                  self.thresholdToggle, self.thresholdValue,
                  self.averageToggle, self.flattenBtn, self.fixZeroBtn):
            w.observe(self._schedule_redraw, names='value')

        # Title-only — patch axes title in-place; no rebuild
        for w in (self.titleToggle, self.nameToggle, self.setpointToggle,
                  self.feedbackToggle, self.locationToggle, self.depthSelection,
                  self.dateToggle):
            w.observe(self._update_title, names='value')
        # SVG filter affects data pipeline → tier 1
        for w in (self.svgToggle, self.svgSize, self.svgOrder):
            w.observe(self._schedule_redraw, names='value')

        # Tier 3 — full pipeline (data reload + render); debounced with dirty flag
        for w in (self.stmlToggle,
                  self.normalizeTimeBtn, self.normalizeCurrentBtn,
                  self.normalizeEnergyBtn, self.normalizePlasmonBtn):
            w.observe(self._schedule_redraw_dirty, names='value')

        # legend group
        self.groupSize.observe(self.update_legend_settings, names='value')
        self.groupSize.observe(self._schedule_redraw, names='value')
        self.averageToggle.observe(self.update_legend_settings, names='value')
        self.legendModeToggle.observe(self._on_legend_mode_change, names='value')
        self.legendUpdate.on_click(self.update_legend_entry)

        self.settingsBtn.observe(self.handler_settingsDisplay, names='value')
        self.codeBtn.on_click(self._export_code_snippet)
        self.yLimitsBtn.on_click(self.handler_update_axes_limits)
        self.xLimitsBtn.on_click(self.handler_update_axes_limits)

        # 1D plot tool — tier 1 (rebuild needed for scale/label changes)
        for w in (self.xScaleMode, self.yScaleMode,
                  self.xLabel1D, self.yLabel1D,
                  self.xCustomFormula, self.yCustomFormula):
            w.observe(self._schedule_redraw, names='value')
        # Show/hide formula entry when Custom is toggled
        self.xScaleMode.observe(self._toggle_formula_visibility, names='value')
        self.yScaleMode.observe(self._toggle_formula_visibility, names='value')

        self.copyBtn.on_click(self.copy_figure)
        self.csvBtn.on_click(self.save_data)
        self.refreshBtn.on_click(self.handler_root_folder_update)
        self.refreshBtn.on_click(self.handler_folder_selection)

        self.referenceLocBtn.on_click(self.plotSpectrumLocations)
        self.plot2DUpdateBtn.on_click(self._redraw)
        self.plot2DClimMode.observe(self._update_clim_widget_state, names='value')
        for w in (self.plot2DYMin, self.plot2DYMax,
                  self.plot2DXMin, self.plot2DXMax,
                  self.plot2DVMin, self.plot2DVMax):
            w.observe(self._schedule_redraw, names='value')

        # Condensed-legend size / placement: re-grid + rebuild (debounced)
        self.figCdnLegendW.observe(self._schedule_redraw, names='value')
        self.figCdnLegendLoc.observe(self._schedule_redraw, names='value')

        # plasmonReference: single observer only — loads file then redraws
        self.plasmonReference.observe(self.handler_update_plasmonic_reference, names='value')
        self.cmapCategory.observe(self._on_cmap_category_change, names='value')

        self.rootFolder.observe(self.handler_root_folder_update, names='value')
        self.directorySelection.observe(self.handler_folder_selection, names=['value'])
        self.selectionList.observe(self.handler_file_selection, names=['value'])
        self.filterSelection.observe(self.handler_folder_selection, names='value')
        self.addFilterBtn.on_click(self.handler_update_filters)
        self.channelXSelect.observe(self.handler_channel_selection, names='value')
        self.channelYSelect.observe(self.handler_channel_selection, names=['value'])
        self.headerKeySelect.observe(self._on_header_key_select, names='value')

        self._connect_figure_settings_observers()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display(self) -> None:
        """Render the browser widget."""
        self._set_settings_visibility(True)
        display.clear_output(wait=True)
        display.display(self.h_main_layout)

    def _on_cmap_category_change(self, change) -> None:
        opts = _SEQ_CMAPS if change['new'] == 'Sequential' else _QUAL_CMAPS
        cur  = self.cmapSelection.value
        self.cmapSelection.options = opts
        self.cmapSelection.value   = cur if cur in opts else opts[0]

    def _redraw(self, *_) -> None:
        """Route to 2D or 1D rendering, then issue the single canvas draw.

        All data preparation happens inside update_axes/_render_2d without
        touching the screen; draw_idle here is the one render step (§7)."""
        self._set_busy(True, 'Rendering...')
        try:
            if self.plot2DToggle.value:
                self._render_2d()
            else:
                self.update_axes()
            self._export_dirty = True
            self.figure.tight_layout()        # title line count may have changed
            self.canvas.draw_idle()
        finally:
            self._set_busy(False)

    def _apply_figure_title(self) -> None:
        """Set the axes title (multi-line, top-left) from spec_label."""
        self.ax.set_title(self.spec_label, loc='left',
                          fontsize=self.figTitleSize.value,
                          color=self.figTitleColor.value)

    def _update_title(self, *_) -> None:
        """Patch the title in-place; no artist rebuild."""
        if self._loading:
            return
        self.update_scan_info()
        self._apply_figure_title()
        self._export_dirty = True
        self.canvas.draw_idle()

    def _update_display(self, *_) -> None:
        """Patch line colors, labels, and legend in-place; no rebuild.

        Called by display-only widgets (legendToggle, cmapSelection). 2D and
        Condensed modes involve a colorbar rebuild → route to full _redraw.
        """
        if self._loading:
            return
        if not self._lines or self.loaded_experiments is None:
            self._redraw()
            return
        if self.plot2DToggle.value or self.legendModeToggle.value == 'Condensed':
            self._redraw()
            return
        cmap   = self._resolve_cmap(self.cmapSelection.value)
        n      = len(self._lines)
        colors = cmap(np.asarray(self._color_positions(n)))
        labels = self._build_legend_labels(self.spec_x, self.spec_data)
        for i, line in enumerate(self._lines):
            line.set_color(colors[i])
            if i < len(labels):
                line.set_label(labels[i])
        self._refresh_legend(n)
        self._export_dirty = True
        self.canvas.draw_idle()

    def _refresh_legend(self, n_traces: int) -> None:
        """Create or remove the axes legend per legendToggle."""
        leg = self.ax.get_legend()
        if self.legendToggle.value and self._lines \
                and self.legendModeToggle.value != 'Condensed':
            fs = self.legendFontsize[bisect.bisect_left([4, 16], n_traces)]
            self.ax.legend(prop={'size': fs},
                           labelcolor=self.figLegendColor.value, frameon=False,fontsize=self.figLegendSize.value)
        elif leg is not None:
            leg.remove()

    def _cdn_active(self) -> bool:
        """The condensed filename legend applies: 1D view + Condensed mode."""
        return (self.legendModeToggle.value == 'Condensed'
                and not self.plot2DToggle.value)

    def _legend_cell_used(self) -> bool:
        """The legend cell is occupied: condensed legend (1D) or the value
        colorbar (2D view)."""
        return self._cdn_active() or self.plot2DToggle.value

    def _legend_horizontal(self) -> bool:
        return (getattr(self, 'figCdnLegendLoc', None) is not None
                and self.figCdnLegendLoc.value == 'Bottom')

    def _fig_in_total(self) -> tuple:
        """(w, h) inches: user size + legend size on the placement dimension,
        added only while the legend cell is occupied. Pure widget floats."""
        L = self.figCdnLegendW.value if self._legend_cell_used() else 0.0
        if self._legend_horizontal():
            return self.figWidth.value, self.figHeight.value + L
        return self.figWidth.value + L, self.figHeight.value

    def _ensure_grid(self) -> None:
        """Re-shape the 2-cell grid when placement or sizes change.

        set_subplotspec moves the SAME axes objects into the new grid, so
        object identity (artists, callbacks, self.ax refs) is preserved."""
        if getattr(self, 'figCdnLegendLoc', None) is None:
            return   # widgets not built yet (init-time provisional grid holds)
        L = max(self.figCdnLegendW.value, 0.1)   # gridspec ratio must be > 0
        state = (self.figCdnLegendLoc.value, round(self.figWidth.value, 3),
                 round(self.figHeight.value, 3), round(L, 3))
        if state == self._grid_state:
            return
        self._grid_state = state
        if self._legend_horizontal():
            gs = self.figure.add_gridspec(
                2, 1, height_ratios=[self.figHeight.value, L])
        else:
            gs = self.figure.add_gridspec(
                1, 2, width_ratios=[self.figWidth.value, L])
        self.ax.set_subplotspec(gs[0])
        self._cax.set_subplotspec(gs[1])

    def _sync_figure_size(self) -> None:
        """Grid shape + figure size from the widget inches, verbatim.
        set_size_inches is a no-op when unchanged — free on every render."""
        self._ensure_grid()
        w, h = self._fig_in_total()
        self.figure.set_size_inches(w, h, forward=True)

    def _apply_figure_layout(self, _=None) -> None:
        """'Apply Settings' callback: size + styles only. No margin argument —
        the constrained layout engine owns all margin/placement decisions, so
        Apply Settings must never touch axes geometry."""
        self._ensure_grid()
        w, h = self._fig_in_total()
        self._figure_layout_update(width=int(round(w * self._DPI)),
                                   height=int(round(h * self._DPI)))

    def _template_extra_save(self) -> dict:
        """Store the Condensed-legend size and placement with the template."""
        return {'cdn_legend_w': self.figCdnLegendW.value,
                'cdn_legend_loc': self.figCdnLegendLoc.value}

    def _template_extra_apply(self, entry: dict) -> None:
        """Restore the Condensed-legend width from a template (observers
        suppressed; _apply_selected_template re-applies the layout after)."""
        if 'cdn_legend_w' in entry:
            val = entry['cdn_legend_w']
            # Templates saved before the inch conversion stored px (> any
            # plausible inch count) — convert on load.
            if isinstance(val, (int, float)) and val > 10:
                val = round(val / self._DPI, 2)
            self._loading = True
            try:
                self.figCdnLegendW.value = val
            finally:
                self._loading = False
        if entry.get('cdn_legend_loc') in ('Right', 'Bottom'):
            self._loading = True
            try:
                self.figCdnLegendLoc.value = entry['cdn_legend_loc']
            finally:
                self._loading = False

    # ------------------------------------------------------------------
    # Axes positioning (fixed pixel margins; see BaseBrowser._ax_rect)
    # ------------------------------------------------------------------


    def _remove_dynamic(self) -> None:
        """Reset per-render artists that ax.clear() does not own.

        The legend cax is a fixed gridspec cell: CLEAR and hide it, never
        remove it (Colorbar.remove() would delete the cax from the figure).
        The secondary axis is genuinely removed."""
        self._cbar = None                    # Colorbar object dies with clear()
        try:
            self._cax.clear()
        except Exception:
            pass
        self._cax.set_visible(False)         # re-enabled by whoever draws in it
        if self._sec_ax is not None:
            try:
                self._sec_ax.remove()
            except Exception:
                pass
            self._sec_ax = None
        self._mesh = None

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _figure_stem(self, dir_name: str) -> str:
        return f'{dir_name}_{self.spec[0].name.split(".")[0]}_{self.channelYSelect.value[0]}'

    def save_data(self, a) -> None:
        """Save current spectra to browser_outputs/ as CSV."""
        self.csvBtn.icon = 'hourglass-start'
        try:
            out_dir = self.active_dir / 'browser_outputs'
            out_dir.mkdir(exist_ok=True)
            stem = (f'{self._current_dir_name()}'
                    f'_{self.spec[0].name.split(".")[0]}_{self.channelYSelect.value[0]}')
            if self.saveNote.value:
                stem += f'_{self.saveNote.value}'
            fname = out_dir / stem
            x_hdr = f'{self.spec_info[0]["x_label"]} ({self.spec_info[0]["x_unit"]})'
            header = ','.join(f'{x_hdr},{lbl.removesuffix(".dat")}' for lbl in self.labels)
            out = np.column_stack([arr for pair in zip(self.spec_x, self.spec_data) for arr in pair])
            np.savetxt(f'{fname}.csv', out, delimiter=',', header=header)
            self.updateErrorText('Saved CSV')
        except Exception as err:
            self.updateErrorText(f'CSV save error: {err}')
        finally:
            self.csvBtn.icon = 'list-ul'

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_new_image(self, filename=None):
        """Load the currently selected DAT file(s) and populate channel lists."""
        directory = self.directories[self.directorySelection.index]
        if directory != self.active_dir:
            directory = os.path.join(self.active_dir, directory)
        if filename is None:
            files = [os.path.join(directory, self.dat_files[idx])
                     for idx in self.spec_index]
            self.spec = self._load_spec_cached(files)   # F6: reuse cached Spm objects
            self.filenameText.value = ','.join(
                self.dat_files[idx] for idx in self.spec_index)
            self.loaded_experiments = [s.header['Experiment'] for s in self.spec]
            self._scan_cache = {}
            self._populate_header_inspector()
            self._loading = True
            self._update_channel_selection()   # channel observers suppressed
            self._loading = False
        else:
            return _make_spm(os.path.join(directory, filename))

    def _load_spec_cached(self, files: list) -> list:
        """Return Spm objects for *files*, reusing cache entries when mtime matches.

        §7 selection-diff: files kept across a selection change hit the cache
        and are never re-read; only misses touch the disk (in parallel, each
        construction under the global init lock, F6). The cache is pruned to
        the current selection so it stays bounded.
        """
        results: dict = {}
        misses: list = []
        for path in files:
            try:
                mt = os.path.getmtime(path)
            except OSError:
                mt = None
            cached = self._spm_cache.get(path)
            if cached is not None and mt is not None and cached[0] == mt:
                results[path] = cached[1]     # fresh cache hit — no re-read
            else:
                misses.append((path, mt))
        if misses:
            with ThreadPoolExecutor(max_workers=min(len(misses), 8)) as ex:
                loaded = list(ex.map(_make_spm, [p for p, _ in misses]))
            for (path, mt), obj in zip(misses, loaded):
                results[path] = obj
                if mt is not None:
                    self._spm_cache[path] = (mt, obj)
        for stale in [p for p in self._spm_cache if p not in files]:
            del self._spm_cache[stale]
        return [results[p] for p in files]

    def smooth_data(self, data):
        """Apply Savitzky-Golay filter using current widget params."""
        return smooth_data(data, self.svgSize.value, self.svgOrder.value)

    def _cache_scan_param(self, spec, key: str, default=('N/A', '')):
        """Return spec.get_param(key) with per-file caching (None → default)."""
        cache_key = (id(spec), key)
        if cache_key not in self._scan_cache:
            val = spec.get_param(key)
            self._scan_cache[cache_key] = val if val is not None else default
        return self._scan_cache[cache_key]

    def update_image_data(self) -> None:
        """Reload channel data, apply STML normalization if active."""
        channelX = self.channelXSelect.value
        channelY = self.channelYSelect.value
        self.spec_data, self.spec_info, self.labels, self.spec_x = [], [], [], []
        if len(self.selectionList.value) >= 1 and len(channelY) == 1:
            ch_y = channelY[0]
            use_index = (channelX == 'Index')

            def _read_one(spec):
                yd, yu = spec.get_channel(ch_y)
                xd, xu = (np.arange(len(yd)), 'N') if use_index else spec.get_channel(channelX)
                return yd, yu, xd, xu, spec.name

            with ThreadPoolExecutor(max_workers=min(len(self.spec), 8)) as ex:
                rows = list(ex.map(_read_one, self.spec))
            for yd, yu, xd, xu, name in rows:
                self.spec_data.append(yd)
                self.spec_info.append({'x_unit': xu, 'y_unit': yu, 'x_label': channelX})
                self.spec_x.append(xd)
                self.labels.append(name)
        if len(self.selectionList.value) == 1 and len(channelY) > 1:
            spec = self.spec[0]
            # resolve X once; for 'Index' derive length from first Y channel
            shared_x = shared_xunit = None
            for ch in channelY:
                spec_data, yunit = spec.get_channel(ch)
                if shared_x is None:
                    shared_x, shared_xunit = (
                        (np.arange(len(spec_data)), 'N') if channelX == 'Index'
                        else spec.get_channel(channelX))
                self.spec_data.append(spec_data)
                self.spec_info.append({'x_unit': shared_xunit, 'y_unit': yunit, 'x_label': channelX})
                self.spec_x.append(shared_x)
                self.labels.append(ch)
        # Notify on the case that produces a silent blank plot
        elif len(self.selectionList.value) > 1 and len(channelY) > 1:
            self.updateErrorText(
                'Select 1 file with multiple Y channels, OR multiple files with 1 Y channel.')
        try:
            self.update_scan_info()
        except Exception as err:
            # Bad/missing header keys must not prevent the plot from rendering
            self.updateErrorText(f'Header read error: {err}')

        # STML normalization
        if self.stmlToggle.value and 'stml' in (self.loaded_experiments or [''])[0].lower():
            data, xx = [], []
            for i, spec in enumerate(self.spec):
                t = float(spec.header['Exposure Time [ms]']) / 1000
                normfactor = 1.0
                if self.normalizeTimeBtn.value:
                    normfactor *= t
                if self.normalizeCurrentBtn.value:
                    normfactor *= abs(np.average(spec.get_channel('I')[0]))
                if self.normalizeEnergyBtn.value:
                    energies, intensities = rebin_intensity_nm_to_ev(
                        self.spec_x[i], self.spec_data[i])
                else:
                    energies    = 1240 / self.spec_x[i]
                    intensities = self.spec_data[i]
                if (self.plasmonReference.value != 'None'
                        and self.plasmonInfo['file'] is not None
                        and self.normalizePlasmonBtn.value):
                    plasmon = abs(self.plasmonInfo['interp'](energies)) + .1
                    data.append(intensities / normfactor / plasmon
                                * (plasmon > 0) * (self.spec_data[i] > 5))
                else:
                    data.append(intensities / normfactor)
                xx.append(energies)
            self.spec_data = data
            self.spec_x    = xx
            self.spec_info[0]['x_unit']  = 'eV'
            self.spec_info[0]['x_label'] = 'Energy'
            factor_list = ['cts']
            if self.normalizeCurrentBtn.value and self.normalizeTimeBtn.value:
                factor_list += ['pC']
            elif self.normalizeCurrentBtn.value:
                factor_list += ['pA']
            elif self.normalizeTimeBtn.value:
                factor_list += ['s']
            if self.normalizeEnergyBtn.value:
                factor_list += ['eV']
            self.spec_info[0]['y_unit'] = '/'.join(factor_list)

    # ------------------------------------------------------------------
    # Scan info helpers
    # ------------------------------------------------------------------

    def _bias_tuple(self, bias):
        """Convert bias to mV if < 100 mV."""
        b = list(bias)
        if abs(b[0]) < 0.1:
            b[0] *= 1000
            b[1] = 'mV'
        return tuple(b)

    def _info_stml(self, spec, label: list) -> tuple:
        fb_enable = spec.header['Z-Controller>Controller status']
        set_point = self._cache_scan_param(spec, 'setpoint_spec')
        bias      = self._bias_tuple(self._cache_scan_param(spec, 'V_spec'))
        feedback_str = 'feedback on' if fb_enable == 'ON' else 'feedback off'
        label.append(
            f'Exposure Time (s): {float(spec.header["Exposure Time [ms]"])/1000:.0f}, '
            f'λc: {spec.header["Center Wavelength [nm]"]}, '
            f'grating: {spec.header["Selected Grating"]}')
        setpoint_str = 'setpoint: I = %.0f%s, V = %.1f%s' % (set_point + bias)
        return setpoint_str, feedback_str

    def _info_bias_spec(self, spec, label: list) -> tuple:
        fb_enable = spec.get_param('Z-Ctrl hold')
        set_point = self._cache_scan_param(spec, 'setpoint_spec')
        bias      = self._bias_tuple(self._cache_scan_param(spec, 'V_spec'))
        lockin_a  = float(spec.header['Lock-in>Amplitude']) * 1e3
        lockin_ph = float(spec.header['Lock-in>Reference phase D1 (deg)'])
        lockin_f  = float(spec.header['Lock-in>Frequency (Hz)'])
        feedback_str = 'feedback on' if fb_enable == 'FALSE' else 'feedback off'
        label.append(f'lockin: A = {lockin_a:.0f} mV, θ = {lockin_ph:.1f} deg, f = {lockin_f:.0f} Hz')
        setpoint_str = 'setpoint: I = %.0f%s, V = %.1f%s' % (set_point + bias)
        return setpoint_str, feedback_str

    def _info_thz(self, spec, label: list) -> tuple:
        label.append(f'Laser Rep. Rate: {spec.header["Ext. VI 1>Laser>PP Frequency (MHz)"]}')
        label.append(f'Pulse Polarity: THz1;{spec.header["Ext. VI 1>THzPolarity>THz1"]}, '
                     f'THz2;{spec.header["Ext. VI 1>THzPolarity>THz2"]}')
        label.append(f'Delay Positions: THz1;{spec.header["Ext. VI 1>Position>PP1 (m)"]}, '
                     f'THz2;{spec.header["Ext. VI 1>Position>PP2(m)"]}')
        return '', ''

    def _info_z_spec(self, spec, label: list) -> tuple:
        set_point = self._cache_scan_param(spec, 'setpoint_spec')
        bias      = self._bias_tuple(self._cache_scan_param(spec, 'V_spec'))
        label.append(f'Spec Points: {len(self.spec_data)}')
        label.append(f'Integration time (s): {spec.header["Integration time (s)"]}')
        label.append(f'z-sweep (m): {spec.header["Z sweep distance (m)"]}')
        return 'setpoint: I = %.0f%s, V = %.1f%s' % (set_point + bias), ''

    def _info_history(self, spec, label: list) -> tuple:
        set_point = self._cache_scan_param(spec, 'setpoint_spec')
        bias      = self._bias_tuple(self._cache_scan_param(spec, 'V_spec'))
        label.append(f'Bias (V): {spec.header["Bias>Bias (V)"]}')
        label.append(f'Feedback: {spec.header["Z-Controller>Controller status"]}')
        label.append(f'Sample Period (ms): {spec.header["Sample Period (ms)"]}')
        return 'setpoint: I = %.0f%s, V = %.1f%s' % (set_point + bias), ''

    def update_scan_info(self) -> None:
        """Build spec_label from metadata; dispatches per experiment type."""
        if self.loaded_experiments is None:
            return
        experiments = self.loaded_experiments
        if experiments.count(experiments[0]) != len(experiments):
            self.updateErrorText('Please ensure all selections are the same measurement type')
            return

        spec       = self.spec[0]
        experiment = experiments[0]
        label: list = []
        setpoint_str = feedback_str = ''

        if self.nameToggle.value:
            if len(self.spec) > 1:
                label.append(f'Experiment: {experiment} → filename: {self.spec[0].name} → {self.spec[-1].name}')
            else:
                label.append(f'Experiment: {experiment} → filename: {spec.name}')

        for key, method in self._INFO_BUILDERS.items():
            if key in experiment:
                setpoint_str, feedback_str = getattr(self, method)(spec, label)
                break

        # date
        if len(self.spec) > 1:
            date_str = f'Date: {self.spec[0].header["Saved Date"]} → {self.spec[-1].header["Saved Date"]}'
        else:
            date_str = f'Date: {spec.header["Saved Date"]}'

        if self.setpointToggle.value and setpoint_str:
            if self.feedbackToggle.value and feedback_str:
                setpoint_str += f' → {feedback_str}'
            label.append(setpoint_str)
        if self.locationToggle.value:
            location = self.directories[self.directorySelection.index]
            if self.depthSelection.value != 'full':
                location = '\\'.join(str(location).split('\\')[-int(self.depthSelection.value):])
            label.append(f'location: {location}')
        if self.dateToggle.value:
            label.append(date_str)
        if self.svgToggle.value:
            label.append(f'Savitzky-Golay Filter → Window: {self.svgSize.value}, '
                         f'Order: {self.svgOrder.value}')

        self.spec_label = '\n'.join(label) if self.titleToggle.value else ''

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _apply_filters(self, x_values: list, y_values: list) -> tuple:
        """Apply all active filters to copies of y_values; return (x_out, y_out)."""
        y_out = [y.copy() for y in y_values]
        x_out = list(x_values)
        if self.averageToggle.value and len(self.spec_data) % self.groupSize.value == 0:
            x_out, y_out = group_average(self.spec_data, self.spec_x, self.groupSize.value)
        for i, y in enumerate(y_out):
            if self.fixZeroBtn.value:
                y = y - np.mean(self.spec_data[i][np.where(abs(x_out[i]) < 0.1)[0]])
            if self.thresholdToggle.value:
                y = y * (y < self.thresholdValue.value)
            if self.despikeBtn.value:
                y = despike_z_score(y, self.despikeWindow.value, self.despikeThreshold.value)
            if self.svgToggle.value:
                y = self.smooth_data(y)
            if self.movAvgBtn.value:
                y = moving_average(y, self.movAvgSize.value)
            if self.medFiltBtn.value:
                y = medfilt(y, self.medFiltSize.value)
            if self.flattenBtn.value:
                y = y / np.max(y)
            if self.offsetToggle.value:
                y = y + i * self.offsetSize.value
            y_out[i] = y
        return x_out, y_out

    def _build_legend_labels(self, x_values: list, y_out: list) -> list:
        """Return label list for the current legend mode."""
        mode = self.legendModeToggle.value
        if mode == 'Parameter':
            if self.parameterLegendList.value != 'Filename':
                labels = [spec.header[self.parameterLegendList.value] for spec in self.spec]
                param = self.parameterLegendList.value
                for i, val in enumerate(labels):
                    try:
                        v = float(val)
                    except (ValueError, TypeError):
                        # Non-numeric header value: show it verbatim, skip formatting.
                        labels[i] = str(val)
                        continue
                    if 'z' in param[0].lower():
                        labels[i] = f'{v * 1e9:.2f} nm'
                    elif 'current' in param.lower():
                        labels[i] = f'{v * 1e12:.2f} pA'
                    elif 'bias' in param.lower():
                        labels[i] = f'{v * 1e3:.2f} mV' if abs(v) < 0.1 else f'{v:.2f} V'
                    elif param == 'Exposure Time [ms]':
                        labels[i] = f'{v / 1000:.2f} s'
            else:
                labels = [spec.name for spec in self.spec]
            if self.averageToggle.value and len(self.spec_data) % self.groupSize.value == 0:
                labels = [labels[i] for i in range(0, len(labels), self.groupSize.value)]
        elif mode == 'Custom':
            labels = list(self.legendText.options)
        else:
            # Condensed: representative filename per group
            k = self.groupSize.value
            if (self.averageToggle.value
                    and len(self.labels) >= k
                    and len(self.labels) % k == 0):
                labels = [self.labels[i] for i in range(0, len(self.labels), k)]
            else:
                labels = self.labels
        return labels

    def _color_positions(self, n_visible: int) -> list:
        """Return colormap sample positions for n_visible traces.

        When averaging is active the positions reflect the centre of each group
        within the full un-averaged dataset, keeping trace colours consistent
        with the condensed colorbar labels at the extremes.
        """
        N = len(self.spec_data)
        k = self.groupSize.value
        averaged = (self.averageToggle.value
                    and N >= k and N % k == 0
                    and n_visible == N // k)
        if averaged and N > 1:
            return [((i * k + (k - 1) / 2) / (N - 1)) for i in range(n_visible)]
        return list(np.linspace(0, 1, max(n_visible, 2)))[:n_visible] \
            if n_visible == 1 else list(np.linspace(0, 1, max(n_visible, 2)))

    def _add_condensed_colorbar(self, colors, min_name: str, max_name: str) -> None:
        """Discrete colorbar whose bands match the trace colours exactly.

        A ListedColormap of the sampled trace colours + a ScalarMappable stands
        in for plotly's dummy-trace trick; ticks at the extremes carry the
        first/last filenames.
        """
        sm = ScalarMappable(norm=Normalize(0, 1), cmap=ListedColormap(list(colors)))
        sm.set_array([])
        # cax= into the persistent grid cell — no gridspec surgery, ever.
        horizontal = self._legend_horizontal()
        self._cax.set_visible(True)
        self._cbar = self.figure.colorbar(
            sm, cax=self._cax, ticks=[0, 1],
            orientation='horizontal' if horizontal else 'vertical')
        if horizontal:
            self._cax.set_xticklabels([min_name, max_name],
                                      fontsize=self.figLegendSize.value)
            labs = self._cax.get_xticklabels()
            if labs:               # tuck end labels under the bar ends
                labs[0].set_horizontalalignment('left')
                labs[-1].set_horizontalalignment('right')
        else:
            self._cax.set_yticklabels([min_name, max_name],
                                      fontsize=self.figLegendSize.value)

    def _add_stml_axis(self) -> None:
        """Wavelength (nm) axis on top of the energy (eV) axis.

        secondary_xaxis takes the eV↔nm conversion (self-inverse: 1240/x) and
        keeps itself in sync on every zoom/pan — no manual tick bookkeeping."""
        xmin, xmax = self.ax.get_xlim()
        if xmin <= 0 or xmax <= 0:
            return
        conv = lambda v: 1240.0 / np.clip(np.asarray(v, dtype=float), 1e-9, None)
        self._sec_ax = self.ax.secondary_xaxis('top', functions=(conv, conv))
        self._sec_ax.set_xlabel('Wavelength (nm)',
                                fontsize=self.figAxesLabelSize.value,
                                color=self.figAxesLabelColor.value)
        self._sec_ax.tick_params(
            direction='in' if self.figTicksMode.value == 'inside' else 'out',
            labelsize=self.figTickSize.value, colors=self.figTickColor.value)

    def update_axes(self) -> None:
        """Re-render all spectra with current filter/display settings.

        All computation runs first; artists are then rebuilt on the persistent
        axes. The canvas draw is issued once, by _redraw (§7 single render).
        """
        # ── computation (no figure mutations) ─────────────────────────────
        x_values, y_values = self._apply_filters(self.spec_x, self.spec_data)

        x_mode = self.xScaleMode.value
        y_mode = self.yScaleMode.value
        if x_mode == 'Custom' and self.xCustomFormula.value.strip():
            try:
                x_values = self._apply_axis_scale(x_values, self.xCustomFormula.value, 'x')
            except Exception as err:
                self.updateErrorText(f'X formula error: {err}')
        if y_mode == 'Custom' and self.yCustomFormula.value.strip():
            try:
                y_values = self._apply_axis_scale(y_values, self.yCustomFormula.value, 'y')
            except Exception as err:
                self.updateErrorText(f'Y formula error: {err}')

        labels    = self._build_legend_labels(x_values, y_values)
        n         = len(y_values)
        positions = np.asarray(self._color_positions(n)) if n else np.array([])
        colors    = self._resolve_cmap(self.cmapSelection.value)(positions)
        condensed = self.legendModeToggle.value == 'Condensed'

        # ── artist rebuild on the persistent axes ─────────────────────────
        self._sync_figure_size()      # ± Condensed-legend width reservation
        self._remove_dynamic()
        self.ax.clear()
        self._lines = []
        for x, y, lbl, c in zip(x_values, y_values, labels, colors):
            line, = self.ax.plot(x, y, color=c,
                                 label='_nolegend_' if condensed else lbl)
            self._lines.append(line)

        auto_x  = f'{self.spec_info[0]["x_label"]} ({self.spec_info[0]["x_unit"]})'
        auto_y  = f'{self.channelYSelect.value[0]} ({self.spec_info[0]["y_unit"]})'
        self.ax.set_xlabel(self.xLabel1D.value.strip() or auto_x)
        self.ax.set_ylabel(self.yLabel1D.value.strip() or auto_y)
        self.ax.set_xscale('log' if x_mode == 'Log' else 'linear')
        self.ax.set_yscale('log' if y_mode == 'Log' else 'linear')
        self.ax.minorticks_on()

        if self.xLimitLock.value:
            self.ax.set_xlim(self.xLimitsMin.value, self.xLimitsMax.value)
        if self.yLimitLock.value:
            self.ax.set_ylim(self.yLimitsMin.value, self.yLimitsMax.value)

        self._refresh_legend(n)
        if condensed and n:
            min_name, max_name = self._condensed_range_labels()
            self._add_condensed_colorbar(colors, min_name, max_name)

        self._apply_figure_title()
        self._style_axes(self.ax)
        self._sync_axis_limit_sliders()
        if self.stmlToggle.value and 'stml' in (self.loaded_experiments or [''])[0].lower():
            self._add_stml_axis()

    # ------------------------------------------------------------------
    # Alternate views
    # ------------------------------------------------------------------

    def plotSpectrumLocations(self, a) -> None:
        """Overlay tip positions as markers on the linked SXM browser axes."""
        if self.sxmBrowser is None or self.sxmBrowser.img is None:
            return
        sxm = self.sxmBrowser
        # Clear previous overlay artists (SXM tracks them in _overlay_artists)
        for art in getattr(sxm, '_overlay_artists', []):
            try:
                art.remove()
            except Exception:
                pass
        sxm._overlay_artists = []
        n_traces = max(len(self._lines), 2)
        colors = self._resolve_cmap(self.cmapSelection.value)(
            np.linspace(0, 1, n_traces))
        k = 0
        for i, spec in enumerate(self.spec):
            if self.averageToggle.value and len(self.spec_data) % self.groupSize.value == 0:
                if i not in range(0, len(self.spec), self.groupSize.value):
                    continue
            rx, ry = relative_position(sxm.img, spec)
            c   = colors[k % len(colors)]
            sym = self.markerSelection.value
            if sym != 'N':
                marker, = sxm.ax.plot([rx], [ry], linestyle='none',
                                      marker=_MARKER_MAP.get(sym, 'o'),
                                      markersize=10, color=c)
                sxm._overlay_artists.append(marker)
            else:
                start = max(self.spec_index) - min(self.spec_index) + 1
                txt = sxm.ax.annotate(f'{start - i}', (rx, ry),
                                      color='red', fontsize=10)
                sxm._overlay_artists.append(txt)
            k += 1
        sxm._export_dirty = True
        sxm.canvas.draw_idle()

    def _compute_clim(self, z, x_ref, y_vals) -> tuple:
        """Return (vmin, vmax) for the 2D colour scale based on current mode."""
        mode = self.plot2DClimMode.value
        if mode == 'Custom':
            vmin, vmax = self.plot2DVMin.value, self.plot2DVMax.value
            return (vmin, vmax) if vmin != vmax else (np.nanmin(z), np.nanmax(z))
        if mode == 'Full':
            return np.nanmin(z), np.nanmax(z)
        # Visible: restrict to data within the current axis range inputs
        xlo, xhi = self.plot2DXMin.value, self.plot2DXMax.value
        ylo, yhi = self.plot2DYMin.value, self.plot2DYMax.value
        xm = ((x_ref >= xlo) & (x_ref <= xhi)) if xlo != xhi else np.ones(len(x_ref), bool)
        ym = ((y_vals >= ylo) & (y_vals <= yhi)) if ylo != yhi else np.ones(len(y_vals), bool)
        z_vis = z[np.ix_(ym, xm)]
        return (np.nanmin(z_vis), np.nanmax(z_vis)) if z_vis.size else (np.nanmin(z), np.nanmax(z))

    def _update_clim_widget_state(self, *_) -> None:
        """Enable custom clim fields only when Custom mode is active."""
        is_custom = self.plot2DClimMode.value == 'Custom'
        self.plot2DVMin.disabled = not is_custom
        self.plot2DVMax.disabled = not is_custom

    def _render_2d(self) -> None:
        """Render filtered spectra as a 2D pcolormesh with colorbar."""
        if not self.spec or self.loaded_experiments is None:
            return

        x_values, y_values = self._apply_filters(self.spec_x, self.spec_data)
        n_spec = len(y_values)

        # Unified x grid: use the longest x array as reference
        n_x   = max(len(x) for x in x_values)
        x_ref = x_values[[len(x) for x in x_values].index(n_x)]

        # Stack spectra into (n_spec, n_x), interpolating shorter arrays
        z = np.zeros((n_spec, n_x))
        for i, (xi, yi) in enumerate(zip(x_values, y_values)):
            z[i] = yi if len(yi) == n_x else np.interp(x_ref, xi, yi)

        # Y axis values from header parameter or sequential index
        y_param  = self.plot2DYParam.value
        averaged = (self.averageToggle.value
                    and len(self.spec_data) % self.groupSize.value == 0)
        step = self.groupSize.value if averaged else 1
        if y_param == 'Index':
            y_vals  = np.arange(n_spec, dtype=float)
            y_label = 'Spectrum Index'
        elif y_param == 'Position (nm)':
            if self.sxmBrowser is not None and self.sxmBrowser.img is not None:
                pos = []
                for sp in self.spec[::step]:
                    try:
                        rx, ry = relative_position(self.sxmBrowser.img, sp)
                        pos.append((rx, ry))
                    except Exception:
                        pos.append((np.nan, np.nan))
                pos = np.array(pos[:n_spec])
                x0, y0 = pos[0]
                y_vals = np.sqrt((pos[:, 0] - x0) ** 2 + (pos[:, 1] - y0) ** 2)
            else:
                y_vals = np.arange(n_spec, dtype=float)
            y_label = 'Position (nm)'
        else:
            raw = []
            for sp in self.spec[::step]:
                try:
                    raw.append(float(sp.header[y_param]))
                except (KeyError, ValueError, TypeError):
                    raw.append(np.nan)
            y_arr = np.array(raw[:n_spec])
            if 'Z' in y_param:
                y_vals, y_label = y_arr * 1e9, 'Z (nm)'
            elif 'Current' in y_param:
                y_vals, y_label = y_arr * 1e12, 'Current (pA)'
            elif 'Exposure' in y_param:
                y_vals, y_label = y_arr / 1000, 'Exposure Time (s)'
            else:
                y_vals, y_label = y_arr, y_param

        vmin, vmax = self._compute_clim(z, x_ref, y_vals)
        y_ch   = f'{self.channelYSelect.value[0]} ({self.spec_info[0]["y_unit"]})'
        x_auto = f'{self.spec_info[0]["x_label"]} ({self.spec_info[0]["x_unit"]})'

        # ── artist rebuild ────────────────────────────────────────────────
        self._sync_figure_size()      # drops the Condensed reservation in 2D
        self._remove_dynamic()
        self.ax.clear()
        self._lines = []
        self._mesh = self.ax.pcolormesh(
            x_ref, y_vals, z,
            cmap=self._resolve_cmap(self.cmapSelection.value),
            vmin=vmin, vmax=vmax, shading='nearest')
        self._cax.set_visible(True)
        self._cbar = self.figure.colorbar(
            self._mesh, cax=self._cax,
            orientation='horizontal' if self._legend_horizontal() else 'vertical')
        self._cbar.set_label(y_ch, fontsize=self.figAxesLabelSize.value)
        self._cbar.ax.tick_params(labelsize=self.figTickSize.value)

        xlo, xhi = self.plot2DXMin.value, self.plot2DXMax.value
        ylo, yhi = self.plot2DYMin.value, self.plot2DYMax.value
        self.ax.set_xlabel(self.plot2DXLabel.value or x_auto)
        self.ax.set_ylabel(self.plot2DYLabel.value or y_label)
        if xlo != xhi:
            self.ax.set_xlim(xlo, xhi)
        if ylo != yhi:
            self.ax.set_ylim(ylo, yhi)
        self.ax.minorticks_on()
        self._apply_figure_title()
        self._style_axes(self.ax)

    # ------------------------------------------------------------------
    # Navigation / selection state
    # ------------------------------------------------------------------

    def _update_channel_selection(self) -> None:
        if not self.spec:
            return
        exp = self.loaded_experiments[0]
        default_channels = self.default_channel.get(exp, [None, None])
        cur_x = self.channelXSelect.value
        cur_y = self.channelYSelect.value[0] if self.channelYSelect.value else None
        self.channelXSelect.options = ['Index'] + self.spec[0].channels
        self.channelYSelect.options = self.spec[0].channels
        if self.current_experiment == exp and cur_x in self.spec[0].channels:
            self.channelXSelect.value = cur_x
        elif default_channels[0] is not None:
            self.channelXSelect.value = default_channels[0]
        else:
            self.channelXSelect.value = 'Index'
        if self.current_experiment == exp and cur_y in self.spec[0].channels:
            self.channelYSelect.value = [cur_y]
        elif default_channels[1] is not None:
            self.channelYSelect.value = [default_channels[1]]
        else:
            self.channelYSelect.value = [self.spec[0].channels[0]]
        self.current_experiment = exp

    def _populate_header_inspector(self) -> None:
        """Fill the header key Select from the first loaded spectrum."""
        if not self.spec:
            return
        keys = list(self.spec[0].header.keys())
        self.headerKeySelect.options = keys
        if keys:
            self.headerKeySelect.value = keys[0]
            self.headerValueText.value = str(self.spec[0].header[keys[0]])

    def _on_header_key_select(self, change) -> None:
        """Show the selected header key's value in the read-only Textarea."""
        if not self.spec or change['new'] is None:
            return
        self.headerValueText.value = str(self.spec[0].header.get(change['new'], ''))

    def _sync_axis_limit_sliders(self) -> None:
        """Update limit widgets from axes ranges (autoscaled when unlocked)."""
        if not self.xLimitLock.value or not self.yLimitLock.value:
            self.ax.relim()
            self.ax.autoscale_view(scalex=not self.xLimitLock.value,
                                   scaley=not self.yLimitLock.value)
        if not self.xLimitLock.value:
            self.xLimitsMin.value, self.xLimitsMax.value = self.ax.get_xlim()
        if not self.yLimitLock.value:
            self.yLimitsMin.value, self.yLimitsMax.value = self.ax.get_ylim()

    def _update_legend_on_load(self) -> None:
        """Sync legend parameter list and custom-text entries after a file load.

        Runs under _loading so observers do not fire on stale traces before _redraw().
        Auto-activates Condensed mode when ≥10 traces are plotted and filenames follow a numbering pattern.
        """
        if not self.spec:
            return
        self._loading = True
        self.parameterLegendList.options = ['Filename'] + list(self.spec[0].header.keys())
        self.parameterLegendList.value   = 'Filename'
        new_selection = list(self.selectionList.value)
        if self.averageToggle.value and len(self.spec_data) % self.groupSize.value == 0:
            new_selection = [new_selection[i]
                             for i in range(0, len(new_selection), self.groupSize.value)]
        self.legendText.options = new_selection
        if new_selection:
            self.legendText.value = new_selection[0]
        # Auto-manage Condensed mode based on numbering pattern detection
        has_pattern = len(self.spec_data) >= 10 and self._detect_numbering_pattern() is not None
        if has_pattern and self.legendModeToggle.value != 'Condensed':
            self._auto_condensed = True
            self.legendModeToggle.value = 'Condensed'       # fires _on_legend_mode_change
        elif not has_pattern and self._auto_condensed:
            self._auto_condensed = False
            self.legendModeToggle.value = 'Parameter'       # revert auto-set
        self._loading = False

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _build_export_code(self) -> str:
        """Build a self-contained Python snippet reproducing the current figure."""
        files  = list(self.selectionList.value) or self.dat_files
        d      = str(self.directories[self.directorySelection.index]).replace('\\', '/')
        x_ch   = self.channelXSelect.value
        y_ch   = self.channelYSelect.value[0] if self.channelYSelect.value else ''
        exp    = (self.loaded_experiments or [''])[0].lower()
        x_mode = self.xScaleMode.value
        y_mode = self.yScaleMode.value

        # STML flags — all gated on toggle *and* experiment type
        is_stml     = self.stmlToggle.value and 'stml' in exp
        use_time    = is_stml and self.normalizeTimeBtn.value
        use_current = is_stml and self.normalizeCurrentBtn.value
        use_rebin   = is_stml and self.normalizeEnergyBtn.value
        use_plasmon = (is_stml and self.normalizePlasmonBtn.value
                       and self.plasmonInfo['file'] is not None)

        # ── imports ──────────────────────────────────────────────────────────
        scipy_sig = 'medfilt, savgol_filter' if use_plasmon else 'medfilt'
        lines = [
            "import numpy as np",
            "from pathlib import Path",
            f"from scipy.signal import {scipy_sig}",
            "import matplotlib.pyplot as plt",
            "from spmpy import Spm",
            "from simpleNFB.process_utils import (",
            "    rebin_intensity_nm_to_ev, smooth_data, group_average,",
            "    despike_z_score, moving_average,",
            ")",
        ]
        if use_plasmon:
            lines.append("from scipy.interpolate import interp1d")
        lines.append("")

        # ── file list ────────────────────────────────────────────────────────
        lines += [f"data_dir = Path(r'{d}')"]
        lines += ["files = ["]
        for f in files:
            lines += [f"    '{f}',"]
        lines += ["]", ""]

        # ── load and extract ─────────────────────────────────────────────────
        lines += ["spec = [Spm(str(data_dir / f)) for f in files]"]
        if x_ch == 'Index':
            lines += [
                f"y_ch = '{y_ch}'",
                "spec_data = [s.get_channel(y_ch)[0] for s in spec]",
                "spec_x    = [np.arange(len(y)) for y in spec_data]",
            ]
        else:
            lines += [
                f"x_ch = '{x_ch}'",
                f"y_ch = '{y_ch}'",
                "spec_x    = [s.get_channel(x_ch)[0] for s in spec]",
                "spec_data = [s.get_channel(y_ch)[0] for s in spec]",
            ]
        lines += [""]

        # ── STML normalization ───────────────────────────────────────────────
        if is_stml:
            if use_plasmon:
                pfile = self.plasmonInfo['file']
                lines += [
                    f"plasmon_ref = Spm(str(data_dir / '{pfile}'))",
                    "ref_y, _ = plasmon_ref.get_channel('Intensity')",
                    "ref_x, _ = plasmon_ref.get_channel('Wavelength')",
                    "ref_y = abs(savgol_filter(ref_y, 21, 1))",
                    "ref_y *= (ref_y > 35)",
                    "ref_y /= np.max(ref_y)",
                    "plasmon_interp = interp1d(1240 / ref_x, ref_y,"
                    " bounds_error=False, fill_value=0.0)",
                    "",
                ]
            lines += ["xx, data = [], []",
                      "for i, (x, y, s) in enumerate(zip(spec_x, spec_data, spec)):"]
            lines += ["    normfactor = 1.0"]
            if use_time:
                lines += ["    normfactor *= float(s.header['Exposure Time [ms]']) / 1000"]
            if use_current:
                lines += ["    normfactor *= abs(np.average(s.get_channel('I')[0]))"]
            if use_rebin:
                lines += ["    energies, intensities = rebin_intensity_nm_to_ev(x, y)"]
            else:
                lines += ["    energies = 1240 / x", "    intensities = y"]
            if use_plasmon:
                lines += [
                    "    plasmon = abs(plasmon_interp(energies)) + 0.1",
                    "    data.append(intensities / normfactor / plasmon"
                    " * (plasmon > 0) * (y > 5))",
                ]
            else:
                lines += ["    data.append(intensities / normfactor)"]
            lines += ["    xx.append(energies)",
                      "spec_x, spec_data = xx, data", ""]

        # ── group average (before per-trace loop) ────────────────────────────
        if (self.averageToggle.value
                and len(self.spec_data) >= self.groupSize.value
                and len(self.spec_data) % self.groupSize.value == 0):
            k = self.groupSize.value
            lines += [
                f"spec_x, spec_data = group_average(spec_data, spec_x, {k})",
                "",
            ]

        # ── per-trace filters ────────────────────────────────────────────────
        filter_body = []
        if self.fixZeroBtn.value:
            filter_body += ["    y = y - np.mean(y[np.where(abs(x) < 0.1)[0]])"]
        if self.thresholdToggle.value:
            filter_body += [f"    y = y * (y < {self.thresholdValue.value})"]
        if self.despikeBtn.value:
            filter_body += [
                f"    y = despike_z_score(y, {self.despikeWindow.value},"
                f" {self.despikeThreshold.value})"
            ]
        if self.svgToggle.value:
            filter_body += [
                f"    y = smooth_data(y, {self.svgSize.value}, {self.svgOrder.value})"
            ]
        if self.movAvgBtn.value:
            filter_body += [f"    y = moving_average(y, {self.movAvgSize.value})"]
        if self.medFiltBtn.value:
            filter_body += [f"    y = medfilt(y, {self.medFiltSize.value})"]
        if self.flattenBtn.value:
            filter_body += ["    y = y / np.max(y)"]
        if self.offsetToggle.value:
            filter_body += [f"    y = y + i * {self.offsetSize.value}"]

        if filter_body:
            lines += ["for i, (x, y) in enumerate(zip(spec_x, spec_data)):"]
            lines += filter_body
            lines += ["    spec_data[i] = y", ""]

        # ── axis scale transforms (Custom only; Log handled by ax.set_*scale) ─
        if x_mode == 'Custom' and self.xCustomFormula.value.strip():
            expr = self.xCustomFormula.value.strip()
            lines += [f"spec_x = [eval('{expr}') for x in spec_x]", ""]
        if y_mode == 'Custom' and self.yCustomFormula.value.strip():
            expr = self.yCustomFormula.value.strip()
            lines += [f"spec_data = [eval('{expr}') for y in spec_data]", ""]

        # ── plot ─────────────────────────────────────────────────────────────
        x_info  = self.spec_info[0] if self.spec_info else {}
        auto_x  = f"{x_info.get('x_label', 'x')} ({x_info.get('x_unit', '')})"
        auto_y  = f"{y_ch} ({x_info.get('y_unit', '')})"
        x_label = self.xLabel1D.value.strip() or auto_x
        y_label = self.yLabel1D.value.strip() or auto_y
        avg_active = (self.averageToggle.value
                      and len(self.spec_data) >= self.groupSize.value
                      and len(self.spec_data) % self.groupSize.value == 0)
        if avg_active:
            k = self.groupSize.value
            lines += [f"labels = files[::{k}]", ""]
        else:
            lines += ["labels = files"]
        lines += [
            "fig, ax = plt.subplots()",
            "for x, y, lbl in zip(spec_x, spec_data, labels):",
            "    ax.plot(x, y, label=lbl)",
            f"ax.set_xlabel('{x_label}')",
            f"ax.set_ylabel('{y_label}')",
        ]
        if x_mode == 'Log':
            lines += ["ax.set_xscale('log')"]
        if y_mode == 'Log':
            lines += ["ax.set_yscale('log')"]
        lines += ["ax.legend()", "plt.tight_layout()", "plt.show()"]

        return '\n'.join(lines)

    def _export_code_snippet(self, a=None) -> None:
        """Export reproducible code: auto-copy to clipboard and show panel in browser."""
        code = self._build_export_code()

        # Auto-copy immediately; record outcome for the status label
        try:
            import win32clipboard
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(code, win32clipboard.CF_UNICODETEXT)
            win32clipboard.CloseClipboard()
            status_html = '<span style="font-size:11px;color:#2a9d2a;"><b>Copied to clipboard.</b> Paste into a new cell to run.</span>'
        except ImportError:
            status_html = ('<span style="font-size:11px;color:#888;">'
                           'pywin32 not installed &mdash; select all &amp; copy (Ctrl+A, Ctrl+C) below.</span>')
        except Exception as e:
            status_html = f'<span style="font-size:11px;color:#c0392b;">Clipboard error: {e}</span>'

        status = widgets.HTML(value=status_html)
        close_btn = widgets.Button(description='Close', layout=widgets.Layout(width='60px', height='26px'))
        code_area = widgets.Textarea(
            value=code,
            layout=widgets.Layout(width='100%', height='220px', font_family='monospace',
                                  font_size='11px'),
        )

        def _close(_):
            self._code_out.clear_output()
            self._code_out.layout.display = 'none'
        close_btn.on_click(_close)

        panel = widgets.VBox(
            children=[
                widgets.HBox(
                    children=[status, close_btn],
                    layout=widgets.Layout(justify_content='space-between',
                                         align_items='center', margin='0 0 4px 0'),
                ),
                code_area,
            ],
            layout=widgets.Layout(width='100%', padding='6px',
                                  border='1px solid #ccc', border_radius='4px'),
        )

        self._code_out.layout.display = ''
        self._code_out.clear_output(wait=True)
        with self._code_out:
            display.display(panel)

        # Also attempt cell injection for JupyterLab / Notebook 7 users
        self._inject_cell(code)

    def handler_settingsDisplay(self, a) -> None:
        self._set_settings_visibility(self.settingsBtn.value)

    def _detect_numbering_pattern(self) -> tuple | None:
        """Return (min_name, max_name) if every selected filename contains an embedded number."""
        names = list(self.selectionList.value)
        if not names:
            return None
        numbered = []
        for name in names:
            m = re.search(r'\d+', name)
            if m:
                numbered.append((int(m.group()), name))
        if len(numbered) < len(names):
            return None   # some filenames had no number
        numbered.sort()
        return numbered[0][1], numbered[-1][1]

    def _condensed_range_labels(self) -> tuple:
        """Return (first_name, last_name) for the condensed colorbar tick labels.

        Labels come from the display-order selection so they match the trace
        at colorbar position 0 (first trace) and 1 (last trace). When averaging
        is active the representative file (first of each group) is used."""
        names = list(self.selectionList.value)
        if not names:
            return ('', '')
        k = self.groupSize.value
        if (self.averageToggle.value
                and len(names) >= k
                and len(names) % k == 0):
            names = [names[i] for i in range(0, len(names), k)]
        return (names[0], names[-1]) if len(names) >= 2 else (names[0], names[0])

    def _apply_legend_mode_visibility(self, mode: str) -> None:
        """Show/hide mode-specific legend widgets without triggering a redraw."""
        param_vis  = 'flex' if mode == 'Parameter'  else 'none'
        custom_vis = 'flex' if mode == 'Custom'      else 'none'
        self.parameterLegendList.layout.display = param_vis
        for w in (self.legendText, self.legendEntry, self.legendUpdate):
            w.layout.display = custom_vis

    def _on_legend_mode_change(self, change) -> None:
        """Observer for legendModeToggle: update section visibility and redraw."""
        if not self._loading:
            self._auto_condensed = False    # manual change clears auto-tracking
        self._apply_legend_mode_visibility(change['new'])
        if not self._loading:
            self._redraw()

    def update_legend_entry(self, a) -> None:
        """Apply legendEntry text to the selected legendText item then redraw."""
        entry   = self.legendText.value
        options = list(self.legendText.options)
        if entry in options:
            options[options.index(entry)] = self.legendEntry.value
            self.legendText.options = options
        self._redraw()

    def update_legend_settings(self, a) -> None:
        """Re-slice legendText entries when averaging group size changes."""
        new_selection = list(self.selectionList.value)
        if self.averageToggle.value and len(self.spec_data) % self.groupSize.value == 0:
            new_selection = [new_selection[i]
                             for i in range(0, len(new_selection), self.groupSize.value)]
        self.legendText.options = new_selection
        if new_selection:
            self.legendText.value = new_selection[0]

    def handler_folder_selection(self, a) -> None:
        self._set_busy(True, 'Scanning folder...')
        try:
            index = 0
            directory = self.directories[self.directorySelection.index]
            filter_vals = self.filterSelection.value
            self.sxm_files = []
            dat_entries = []
            with os.scandir(directory) as it:
                for e in it:
                    if e.name.endswith('.sxm'):
                        self.sxm_files.append(e.name)
                    elif e.name.endswith('.dat'):
                        if 'all' in filter_vals or any(f in e.name for f in filter_vals):
                            dat_entries.append(e)
            dat_entries.sort(key=lambda e: e.stat().st_mtime, reverse=True)
            self.dat_files = [e.name for e in dat_entries]
            self.all_files = self.sxm_files + self.dat_files
            # suppress handler_file_selection while syncing list widget state
            self._loading = True
            self.selectionList.options = self.dat_files
            if self.dat_files:
                first = self.dat_files[index]
                self.filenameText.value = first
                self.selectionList.value = [first]
                self.plasmonReference.options = (
                    ['None'] + [f for f in self.dat_files if 'stml' in f.lower()])
                self.plasmonReference.value = 'None'
            self._loading = False
            if self.dat_files:
                self.spec_index = [0]
                self._current_selection = tuple(self.selectionList.value)
                try:
                    self.load_new_image()
                    self.update_image_data()
                    self._redraw()
                except Exception as err:
                    self.updateErrorText('folder selection error: ' + str(err))
        finally:
            self._set_busy(False)

    def handler_file_selection(self, update) -> None:
        if self._loading:
            return
        # §7: a re-fired event with an unchanged selection is a no-op —
        # no file read, no redraw.
        new_sel = tuple(self.selectionList.value)
        if new_sel == self._current_selection:
            return
        self._set_busy(True, 'Loading files...')
        try:
            self.spec_index = [self.dat_files.index(v) for v in new_sel]
            self._current_selection = new_sel
            if new_sel:
                # Prepare everything (cached Spm reuse, channel data, legend
                # state) before the single _redraw at the end (§7).
                self.load_new_image()
                self.update_image_data()
                self._update_legend_on_load()
                self._redraw()
        except Exception as err:
            self.updateErrorText('file selection error: ' + str(err))
            print(traceback.format_exc())
        finally:
            self._set_busy(False)

    def handler_channel_selection(self, update) -> None:
        if self._loading:
            return
        try:
            if self.spec:
                self.update_image_data()
                self._redraw()
        except Exception as err:
            self.updateErrorText('channel selection error: ' + str(err))

    def handler_update_filters(self, update) -> None:
        val = self.newFilterText.value
        if val and val not in self.filterSelection.options:
            self.filterSelection.options = list(self.filterSelection.options) + [val]
            self.newFilterText.value = ''

    def handler_update_plasmonic_reference(self, update) -> None:
        if self._loading or self.plasmonReference.value == 'None':
            return
        directory = self.directories[self.directorySelection.index]
        if directory != self.active_dir:
            directory = os.path.join(self.active_dir, directory)
        if self.plasmonReference.value in self.dat_files:
            ref = _make_spm(os.path.join(directory, self.plasmonReference.value))
            ref_y, _ = ref.get_channel('Intensity')
            ref_x, _ = ref.get_channel('Wavelength')
            ref_y = abs(savgol_filter(ref_y, 21, 1))
            ref_y *= (ref_y > 35)
            ref_y /= np.max(ref_y)
            self.plasmonInfo['spm']    = ref
            self.plasmonInfo['file']   = self.plasmonReference.value
            self.plasmonInfo['interp'] = interp1d(1240 / ref_x, ref_y,
                                                   bounds_error=False, fill_value=0.0)
            self._redraw()   # reference is now loaded; redraw uses fresh interpolation

    def _apply_axis_scale(self, data_list: list, formula: str, var: str) -> list:
        """Eval formula on each array in data_list. var is 'x' or 'y'."""
        ns = {'np': np}
        result = []
        for arr in data_list:
            ns[var] = arr
            # trusted input only — formula comes from the user's own
            # xCustomFormula/yCustomFormula widget in a local lab session.
            result.append(np.asarray(eval(formula, ns)))  # noqa: S307
        return result

    def _toggle_formula_visibility(self, change) -> None:
        """Show/hide custom formula Text when scale mode changes."""
        is_custom = change['new'] == 'Custom'
        display_val = 'flex' if is_custom else 'none'
        if change['owner'] is self.xScaleMode:
            self.xCustomFormula.layout.display = display_val
        elif change['owner'] is self.yScaleMode:
            self.yCustomFormula.layout.display = display_val

    def handler_update_axes_limits(self, a) -> None:
        if a == self.xLimitsBtn:
            self.ax.set_xlim(self.xLimitsMin.value, self.xLimitsMax.value)
        elif a == self.yLimitsBtn:
            self.ax.set_ylim(self.yLimitsMin.value, self.yLimitsMax.value)
        # the secondary (wavelength) axis follows automatically — no manual sync
        self._export_dirty = True
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _get_spec_selection(self, filter: bool = False) -> list:
        """Return Spm objects matching get_file_selection(filter=filter)."""
        if filter and self.averageToggle.value:
            k = self.groupSize.value
            if len(self.spec) >= k and len(self.spec) % k == 0:
                return [self.spec[i] for i in range(0, len(self.spec), k)]
        return list(self.spec)

    def get_file_selection(self, filter: bool = False) -> list:
        """Return the currently selected filenames.

        Parameters
        ----------
        filter : bool
            When True and the averaging filter is active, return one filename
            per averaged group (the first file of each group), matching the
            number of traces in the figure.
        """
        files = list(self.selectionList.value)
        if filter and self.averageToggle.value:
            k = self.groupSize.value
            if len(files) >= k and len(files) % k == 0:
                files = [files[i] for i in range(0, len(files), k)]
        return files

    def get_parameter(self, key: str, filter: bool = False) -> list:
        """Return get_param(key) for each spectrum in the current file selection."""
        return [self._cache_scan_param(s, key) for s in self._get_spec_selection(filter=filter)]

    def get_channel(self, channel: str, filter: bool = False) -> list:
        """Return raw channel data arrays for each spectrum in the current selection.

        Returns
        -------
        list of ndarray -- one entry per spectrum (or group when filter=True).
        """
        return [s.get_channel(channel)[0] for s in self._get_spec_selection(filter=filter)]

    def get_plot_data(self) -> list:
        """Return the processed (x, y) data currently shown in the figure.

        Reads from the live Line2D artists (self._lines), so all active
        filters, normalizations and axis transforms are already applied.

        Returns
        -------
        list of (ndarray, ndarray) -- one (x, y) tuple per visible trace.
        """
        return [(np.asarray(l.get_xdata()), np.asarray(l.get_ydata()))
                for l in self._lines]
