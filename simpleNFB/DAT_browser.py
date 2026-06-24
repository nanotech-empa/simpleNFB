'''
DAT_browser.py
--------------
spectrumBrowser widget for Nanonis DAT spectroscopy data in Jupyter notebooks.
'''

import bisect
import os
import traceback
from pathlib import Path

import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d
from scipy.signal import medfilt, savgol_filter

from spmpy import Spm
from .base_browser import BaseBrowser
from .process_utils import (rebin_intensity_nm_to_ev, smooth_data, group_average,
                             relative_position, despike_z_score, moving_average)
from .widget_helpers import HBox, VBox, Btn_Widget, Text_Widget, Selection_Widget


class fileBrowser(BaseBrowser):
    '''
    Interactive browser for Nanonis DAT spectroscopy data.

    Public attributes:
        figure    – matplotlib Figure
        axes      – matplotlib Axes
        spec      – list of Spm objects for loaded files
        spec_data – list of ndarrays (raw / processed y-data)
        spec_x    – list of ndarrays (x-data per spectrum)

    Key methods:
        update_axes()      – refresh plot from current spec_data / spec_x
        update_image_data() – reload and process channel data
        save_figure(a)     – save figure to browser_outputs/
        save_data(a)       – save CSV to browser_outputs/
    '''

    _INFO_BUILDERS: dict = {
        'STML':               '_info_stml',
        'bias spectroscopy':  '_info_bias_spec',
        'THz amplitude sweep':'_info_thz',
        'Z spectroscopy':     '_info_z_spec',
        'History Data':       '_info_history',
    }

    def __init__(self, figsize=(3.5, 2.8), fontsize: int = 8, titlesize: int = 5,
                 cmap: str = 'Greys_r', home_directory: str = './',
                 sxmBrowser=None) -> None:
        # --- matplotlib defaults ---
        mpl.rcParams['figure.dpi']          = 100
        mpl.rcParams['axes.linewidth']      = 0.8
        mpl.rcParams['font.family']         = ['Microsoft Sans Serif']
        mpl.rcParams['font.size']           = 8
        mpl.rcParams['axes.labelpad']       = 3
        mpl.rcParams['xtick.labelsize']     = 7
        mpl.rcParams['ytick.labelsize']     = 7
        mpl.rcParams['axes.labelsize']      = 7

        # --- state ---
        self._aspect   = figsize[1] / figsize[0]  # height / width, preserved on resize
        self._resizing = False
        self.img           = None
        with plt.ioff():
            self.figure, self.axes = plt.subplots(ncols=1, figsize=figsize, num='dat', dpi=150)
        self.figure.canvas.header_visible  = False
        self.figure.canvas.resizable       = True
        self.figure.canvas.layout.width    = '100%'
        self.figure.canvas.mpl_connect('draw_event', self._on_figure_draw)
        self.axes2         = None
        self.cb            = None
        self.wfAxes        = None
        self.sxmBrowser    = sxmBrowser
        self.fontsize      = fontsize
        self.titlesize     = titlesize
        self.colorMap      = cmap
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

        self.axes.plot(self.spec_x[0], self.spec_data[0])

        self._build_widgets()
        self._build_layout()
        self._connect_observers()

        # sxmBrowser coupling must happen after widgets are built
        if sxmBrowser is None:
            self.referenceLocBtn.disabled = True

        self.display()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        """Instantiate all ipywidgets; no observers set here."""
        L   = lambda w: widgets.Layout(visibility='visible', width=f'{w}px')
        FL  = lambda w: widgets.Layout(display='flex', width=f'{w}%')
        FLB = lambda w: widgets.Layout(display='flex', width=f'{w}%',
                                       align_items='center', justify_content='center')
        FLH = lambda w: widgets.Layout(visibility='hidden', display='flex', width=f'{w}%')

        # selections
        self.rootFolder = widgets.Text(description='',
                                       layout=widgets.Layout(display='flex', width='95%'))
        self.directorySelection = widgets.Select(options=self.directories, rows=8,
                                                 layout=FL(98))
        self.directoryDisplayDepth = widgets.Dropdown(
            description='depth', value=1, options=['full', 1, 2, 3, 4, 5],
            tooltip='Depth of folder structure shown in selection menu',
            layout=FLB(75), style={'description_width': '40px'})
        self.selectionList = widgets.SelectMultiple(
            options=self.dat_files, value=[], description='', rows=30, layout=FL(98))
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

        # text
        self.filenameText = Text_Widget('')
        self.indexText    = Text_Widget('0')
        self.errorText    = Selection_Widget([], 'Out:', rows=5)
        self.saveNote     = Text_Widget('', description='note',
                                        layout=FL(98), style={'description_width': '30px'})

        # image-control buttons
        self.nextBtn         = Btn_Widget('', layout=L(30), icon='arrow-circle-down',
                                          tooltip='Load next file in list')
        self.previousBtn     = Btn_Widget('', layout=L(30), icon='arrow-circle-up',
                                          tooltip='Load previous file in list')
        self.flattenBtn      = widgets.ToggleButton(description='', value=False, layout=L(30),
                                                    icon='barcode',
                                                    tooltip='Normalize each curve to max=1')
        self.invertBtn       = widgets.ToggleButton(description='', value=False, layout=L(30),
                                                    icon='exchange', tooltip='Invert horizontal direction')
        self.fixZeroBtn      = widgets.ToggleButton(description='', value=False, layout=L(30),
                                                    icon='neuter', tooltip='Subtract local baseline')
        self.referenceLocBtn = Btn_Widget('', layout=L(30), icon='map-marker',
                                          tooltip='Plot tip location on image browser')
        self.saveBtn         = Btn_Widget('', layout=L(30), icon='file-image-o',
                                          tooltip='Save figure to browser_outputs/')
        self.copyBtn         = Btn_Widget('', layout=L(30), icon='clipboard',
                                          tooltip='Save figure and copy to clipboard')
        self.csvBtn          = Btn_Widget('', layout=L(30), icon='list-ul',
                                          tooltip='Save data to browser_outputs/ as .csv')
        self.generateWaterFallBtn = Btn_Widget('Waterfall', disabled=True)
        self.legendBtn       = widgets.ToggleButton(description='', value=True, layout=L(30),
                                                    icon='tags', tooltip='Toggle legend')

        # offset
        self.offsetBtn     = widgets.ToggleButton(description='', value=False, layout=L(30),
                                                   icon='navicon', tooltip='Apply vertical offset')
        self.offset_value  = widgets.FloatText(value=0.1e-12, description='offset:',
                                               step=.1e-12, readout_format='.1e', layout=L(120))

        # colormap / marker
        self.cmapSelection   = widgets.Dropdown(description='colormap:', options=plt.colormaps(),
                                                value=self.colorMap, layout=FL(98),
                                                style={'description_width': '80px'})
        self.markerSelection = widgets.Dropdown(description='marker:',
                                                options=['N', 'o', '*', 's', '^', 'X'],
                                                value='o', layout=FL(98),
                                                style={'description_width': '80px'})

        # smoothing (legacy panel kept for UI compat)
        self.smoothBtn    = widgets.ToggleButton(description='', value=False, layout=L(30),
                                                  icon='filter', tooltip='Apply Savitzky-Golay filter')
        self.windowParam  = widgets.BoundedIntText(description='window:', value=3, min=3, max=101, step=2, layout=L(120))
        self.orderParam   = widgets.BoundedIntText(description='order:', value=1, min=1, max=5, step=1, layout=L(120))

        # settings panel toggle
        self.settingsBtn = widgets.ToggleButton(description='', icon='gear', value=False,
                                                tooltip='Display settings panel', layout=L(30))

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
        self.defaultLegendToggle   = widgets.ToggleButton(value=False, description='default',
                                                           layout=FLB(30))
        self.customLegendToggle    = widgets.ToggleButton(value=False, description='custom',
                                                           layout=FLB(30))
        self.parameterLegendToggle = widgets.ToggleButton(value=True, description='parameter',
                                                           layout=FLB(30))
        self.legendText            = widgets.Select(value='', options=[''], rows=5,
                                                    layout=FL(98), disabled=False)
        self.parameterLegendList   = widgets.Dropdown(
            options=['Filename', 'Z (m)', 'Current [A]', 'Bias [V]',
                     'Exposure Time [ms]', 'Center Wavelength [nm]', 'Selected Grating'],
            value='Filename', description='Parameter:', layout=FL(98),
            style={'description_width': '80px'})
        self.legendEntry  = widgets.Text(description='', tooltip='New legend text',
                                          layout=FL(98), disabled=False)
        self.legendToggle = widgets.ToggleButton(value=True, description='legend', layout=FLB(48))
        self.legendUpdate = widgets.Button(description='Update', layout=FLB(48), disabled=False)

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

    def _build_layout(self) -> None:
        """Assemble widgets into HBox/VBox/Accordion containers."""
        FL  = lambda w: widgets.Layout(display='flex', width=f'{w}%')
        FLB = lambda w: widgets.Layout(display='flex', width=f'{w}%',
                                       align_items='center', justify_content='center')
        FLH = lambda w: widgets.Layout(visibility='hidden', display='flex', width=f'{w}%')

        self.h_new_filter_layout = HBox(children=[
            widgets.Label('New', layout=FLB(24)), self.newFilterText, self.addFilterBtn])
        self.v_filter_layout = VBox(children=[
            widgets.Label('Filter', layout=FL(50)),
            self.filterSelection, self.h_new_filter_layout])
        self.h_process_layout = HBox(children=[
            self.flattenBtn, self.fixZeroBtn, self.referenceLocBtn],
            layout=FL(98))
        self.h_selection_btn_layout = HBox(children=[
            self.refreshBtn, self.csvBtn, self.saveBtn, self.copyBtn, self.settingsBtn],
            layout=FL(98))
        self.v_channel_layout = VBox(children=[
            self.channelXSelect, self.channelYSelect, self.saveNote], layout=FL(48))
        self.v_file_select_layout = VBox(children=[
            HBox(children=[widgets.Label('Folder', layout=FL(24)),
                           self.directoryDisplayDepth], layout=FL(98)),
            self.directorySelection,
            widgets.Label('Files', layout=FL(50)),
            self.selectionList,
            self.v_filter_layout],
            layout=FL(20))
        self.v_btn_layout = VBox(children=[
            self.h_selection_btn_layout, self.h_process_layout,
            self.cmapSelection, self.markerSelection], layout=FL(48))
        self.h_user_layout = HBox(children=[
            self.v_channel_layout, self.v_btn_layout], layout=FLB(98))
        self.v_settings_layout = widgets.Accordion(children=[
            VBox(children=[
                HBox(children=[self.parameterLegendToggle,
                               self.customLegendToggle], layout=FL(98)),
                self.legendText, self.legendEntry,
                HBox(children=[self.legendToggle, self.legendUpdate], layout=FL(98)),
                self.parameterLegendList], layout=FLH(98)),
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
                self.xLimitsMin, self.xLimitsMax,
                HBox(children=[self.xLimitsBtn, self.xLimitLock], layout=FL(98)),
                self.yLimitsMin, self.yLimitsMax,
                HBox(children=[self.yLimitsBtn, self.yLimitLock], layout=FL(98))],
                layout=FLH(98)),
        ], layout=FLH(20),
        titles=['Legend Settings', 'Title Settings', 'Filter Settings',
                'STML Mode', '2D Plot', 'Axes Controls'])

        self.v_image_layout = VBox(children=[
            self.figure.canvas, self.h_user_layout], layout=FL(60))
        self.h_main_layout = VBox(children=[
            HBox(children=[
                widgets.Label('Session', layout=widgets.Layout(
                    display='flex', justify_content='flex-start', width='5%')),
                self.rootFolder], layout=FL(99)),
            HBox(children=[
                self.v_file_select_layout, self.v_image_layout,
                self.v_settings_layout], layout=FL(99))],
            layout=FL(100))
        self.v_settings_layout.layout.min_width  = '200px'
        self.v_file_select_layout.layout.min_width = '200px'

    def _connect_observers(self) -> None:
        """Wire all observe() and on_click() callbacks."""
        for child in self.v_settings_layout.children[:-1]:  # exclude axes controls
            if type(child) == type(self.v_btn_layout):
                for ch in child.children:
                    ch.observe(self.handler_settingsChange, names='value')
            child.observe(self.handler_settingsChange, names='value')

        for btn in (self.offsetToggle, self.offsetSize,
                    self.svgToggle, self.svgSize, self.svgOrder,
                    self.medFiltBtn, self.medFiltSize,
                    self.despikeBtn, self.despikeWindow, self.despikeThreshold,
                    self.movAvgBtn, self.movAvgSize,
                    self.thresholdToggle, self.thresholdValue,
                    self.averageToggle):
            btn.observe(self.redraw_image, names='value')

        self.groupSize.observe(self.update_legend_settings, names='value')
        self.averageToggle.observe(self.update_legend_settings, names='value')
        self.groupSize.observe(self.update_legend_mode, names='value')
        self.defaultLegendToggle.observe(self.update_legend_mode, names='value')
        self.customLegendToggle.observe(self.update_legend_mode, names='value')
        self.legendToggle.observe(self.handler_settingsChange, names='value')
        self.legendUpdate.on_click(self.update_legend_entry)

        self.settingsBtn.observe(self.handler_settingsDisplay, names='value')
        self.yLimitsBtn.on_click(self.handler_update_axes_limits)
        self.xLimitsBtn.on_click(self.handler_update_axes_limits)

        self.saveBtn.on_click(self.save_figure)
        self.copyBtn.on_click(self.copy_figure)
        self.csvBtn.on_click(self.save_data)
        self.generateWaterFallBtn.on_click(self.generateWaterFall)
        self.refreshBtn.on_click(self.handler_root_folder_update)

        for btn in (self.flattenBtn, self.legendBtn, self.fixZeroBtn):
            btn.observe(self.redraw_image, names='value')

        self.referenceLocBtn.on_click(self.plotSpectrumLocations)
        self.plot2DUpdateBtn.on_click(self._redraw)
        self.plot2DClimMode.observe(self._update_clim_widget_state, names='value')
        for widget in (self.plot2DYMin, self.plot2DYMax,
                       self.plot2DXMin, self.plot2DXMax,
                       self.plot2DVMin, self.plot2DVMax):
            widget.observe(self._redraw, names='value')

        self.rootFolder.observe(self.handler_root_folder_update, names='value')
        self.directorySelection.observe(self.handler_folder_selection, names=['value'])
        self.directoryDisplayDepth.observe(self.update_directories, names=['value'])
        self.selectionList.observe(self.handler_file_selection, names=['value'])
        self.filterSelection.observe(self.handler_folder_selection, names='value')
        self.addFilterBtn.on_click(self.handler_update_filters)
        self.channelXSelect.observe(self.handler_channel_selection, names='value')
        self.channelYSelect.observe(self.handler_channel_selection, names=['value'])

        for widget in (self.smoothBtn, self.windowParam, self.orderParam,
                       self.offsetBtn, self.offset_value, self.cmapSelection):
            widget.observe(self.handler_update_axes, names='value')

        self.stmlToggle.observe(self.handler_settingsChange, names='value')
        self.plasmonReference.observe(self.handler_update_plasmonic_reference, names='value')

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display(self) -> None:
        """Render the browser widget."""
        display.clear_output(wait=True)
        display.display(self.h_main_layout)

    def _on_figure_draw(self, event) -> None:
        """Maintain original aspect ratio on every ipympl draw (fires after resize too)."""
        if self._resizing:
            return
        w_in, h_in = self.figure.get_size_inches()
        w = round(w_in * self.figure.dpi)
        if w <= 0:
            return
        h_target = round(w * self._aspect)
        if abs(round(h_in * self.figure.dpi) - h_target) > 2:
            self._resizing = True
            try:
                self.figure.set_size_inches(w_in, h_target / self.figure.dpi)
            finally:
                self._resizing = False

    def _redraw(self, *_) -> None:
        """Update axes and refresh the canvas."""
        if self.plot2DToggle.value:
            self._render_2d()
        else:
            self.update_axes()
        self.figure.canvas.draw()

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def save_figure(self, a) -> None:
        """Save the current figure to browser_outputs/ at 500 dpi."""
        self.saveBtn.icon = 'hourglass-start'
        out_dir = self.active_dir / 'browser_outputs'
        out_dir.mkdir(exist_ok=True)
        stem = (f'{str(self.directorySelection.value).split(chr(92))[-1]}'
                f'_{self.spec[0].name.split(".")[0]}_{self.channelYSelect.value[0]}')
        if self.saveNote.value:
            stem += f'_{self.saveNote.value}'
        self.last_save_fname = str(out_dir / f'{stem}.png')
        self.figure.savefig(self.last_save_fname, dpi=500, format='png',
                            transparent=True, bbox_inches='tight')
        self.updateErrorText('Figure Saved')
        self.saveNote.value = ''
        self.saveBtn.icon = 'file-image-o'

    def save_data(self, a) -> None:
        """Save current spectra to browser_outputs/ as CSV."""
        self.csvBtn.icon = 'hourglass-start'
        out_dir = self.active_dir / 'browser_outputs'
        out_dir.mkdir(exist_ok=True)
        stem = (f'{str(self.directorySelection.value).split(chr(92))[-1]}'
                f'_{self.spec[0].name.split(".")[0]}_{self.channelYSelect.value[0]}')
        if self.saveNote.value:
            stem += f'_{self.saveNote.value}'
        fname = out_dir / stem
        x_hdr = f'{self.spec_info[0]["x_label"]} ({self.spec_info[0]["x_unit"]})'
        header = ','.join(f'{x_hdr},{lbl[:-4]}' for lbl in self.labels)
        out = np.column_stack([arr for pair in zip(self.spec_x, self.spec_data) for arr in pair])
        np.savetxt(f'{fname}.csv', out, delimiter=',', header=header)
        self.updateErrorText('Saved CSV')
        self.csvBtn.icon = 'list-ul'

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------

    def redraw_image(self, a) -> None:
        """Reload and replot when a processing toggle changes."""
        if self.loaded_experiments is not None:
            self.update_image_data()
            self._redraw()

    def load_new_image(self, filename=None):
        """Load the currently selected DAT file(s) and populate channel lists."""
        directory = self.directories[self.directorySelection.index]
        if directory != self.active_dir:
            directory = os.path.join(self.active_dir, directory)
        if filename is None:
            files = [os.path.join(directory, self.dat_files[idx])
                     for idx in self.spec_index]
            self.spec = [Spm(f) for f in files]
            self.filenameText.value = ','.join(
                self.all_files[idx] for idx in self.spec_index)
            self.loaded_experiments = [s.header['Experiment'] for s in self.spec]
            self._scan_cache = {}
            self._update_channel_selection()
            self.update_image_data()
        else:
            return Spm(os.path.join(directory, filename))

    def smooth_data(self, data):
        """Apply Savitzky-Golay filter using current widget params."""
        return smooth_data(data, self.svgSize.value, self.svgOrder.value)

    def _cache_scan_param(self, spec, key: str):
        """Return spec.get_param(key) with per-file caching."""
        cache_key = (id(spec), key)
        if cache_key not in self._scan_cache:
            self._scan_cache[cache_key] = spec.get_param(key)
        return self._scan_cache[cache_key]

    def update_image_data(self, filename=None) -> None:
        """Reload channel data, apply STML normalization if active."""
        channelX = self.channelXSelect.value
        channelY = self.channelYSelect.value
        if filename is None:
            self.spec_data, self.spec_info, self.labels, self.spec_x = [], [], [], []
            if len(self.selectionList.value) >= 1 and len(channelY) == 1:
                for spec in self.spec:
                    spec_data, yunit = spec.get_channel(channelY[0])
                    self.spec_data.append(spec_data)
                    if channelX == 'Index':
                        spec_x, xunit = np.arange(len(spec_data)), 'N'
                    else:
                        spec_x, xunit = spec.get_channel(channelX)
                    self.spec_info.append({'x_unit': xunit, 'y_unit': yunit, 'x_label': channelX})
                    self.spec_x.append(spec_x)
                    self.labels.append(spec.name)
            if len(self.selectionList.value) == 1 and len(channelY) > 1:
                spec = self.spec[0]
                for ch in channelY:
                    spec_data, yunit = spec.get_channel(ch)
                    self.spec_data.append(spec_data)
                    spec_x, xunit = spec.get_channel(channelX)
                    self.spec_info.append({'x_unit': xunit, 'y_unit': yunit, 'x_label': channelX})
                    self.spec_x.append(spec_x)
                    self.labels.append(ch)
            self.update_scan_info()
        else:
            spec = self.load_new_image(filename=filename)
            spec.get_channel(channelY[0])
            spec.get_channel(channelX)

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
            fr'Exposure Time (s): {float(spec.header["Exposure Time [ms]"])/1000:.0f}, '
            fr'$\lambda_c$: {spec.header["Center Wavelength [nm]"]}, '
            fr'grating: {spec.header["Selected Grating"]}')
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
        experiments = self.loaded_experiments
        if experiments.count(experiments[0]) != len(experiments):
            self.updateErrorText('Please ensure all selections are the same measurement type')
            return

        spec       = self.spec[0]
        experiment = experiments[0]
        label: list = []
        setpoint_str = feedback_str = ''

        if self.nameToggle.value:
            label.append(f'Experiment: {experiment} $\\rightarrow$ filename: {spec.name}')

        for key, method in self._INFO_BUILDERS.items():
            if key in experiment:
                setpoint_str, feedback_str = getattr(self, method)(spec, label)
                break

        # date
        if len(self.spec) > 1:
            date_str = f'Date: {self.spec[0].header["Saved Date"]} $\\rightarrow$ {self.spec[-1].header["Saved Date"]}'
        else:
            date_str = f'Date: {spec.header["Saved Date"]}'

        if self.setpointToggle.value and setpoint_str:
            if self.feedbackToggle.value and feedback_str:
                setpoint_str += f' $\\rightarrow$ {feedback_str}'
            label.append(setpoint_str)
        if self.locationToggle.value:
            location = self.directories[self.directorySelection.index]
            if self.depthSelection.value != 'full':
                location = '\\'.join(str(location).split('\\')[-int(self.depthSelection.value):])
            label.append(f'location: {location}')
        if self.dateToggle.value:
            label.append(date_str)
        if self.svgToggle.value:
            label.append(f'Savitzky-Golay Filter $\\rightarrow$ Window: {self.svgSize.value}, '
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
        if self.parameterLegendToggle.value:
            if self.parameterLegendList.value != 'Filename':
                labels = [spec.header[self.parameterLegendList.value] for spec in self.spec]
                param = self.parameterLegendList.value
                for i, val in enumerate(labels):
                    v = float(val)
                    if 'z' in param[0].lower():
                        labels[i] = f'{v * 1e9:.2f} nm'
                    elif 'current' in param.lower():
                        labels[i] = f'{v * 1e12:.2f} pA'
                    elif 'bias' in param.lower():
                        if abs(v) < 0.1:
                            labels[i] = f'{v * 1e3:.2f} mV'
                        else:
                            labels[i] = f'{v:.2f} V'
                    elif param == 'Exposure Time [ms]':
                        labels[i] = f'{v / 1000:.2f} s'
            else:
                labels = [spec.name for spec in self.spec]
            if self.averageToggle.value and len(self.spec_data) % self.groupSize.value == 0:
                labels = [labels[i] for i in range(0, len(labels), self.groupSize.value)]
        elif self.customLegendToggle.value:
            labels = list(self.legendText.options)
        else:
            labels = self.labels
        return labels

    def _plot_spectra(self, ax, x_values: list, y_values: list, labels: list) -> None:
        """Plot all spectra onto ax with colormap-derived colors."""
        colors = plt.cm.get_cmap(str(self.cmapSelection.value))(
            np.linspace(0, 1, len(y_values)))
        for x, y, lbl, c in zip(x_values, y_values, labels, colors):
            ax.plot(x, y, color=c, label=lbl)

    def _add_stml_axis(self, ax) -> None:
        """Add secondary wavelength axis for STML mode."""
        xmin, xmax, ymin, ymax = ax.axis()
        if self.axes2 is not None:
            self.axes2.remove()
            self.axes2 = None
        self.axes2 = ax.twiny()
        self.axes2.set_xlabel('Wavelength (nm)', fontsize=self.fontsize)
        self.axes2.set_xscale('function',
                               functions=(lambda en: 1240 / (en + 1e-9),
                                          lambda lm: 1240 / (lm + 1e-9)))
        for x in self.spec_x:
            energy = x[(x >= xmin) & (x <= xmax)]
            self.axes2.plot(1240 / energy, energy, alpha=0)
        self.axes2.xaxis.set_minor_locator(AutoMinorLocator(2))
        self.axes2.tick_params(axis='x', which='both', direction='in')
        self.axes2.set_zorder(ax.get_zorder() - 1)
        self.axes2.xaxis.set_zorder(ax.xaxis.get_zorder() + 1)
        ax.set_ylim(ymin, ymax)

    def update_axes(self) -> None:
        """Re-render all spectra with current filter/display settings."""
        if not self.figure:
            self.figure, self.axes = plt.subplots(ncols=1, figsize=(8, 8))
        ax = self.axes
        if self.cb:
            self.cb.remove()
            self.cb = None
        ax.clear()
        if self.axes2 is not None:
            self.axes2.remove()
            self.axes2 = None

        x_values, y_values = self._apply_filters(self.spec_x, self.spec_data)
        labels = self._build_legend_labels(x_values, y_values)
        self._plot_spectra(ax, x_values, y_values, labels)

        ax.set_title(self.spec_label, fontsize=self.titlesize, loc='left')
        ax.set_xlabel(f'{self.spec_info[0]["x_label"]} ({self.spec_info[0]["x_unit"]})',
                      fontsize=self.fontsize)
        ax.set_ylabel(f'{self.channelYSelect.value[0]} ({self.spec_info[0]["y_unit"]})',
                      fontsize=self.fontsize)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis='y', which='both', direction='in')
        ax.tick_params(axis='x', which='both', direction='in')

        if self.legendToggle.value:
            idx = bisect.bisect_left([4, 16], len(ax.lines))
            ax.legend(draggable=True, fontsize=self.legendFontsize[idx], frameon=False)

        if self.stmlToggle.value and 'stml' in (self.loaded_experiments or [''])[0].lower():
            self._add_stml_axis(ax)

        self._sync_axis_limit_sliders()
        if self.xLimitLock.value:
            self.axes.set_xlim(self.xLimitsMin.value, self.xLimitsMax.value)
            if self.axes2 is not None:
                self.axes2.set_xlim(1240 / self.xLimitsMax.value,
                                    1240 / self.xLimitsMin.value)
        if self.yLimitLock.value:
            self.axes.set_ylim(self.yLimitsMin.value, self.yLimitsMax.value)

        self.figure.tight_layout(pad=1)

    # ------------------------------------------------------------------
    # Alternate views
    # ------------------------------------------------------------------

    def generateWaterFall(self, a=None) -> None:
        if self.wfAxes is None:
            return
        positions = [[spec.get_param('x')[0], spec.get_param('y')[0]] for spec in self.spec]
        x0, y0 = positions[0]
        distances = [np.sqrt((p[0] - x0) ** 2 + (p[1] - y0) ** 2) for p in positions]
        self.wfAxes.imshow(np.rot90(self.spec_data),
                           extent=[distances[0], distances[-1],
                                   self.spec_x[0].min(), self.spec_x[0].max()],
                           aspect='auto', cmap=self.cmapSelection.value)
        self.wfAxes.set_xlabel('distance (nm)')
        self.wfAxes.set_ylabel('Bias (V)')
        self.wfAxes.set_title('dI/dV')
        self.save_figure(self.generateWaterFallBtn)

    def plotSpectrumLocations(self, a) -> None:
        if self.sxmBrowser is None or self.sxmBrowser.img is None:
            return
        ax     = self.sxmBrowser.axes
        colors = plt.cm.get_cmap(self.cmapSelection.value)(
            np.linspace(0, 1, len(self.axes.lines)))
        k = 0
        for i, spec in enumerate(self.spec):
            if self.averageToggle.value and len(self.spec_data) % self.groupSize.value == 0:
                if i not in range(0, len(self.spec), self.groupSize.value):
                    continue
            rx, ry = relative_position(self.sxmBrowser.img, spec)
            if self.markerSelection.value != 'N':
                ax.plot(rx, ry, marker=self.markerSelection.value,
                        markersize=10, color=colors[k])
            else:
                start = max(self.spec_index) - min(self.spec_index) + 1
                ax.text(rx, ry, f'{start - i}', color='r', fontsize=10,
                        ha='center', va='center')
            k += 1
        self.sxmBrowser.figure.canvas.draw()

    def _compute_clim(self, z, x_ref, y_vals) -> tuple:
        """Return (vmin, vmax) for the pcolormesh based on the current scale mode."""
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
        """Render filtered spectra as a 2D pcolormesh."""
        if not self.spec or self.loaded_experiments is None:
            return
        ax = self.axes
        if self.cb:
            self.cb.remove()
            self.cb = None
        if self.axes2 is not None:
            self.axes2.remove()
            self.axes2 = None
        ax.clear()

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
        y_param = self.plot2DYParam.value
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
        mesh = ax.pcolormesh(x_ref, y_vals, z,
                             cmap=self.cmapSelection.value, shading='auto',
                             vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        self.cb = self.figure.colorbar(mesh, cax=cax)
        self.cb.set_label(
            f'{self.channelYSelect.value[0]} ({self.spec_info[0]["y_unit"]})',
            fontsize=self.fontsize)

        x_auto = f'{self.spec_info[0]["x_label"]} ({self.spec_info[0]["x_unit"]})'
        ax.set_xlabel(self.plot2DXLabel.value or x_auto, fontsize=self.fontsize)
        ax.set_ylabel(self.plot2DYLabel.value or y_label, fontsize=self.fontsize)

        xlo, xhi = self.plot2DXMin.value, self.plot2DXMax.value
        ylo, yhi = self.plot2DYMin.value, self.plot2DYMax.value
        if xlo != xhi:
            ax.set_xlim(xlo, xhi)
        if ylo != yhi:
            ax.set_ylim(ylo, yhi)

        ax.set_title(self.spec_label, fontsize=self.titlesize, loc='left')
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis='both', which='both', direction='in')

    # ------------------------------------------------------------------
    # Navigation
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

    def _update_info_text(self) -> None:
        idx = self.spec_index[0]
        self.filenameText.value = self.dat_files[idx]
        self.selectionList.value = [self.dat_files[idx]]

    def _sync_axis_limit_sliders(self) -> None:
        xmin, xmax, ymin, ymax = self.axes.axis()
        if not self.xLimitLock.value:
            self.xLimitsMin.value = xmin
            self.xLimitsMax.value = xmax
        if not self.yLimitLock.value:
            self.yLimitsMin.value = ymin
            self.yLimitsMax.value = ymax

    def _update_legend_parameters(self, a) -> None:
        if a['owner'] == self.selectionList:
            parameter_keys = ['Filename'] + list(self.spec[0].header.keys())
            self.parameterLegendList.options = parameter_keys
            self.parameterLegendList.value   = 'Filename'

    def nextDisplay(self, a) -> None:
        idx = self.spec_index[-1]
        if idx < len(self.all_files) - 1:
            self.spec_index = [idx + 1]
            self._update_info_text()

    def previousDisplay(self, a) -> None:
        idx = self.spec_index[0]
        if idx > 0:
            self.spec_index = [idx - 1]
            self._update_info_text()

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def handler_settingsChange(self, a) -> None:
        if a['owner'] != self.selectionList:
            self.redraw_image(a)

    def handler_settingsDisplay(self, a) -> None:
        self._set_settings_visibility(self.settingsBtn.value)

    def update_legend_mode(self, a) -> None:
        if a['owner'] == self.parameterLegendToggle and a['new'] is True:
            self.customLegendToggle.value = False
        elif a['owner'] == self.customLegendToggle and a['new'] is True:
            self.parameterLegendToggle.value = False
        elif a['owner'] == self.groupSize:
            if self.averageToggle.value and len(self.spec_data) % self.groupSize.value != 0:
                self.parameterLegendToggle.value = True
                self.customLegendToggle.value    = False
        self._redraw()

    def update_legend_entry(self, a) -> None:
        entry   = self.legendText.value
        options = list(self.legendText.options)
        index   = options.index(entry)
        options[index] = self.legendEntry.value
        self.legendText.options = options
        self._redraw()

    def update_legend_settings(self, a) -> None:
        if a['owner'] in (self.selectionList, self.averageToggle, self.groupSize):
            new_selection = list(self.selectionList.value)
            if self.averageToggle.value and len(self.spec_data) % self.groupSize.value == 0:
                new_selection = [new_selection[i]
                                 for i in range(0, len(new_selection), self.groupSize.value)]
            self.legendText.options = new_selection
            if new_selection:
                self.legendText.value = new_selection[0]

    def handler_folder_selection(self, a) -> None:
        index = 0
        if type(a) == type(self.refreshBtn):
            index = self.selectionList.index
        directory = self.directories[self.directorySelection.index]
        self.sxm_files, self.dat_files = [], []
        for f in os.listdir(directory):
            if '.sxm' in f:
                self.sxm_files.append(f)
            elif '.dat' in f:
                if 'all' in self.filterSelection.value or any(
                        filt in f for filt in self.filterSelection.value):
                    self.dat_files.append(f)
        files = [(f, os.path.getmtime(os.path.join(directory, f))) for f in self.dat_files]
        self.dat_files = [f[0] for f in sorted(files, key=lambda x: x[1], reverse=True)]
        self.all_files = self.sxm_files + self.dat_files
        self.selectionList.options = self.dat_files
        if self.dat_files:
            self.filenameText.value = (self.dat_files[index] if isinstance(index, int)
                                       else self.dat_files[index[0]])
            if self.filenameText.value in self.selectionList.options:
                self.selectionList.value = [self.filenameText.value]
            self.plasmonReference.options = (
                ['None'] + [f for f in self.dat_files if 'stml' in f.lower()])
            self.plasmonReference.value = 'None'

    def handler_file_selection(self, update) -> None:
        self.spec_index = [self.dat_files.index(v) for v in self.selectionList.value]
        try:
            if self.selectionList.value:
                self.load_new_image()
                self._update_legend_parameters(update)
                self.update_legend_settings(update)
                self._redraw()
        except Exception as err:
            self.updateErrorText('file selection error: ' + str(err))
            print(traceback.format_exc())

    def handler_channel_selection(self, update) -> None:
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
        if self.plasmonReference.value == 'None':
            return
        directory = self.directories[self.directorySelection.index]
        if directory != self.active_dir:
            directory = os.path.join(self.active_dir, directory)
        if self.plasmonReference.value in self.dat_files:
            ref = Spm(os.path.join(directory, self.plasmonReference.value))
            ref_y, _ = ref.get_channel('Intensity')
            ref_x, _ = ref.get_channel('Wavelength')
            ref_y = abs(savgol_filter(ref_y, 21, 1))
            ref_y *= (ref_y > 35)
            ref_y /= np.max(ref_y)
            self.plasmonInfo['spm']    = ref
            self.plasmonInfo['file']   = self.plasmonReference.value
            self.plasmonInfo['interp'] = interp1d(1240 / ref_x, ref_y,
                                                   bounds_error=False, fill_value=0.0)

    def handler_update_axes(self, a) -> None:
        self.update_scan_info()
        self._redraw()

    def handler_update_axes_limits(self, a) -> None:
        if a == self.xLimitsBtn:
            self.axes.set_xlim(self.xLimitsMin.value, self.xLimitsMax.value)
            if self.axes2 is not None:
                self.axes2.set_xlim(1240 / self.xLimitsMax.value,
                                    1240 / self.xLimitsMin.value)
        elif a == self.yLimitsBtn:
            self.axes.set_ylim(self.yLimitsMin.value, self.yLimitsMax.value)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def make_figure(self, figsize=(7, 5), cols: int = 1) -> None:
        try:
            if self.figure:
                plt.close(self.figure)
        except Exception as err:
            print(err)
        self.figure, self.axes = plt.subplots(ncols=cols, figsize=figsize)
        if cols != 1:
            self.axs  = self.axes[1:]
            self.axes = self.axes[0]
