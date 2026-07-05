'''
SXM_browser.py
--------------
imageBrowser widget for Nanonis SXM scan data in Jupyter notebooks.
Uses plotly FigureWidget for rendering (replaces matplotlib).
'''

import os
import traceback
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from IPython import display
from scipy.ndimage import gaussian_filter, gaussian_laplace, median_filter

from spmpy import Spm
from .base_browser import BaseBrowser
from .process_utils import relative_position, remove_line_average
from .widget_helpers import HBox, VBox, Btn_Widget, Text_Widget


class fileBrowser(BaseBrowser):
    '''
    Interactive browser for Nanonis SXM image data.

    Public attributes:
        figure      – plotly FigureWidget (go.FigureWidget with a Heatmap trace)
        image_data  – ndarray of current processed image
        img         – Spm object for the loaded file

    Key methods:
        update_axes()       – refresh the plot from current image_data
        update_image_data() – reload and process channel data
        save_figure(a)      – save to browser_outputs/
    '''

    # Pixels permanently reserved in the right margin for the context legend
    # column. Baking this in keeps the plot box constant regardless of whether
    # context markers are shown, so scaleanchor aspect ratio never reflows.
    _CTX_LEGEND_W = 130

    _TEMPLATES_SUBDIR: str = 'sxm'

    _ALIGN_PARAMS: dict = {
        'upper left':  ('left',  'top'),
        'upper right': ('right', 'top'),
        'lower left':  ('left',  'bottom'),
        'lower right': ('right', 'bottom'),
    }

    def __init__(self, width: int = 1200, height: int = 1000, fontsize: int = 12,
                 titlesize: int = 12, cmap: str = 'greys',
                 home_directory: str = './') -> None:
        # Normalise cmap: strip matplotlib _r suffix → reversescale toggle
        _cmap_lower = cmap.lower()
        _reversed   = _cmap_lower.endswith('_r')
        _cmap_base  = _cmap_lower[:-2] if _reversed else _cmap_lower
        _available  = px.colors.named_colorscales()
        self._cmap  = _cmap_base if _cmap_base in _available else 'greys'

        # --- state ---
        self.img = None
        self.figure = go.FigureWidget(data=[go.Heatmap(
            z=[[0]], colorscale=self._cmap, reversescale=_reversed)])
        self._fig_width, self._fig_height = width, height
        self._last_crop = (1.0, 1.0)  # (h_crop, w_crop) updated on each image load
        self.fontsize  = fontsize
        self.titlesize = titlesize
        self.image_data = np.zeros((64, 64))
        self.image_info = {'height': 1, 'width': 1, 'unit': 'nm'}
        self.scan_dict: dict = {}
        self.scan_info = ''
        self.errors: list = []
        self.image_index = 0
        self._updating_limits = False
        self._scan_cache: dict = {}
        self._context_spm_cache: dict = {}  # dat_name → Spm, valid for current directory

        self.active_dir = Path(home_directory)
        self.sxm_files: list = []
        self.dat_files: list = []
        self.directories = [self.active_dir]

        self._build_widgets(_reversed)
        self._build_layout()
        self._connect_observers()
        self._apply_figure_layout()   # configure appearance after widgets exist
        self.display()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build_widgets(self, reversed_scale: bool = True) -> None:
        """Instantiate all ipywidgets; no observers set here."""
        L, FL, FLB, FLH = self._layout_helpers()
        self._L = L
        self._build_common_file_widgets()

        # selections
        self.selectionList = widgets.Select(
            description='', options=self.sxm_files, rows=27, layout=FL(98))
        self.channelSelect = widgets.Dropdown(description='', layout=L(165))
        self.refreshBtn = Btn_Widget(
            '', icon='refresh', tooltip='Reload file list', layout=FLB(24))
        self.saveNote     = Text_Widget(
            '', description='',
            tooltip='This text is appended to filename when the figure is saved',
            layout=FL(99))

        # image controls
        self.nextBtn      = Btn_Widget('', layout=L(40), icon='arrow-circle-down',
                                       tooltip='Load next image in list')
        self.previousBtn  = Btn_Widget('', layout=L(40), icon='arrow-circle-up',
                                       tooltip='Load previous image in list')
        self.linebylineBtn = widgets.ToggleButton(
            description='', value=False, layout=L(40), icon='align-justify',
            tooltip='Line by line linear subtraction')
        self.flattenBtn   = widgets.ToggleButton(
            description='', value=False, layout=L(40), icon='square-o',
            tooltip='Apply plane fit and subtraction')
        self.edgesBtn     = widgets.ToggleButton(
            description='', value=False, layout=L(40), icon='dot-circle-o',
            tooltip='Apply laplace filter (edge detection)')
        self.gaussianBtn  = widgets.ToggleButton(
            description='', value=False, layout=L(40), icon='bullseye',
            tooltip='Apply a 3x3 Gaussian filter')
        self.invertBtn    = widgets.ToggleButton(
            description='', value=False, layout=L(40), icon='exchange',
            tooltip='Invert sign of the image data')
        self.directionBtn = widgets.ToggleButton(
            description='', value=False, layout=L(40), icon='caret-square-o-right',
            tooltip='Select scan direction, default is forward')
        self.fixZeroBtn   = widgets.ToggleButton(
            description='', value=False, layout=L(40), icon='neuter',
            tooltip='Rescale image data so minimum value is zero')

        # outputs
        self.saveBtn = widgets.ToggleButton(
            value=True, description='', layout=FLB(24), icon='file-image-o',
            tooltip='Save to file when copying (toggle off to copy to clipboard only)')
        self.copyBtn = Btn_Widget(
            '', layout=FLB(24), icon='clipboard',
            tooltip='Copy figure to clipboard')
        self.codeBtn = Btn_Widget(
            '', layout=FLB(24), icon='file-code-o',
            tooltip='Export reproducible Python code to clipboard/new cell')

        # Header inspector (Concern 4): key list + read-only value
        self.headerKeySelect = widgets.Select(options=[], rows=8, layout=FL(98))
        self.headerValueText = widgets.Textarea(
            value='', disabled=True, rows=6, layout=FL(98))

        # Session timeline context (Concern 4): show .dat files taken between
        # this image's mtime and the next image's mtime.
        self.contextToggle = widgets.ToggleButton(
            value=False, description='Context', icon='clock-o', layout=FLB(48),
            tooltip='Show .dat files recorded during this image')
        self.contextSelect = widgets.SelectMultiple(
            options=[], rows=6,
            layout=widgets.Layout(display='none', width='98%'))
        self.session_index: list = []   # [(name, kind, mtime)] built during folder scan
        # Width (px) reserved in the right margin for the context legend column.
        # Exposed as a widget so the user can tune it and save it into a template.
        self.figCtxLegendW = widgets.BoundedIntText(
            value=self._CTX_LEGEND_W, min=0, max=400, description='Ctx legend W:',
            tooltip='Pixels reserved in the right margin for the context marker legend',
            layout=widgets.Layout(width='98%'), style={'description_width': '90px'})

        # colormap
        self.vmin = widgets.FloatText(
            value=0, description='Min:', step=.1,
            layout=FL(50), style={'description_width': '40px'})
        self.vmax = widgets.FloatText(
            value=1, description='Max:', step=.1,
            layout=FL(50), style={'description_width': '40px'})
        self.cmapSelection = widgets.Dropdown(
            description='Color Map:', options=px.colors.named_colorscales(),
            value=self._cmap,
            layout=FL(99), style={'description_width': '77px'})
        self.reverseScaleToggle = widgets.ToggleButton(
            description='', icon='exchange', value=reversed_scale, layout=L(40),
            tooltip='Reverse colorscale direction')

        # settings panel — visibility toggles
        self.configOptionBtn = widgets.ToggleButton(
            description='', icon='gear', value=True,
            tooltip='Display options panel', layout=FLB(24))
        self.titleToggle = widgets.ToggleButton(
            value=True,  description='Show Title',  tooltip='Toggle figure title',  layout=FLH(98))
        self.labelToggle = widgets.ToggleButton(
            value=False, description='Show Labels', tooltip='Toggle figure labels', layout=FLH(98))

        label_options = ['none', 'channel', 'bias', 'setpoint', 'feedback',
                         'date', 'filename', 'scalebar', 'filters']
        self.labelLabel       = widgets.Label(value='Figure Label Settings', layout=FLH(98))
        self.upperLeftSelect  = widgets.Dropdown(value='bias',     options=label_options,
                                                 description='UL:', layout=FLH(98),
                                                 style={'description_width': '40px'})
        self.upperRightSelect = widgets.Dropdown(value='filename', options=label_options,
                                                 description='UR:', layout=FLH(98),
                                                 style={'description_width': '40px'})
        self.lowerLeftSelect  = widgets.Dropdown(value='none',     options=label_options,
                                                 description='LL:', layout=FLH(98),
                                                 style={'description_width': '40px'})
        self.lowerRightSelect = widgets.Dropdown(value='scalebar', options=label_options,
                                                 description='LR:', layout=FLH(98),
                                                 style={'description_width': '40px'})
        self.labelColorSelect = widgets.ColorPicker(
            concise=True, description='Color:', value='red',
            layout=FLH(98), style={'description_width': '40px'})
        self.labelFontSize = widgets.IntText(
            description='size:', value=20,
            style={'description_width': '40px'}, layout=FL(98))

        self.titleLabel      = widgets.Label(value='Figure Title Settings', layout=FLH(98))
        self.channelToggle   = widgets.ToggleButton(value=True,  description='channel',       layout=FLH(98))
        self.setpointToggle  = widgets.ToggleButton(value=True,  description='Setpoint',      layout=FLH(98))
        self.feedbackToggle  = widgets.ToggleButton(value=True,  description='Feedback',      layout=FLH(98))
        self.locationToggle  = widgets.ToggleButton(value=True,  description='file location', layout=FLH(98))
        self.depthSelection  = widgets.Dropdown(
            value='full', options=['full', 1, 2, 3, 4, 5], description='Depth:',
            tooltip='Folder depth shown in title location string',
            layout=FLH(98), style={'description_width': '40px'})
        self.nameToggle      = widgets.ToggleButton(value=True,  description='Filename',  layout=FLH(98))
        self.directionToggle = widgets.ToggleButton(value=True,  description='Direction', layout=FLH(98))
        self.dateToggle      = widgets.ToggleButton(value=True,  description='Date',      layout=FLH(98))
        self.titleFontSize   = widgets.IntText(
            description='size:', value=9,
            style={'description_width': '40px'}, layout=FL(98))

        self.filterLabel    = widgets.Label(value='Image Filter Settings', layout=FL(98))
        self.gaussianToggle = widgets.ToggleButton(value=False, description='Gaussian', layout=FL(60))
        self.gaussianSize   = widgets.BoundedIntText(
            value=2, min=0, max=10, step=1, tooltip='Gaussian kernel size', layout=FL(40))
        self.medianToggle   = widgets.ToggleButton(value=False, description='Median',   layout=FL(60))
        self.medianSize     = widgets.BoundedIntText(
            value=3, min=1, max=20, step=1, tooltip='Median kernel size',   layout=FL(40))
        self.laplacToggle   = widgets.ToggleButton(value=False, description='Laplace',  layout=FL(60))
        self.laplaceSize    = widgets.BoundedIntText(
            value=1, min=1, max=10, step=1, tooltip='Laplace kernel size',  layout=FL(40))
        self._build_figure_settings_widgets(self._fig_width, self._fig_height)

    def _build_layout(self) -> None:
        """Assemble widgets into HBox/VBox containers."""
        _, FL, FLB, FLH = self._layout_helpers()

        self.h_process_btn_layout = HBox(children=[
            self.directionBtn, self.invertBtn])
        self.h_channel_layout = HBox(children=[
            widgets.Label('Channel'), self.channelSelect])
        self.v_color_layout = VBox(children=[
            HBox(children=[self.vmin, self.vmax]),
            HBox(children=[self.cmapSelection, self.reverseScaleToggle])])
        self.h_user_layout = HBox(
            children=[VBox(children=[self.h_channel_layout, self.h_process_btn_layout]),
                      self.v_color_layout],
            layout=widgets.Layout(display='flex', align_items='center',
                                  justify_content='center'))
        self.v_file_layout = VBox(children=[
            widgets.Label('Folder', layout=FL(98)),
            self.directorySelection,
            widgets.Label('Images'), self.selectionList,
            HBox(children=[self.contextToggle], layout=FL(98)),
            self.contextSelect,
            VBox(children=[
                HBox(children=[self.refreshBtn, self.saveBtn, self.copyBtn, self.codeBtn,
                               self.configOptionBtn]),
                widgets.Label('Note')]),
            self.saveNote],        # errorText moved to full-width Messages accordion
            layout=FL(98))
        self.v_settings_layout = widgets.Accordion(children=[
            VBox(children=[
                self.titleToggle, self.labelToggle,
                self.labelLabel,
                self.upperLeftSelect, self.upperRightSelect,
                self.lowerLeftSelect, self.lowerRightSelect,
                self.labelColorSelect, self.labelFontSize,
            ], layout=FLH(98)),
            VBox(children=[
                self.titleLabel,
                self.channelToggle, self.setpointToggle, self.feedbackToggle,
                self.locationToggle, self.depthSelection,
                self.nameToggle, self.directionToggle, self.dateToggle,
                self.titleFontSize,
            ], layout=FLH(98)),
            VBox(children=[
                widgets.Label('── Corrections ──', layout=FL(98)),
                HBox(children=[self.linebylineBtn,
                               widgets.Label('Line-by-line subtraction')], layout=FL(98)),
                HBox(children=[self.flattenBtn,
                               widgets.Label('Plane fit')],               layout=FL(98)),
                HBox(children=[self.fixZeroBtn,
                               widgets.Label('Set zero')],                layout=FL(98)),
                widgets.Label('── Filters ──', layout=FL(98)),
                HBox(children=[self.gaussianToggle, self.gaussianSize],   layout=FL(98)),
                HBox(children=[self.medianToggle,   self.medianSize],     layout=FL(98)),
                HBox(children=[self.laplacToggle,   self.laplaceSize],    layout=FL(98)),
            ], layout=FL(98)),
            VBox(children=[
                widgets.Label('Header key', layout=FL(98)),
                self.headerKeySelect,
                widgets.Label('Value', layout=FL(98)),
                self.headerValueText,
            ], layout=FLH(98)),
            self._figure_settings_tab(),
        ], layout=FLH(98),
        titles=['Label Settings', 'Title Settings', 'Filter Settings',
                'Header', 'Figure Settings'])
        # FigureWidget is itself a widget — no Output wrapper needed.
        # align_items='center'    — centres the fixed-px figure horizontally
        #   (works now that widget_helpers.VBox no longer overwrites it).
        # justify_content='center' — centres figure+controls vertically in the
        #   column, which is stretched to the height of the tall file list.
        self.v_image_layout = VBox(
            children=[self.figure, self.h_user_layout],
            layout=widgets.Layout(display='flex', flex_flow='column',
                                  align_items='center',
                                  justify_content='center',
                                  overflow='auto'))
        self._build_main_layout(self.v_file_layout, self.v_image_layout, 10)
        self._code_out = widgets.Output(
            layout=widgets.Layout(width='100%', display='none'))
        self.h_main_layout.children = tuple(self.h_main_layout.children) + (self._code_out,)

    def _connect_observers(self) -> None:
        """Wire all observe() and on_click() callbacks."""
        # Tier 1 — cosmetic only (I2): patch colorscale/limits/labels in-place.
        # _update_display never re-ships z/x/y over the comm.
        for w in (self.labelToggle,
                  self.upperLeftSelect, self.upperRightSelect,
                  self.lowerLeftSelect, self.lowerRightSelect,
                  self.labelColorSelect, self.labelFontSize, self.titleFontSize,
                  self.reverseScaleToggle, self.cmapSelection):
            w.observe(self._update_display, names='value')

        # Tier 2 — rebuild title text then redraw (no filter pipeline)
        # titleToggle lives here so update_scan_info runs before update_axes reads scan_info
        for w in (self.titleToggle, self.channelToggle, self.setpointToggle,
                  self.feedbackToggle, self.locationToggle, self.depthSelection,
                  self.nameToggle, self.directionToggle, self.dateToggle):
            w.observe(self._refresh_info, names='value')

        # Tier 3 — full pipeline (filter/channel processing); debounced (I1)
        for w in (self.gaussianToggle, self.gaussianSize,
                  self.medianToggle,   self.medianSize,
                  self.laplacToggle,   self.laplaceSize,
                  self.linebylineBtn, self.flattenBtn,
                  self.invertBtn, self.fixZeroBtn):
            w.observe(self._schedule_redraw_dirty, names='value')

        # Square size budget: keep W = H so _compute_fig_size yields a square
        # plot for square scans (non-square scans still fit by aspect ratio).
        # Mutual sync self-terminates: ipywidgets fires only on actual change.
        self.figWidth.value = self.figHeight.value
        self.figWidth.observe(
            lambda ch: setattr(self.figHeight, 'value', ch['new']), names='value')
        self.figHeight.observe(
            lambda ch: setattr(self.figWidth, 'value', ch['new']), names='value')

        self._connect_figure_settings_observers()
        self.copyBtn.on_click(self.copy_figure)
        self.codeBtn.on_click(self._export_code_snippet)
        self.headerKeySelect.observe(self._on_header_key_select, names='value')
        self.contextToggle.observe(self._on_context_toggle, names='value')
        self.contextSelect.observe(self._on_context_select, names='value')
        self.figCtxLegendW.observe(self._on_ctx_legend_w_change, names='value')
        self.configOptionBtn.observe(self.handler_configOptionsDisplay, names='value')
        self.directionBtn.observe(self.update_scan_direction, names='value')
        self.vmin.observe(self._on_limits_change, names='value')
        self.vmax.observe(self._on_limits_change, names='value')

        self.nextBtn.on_click(self.nextDisplay)
        self.previousBtn.on_click(self.previousDisplay)
        self.refreshBtn.on_click(self.handler_root_folder_update)
        self.directorySelection.observe(self.handler_folder_selection, names='value')
        self.selectionList.observe(self.handler_file_selection, names='value')
        self.channelSelect.observe(self.handler_channel_selection, names='value')
        self.rootFolder.observe(self.handler_root_folder_update, names='value')

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display(self) -> None:
        """Render the browser widget."""
        self._set_settings_visibility(True)
        display.clear_output(wait=True)
        display.display(self.h_main_layout)

    def _redraw(self, *_) -> None:
        """Update axes; FigureWidget updates reactively — no canvas.draw() needed."""
        self._set_busy(True, 'Rendering...')
        try:
            self.update_axes()
        finally:
            self._set_busy(False)

    def _on_limits_change(self, _) -> None:
        """Callback for vmin/vmax changes; skipped during batch limit updates.

        vmin/vmax are cosmetic (zmin/zmax on the existing trace), so patch
        in-place via _update_display rather than re-shipping the heatmap (I2).
        """
        if not self._updating_limits:
            self._update_display()

    def _update_display(self, *_) -> None:
        """Patch cosmetic props on the existing heatmap in-place (I2).

        Handles colormap, reverse, vmin/vmax, colorbar ticks, axis-label
        visibility and corner-label annotations without re-sending z/x/y.
        Falls back to a full _redraw when no image is loaded yet.
        """
        if self._loading:
            return
        if self.img is None or not self.figure.data:
            self._redraw()
            return
        vmin, vmax = self.vmin.value, self.vmax.value
        channel = self.channelSelect.value
        unit    = self.image_info['unit']
        h_crop, w_crop = self._last_crop
        with self.figure.batch_update():
            hm = self.figure.data[0]
            hm.zmin       = vmin
            hm.zmax       = vmax
            hm.colorscale = self._resolve_colorscale(self.cmapSelection.value)
            hm.reversescale = self.reverseScaleToggle.value
            hm.colorbar = dict(
                title=dict(text=f'{channel} ({unit})', side='right'),
                thickness=12, len=1.0, lenmode='fraction',
                y=0.5, yanchor='middle',
                tickvals=[vmin, vmax],
                ticktext=[f'{vmin:.2f}', f'{vmax:.2f}'],
            )
            show_axes = not self.labelToggle.value
            self.figure.layout.xaxis.visible = show_axes
            self.figure.layout.yaxis.visible = show_axes
        # Corner labels patched separately (also in-place via batch_update)
        if self.labelToggle.value:
            self._add_figure_labels(w_crop, h_crop)
        else:
            with self.figure.batch_update():
                self.figure.layout.annotations = []
                self.figure.layout.shapes      = []

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _figure_stem(self, dir_name: str) -> str:
        return f'{dir_name}_{self.img.name.split(".")[0]}_{self.channelSelect.value}'

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------

    def redraw_image(self, a) -> None:
        """Reload and replot when a processing toggle changes."""
        if self.img is not None:
            self.update_image_data()
            self._redraw()

    def load_new_image(self) -> None:
        """Load the currently selected SXM file and populate channel list."""
        directory = self.directories[self.directorySelection.index]
        if directory != self.active_dir:
            directory = self.active_dir / directory
        self.img = Spm(str(directory / self.sxm_files[self.image_index]))
        self._scan_cache = {}
        self._context_spm_cache = {}
        self._clear_context_traces()
        self.filenameText.value = self.sxm_files[self.image_index]  # F3: sxm_files not all_files
        self._loading = True
        self.contextSelect.value = ()      # reset selection; observer suppressed
        self._update_channel_selection()   # channel observer suppressed
        self._populate_header_inspector()  # Concern 4: header key list
        self._loading = False
        self._update_context_list()        # Concern 4: refresh time context
        self.update_image_data()           # runs exactly once

    def _cache_scan_param(self, key: str, default=('N/A', '')):
        """Return img.get_param(key), caching the result until next file load.

        Returns *default* when get_param returns None (missing header key) so
        callers can safely index result[0]/result[1] without a TypeError.
        """
        if key not in self._scan_cache:
            val = self.img.get_param(key)
            self._scan_cache[key] = val if val is not None else default
        return self._scan_cache[key]

    def update_image_data(self) -> None:
        """Process channel data from widget state and update image_data."""
        if self.img is None:
            return
        channel   = self.channelSelect.value
        direction = 'backward' if self.directionBtn.value else 'forward'
        try:
            self.image_data, unit = self.img.get_channel(
                channel, direction=direction, flatten=self.flattenBtn.value,
                offset=False, zero=False)
        except Exception:
            self.updateErrorText('Error in flattening: setting flatten=False')
            self.image_data, unit = self.img.get_channel(
                channel, direction=direction, flatten=False, offset=False, zero=False)
        if direction == 'backward':
            self.image_data = np.flip(self.image_data, axis=1)
        self.image_info['unit'] = unit
        if self.invertBtn.value:
            self.image_data *= -1
        if self.linebylineBtn.value:
            self.image_data = remove_line_average(self.image_data)
        if self.fixZeroBtn.value:
            self.image_data -= np.nanmin(self.image_data)
        if self.gaussianToggle.value:
            self.image_data = gaussian_filter(self.image_data, self.gaussianSize.value)
        if self.medianToggle.value:
            self.image_data = median_filter(self.image_data, size=self.medianSize.value)
        if self.laplacToggle.value:
            self.image_data = -gaussian_laplace(self.image_data, self.laplaceSize.value)
        # update limit widgets without triggering a redundant redraw
        self._updating_limits = True
        self.vmin.value = round(float(np.nanmin(self.image_data)), 3)
        self.vmax.value = round(float(np.nanmax(self.image_data)), 3)
        self._updating_limits = False
        try:
            self.update_scan_info()
        except Exception as err:
            # Bad/missing header keys must not prevent the image from rendering
            self.updateErrorText(f'Header read error: {err}')

    def update_scan_info(self) -> None:
        """Build scan_info title string and populate scan_dict from cached params."""
        if self.img is None:
            return
        label = []
        if self.img.type == 'scan':
            fb_enable = self._cache_scan_param('z-controller>controller status')
            bias      = self._cache_scan_param('V')
            set_point = self._cache_scan_param('setpoint')
            height    = self._cache_scan_param('height')
            width     = self._cache_scan_param('width')
            angle     = self._cache_scan_param('angle')
            z_offset  = self._cache_scan_param('z_offset')
            comment   = self._cache_scan_param('comments')

            self.image_info['width']  = width[0]
            self.image_info['height'] = height[0]
            self.scan_dict.update({
                'feedback': fb_enable, 'setpoint': set_point,
                'size': (width, height, angle),
                'z_offset': z_offset, 'comment': comment,
            })
            mode = ('Constant Height → z-offset: %.3f%s' % z_offset
                    if fb_enable == 'OFF' else 'Constant Current')
            if abs(bias[0]) < 0.1:
                bias = (bias[0] * 1000, 'mV')
            self.scan_dict['bias'] = bias

            if self.channelToggle.value:
                ch_str = f'channel: {self.channelSelect.value}'
                label.append(f'{ch_str} → {mode}'
                              if self.feedbackToggle.value else ch_str)
            if self.setpointToggle.value:
                label.append(f'setpoint: I = {set_point[0]:.0f}{set_point[1]}, '
                              f'V = {bias[0]:.2f}{bias[1]}')
            if self.locationToggle.value:
                loc = self.directories[self.directorySelection.index]
                if self.depthSelection.value != 'full':
                    loc = '\\'.join(str(loc).split('\\')[-int(self.depthSelection.value):])
                label.append(f'location: {loc}')
            if self.nameToggle.value:
                name_str = f'filename: {self.img.name}'
                if self.directionToggle.value:
                    d = 'backward' if self.directionBtn.value else 'forward'
                    name_str += f' → direction: {d}'
                label.append(name_str)
            self.scan_dict['filename'] = self.img.name
            if self.dateToggle.value:
                rec_date = self.img.header.get('rec_date', 'N/A')
                rec_time = self.img.header.get('rec_time', '')
                label.append(f'Date: {rec_date} {rec_time}'.strip())
            self.scan_dict['date'] = (f'{self.img.header.get("rec_date", "N/A")} '
                                      f'{self.img.header.get("rec_time", "")}').strip()

        self.scan_info = '<br>'.join(label) if self.titleToggle.value else ''

    def update_axes(self) -> None:
        """Render image_data onto the FigureWidget via batch_update."""
        if self.img is None:
            return
        scan_dir = self._cache_scan_param('scan_dir')
        height   = self.image_info['height']
        width    = self.image_info['width']
        # strip NaN rows introduced by partially-completed scans
        data     = self.image_data[~np.isnan(self.image_data).any(axis=1)]
        row, col = self.image_data.shape
        y_px, x_px = data.shape
        w_crop = width  * x_px / col
        h_crop = height * y_px / row

        # Plotly Heatmap: first z-row → bottom of plot by default.
        # scan_dir='down' matches matplotlib origin='upper' → flip z so the
        # first data row appears at the top (y increases downward in scan coords).
        z = data[::-1] if scan_dir == 'down' else data

        vmin = self.vmin.value
        vmax = self.vmax.value
        channel = self.channelSelect.value
        unit    = self.image_info['unit']

        with self.figure.batch_update():
            hm = self.figure.data[0]
            hm.z          = z
            hm.x          = np.linspace(0, w_crop, x_px)
            hm.y          = np.linspace(0, h_crop, y_px)
            hm.zmin       = vmin
            hm.zmax       = vmax
            hm.colorscale = self._resolve_colorscale(self.cmapSelection.value)
            hm.reversescale = self.reverseScaleToggle.value
            hm.colorbar = dict(
                title=dict(text=f'{channel} ({unit})', side='right'),
                thickness=12,
                len=1.0, lenmode='fraction',
                y=0.5, yanchor='middle',
                tickvals=[vmin, vmax],
                ticktext=[f'{vmin:.2f}', f'{vmax:.2f}'],
            )
            self._last_crop = (h_crop, w_crop)
            show_axes = not self.labelToggle.value
            self.figure.update_layout(
                xaxis=dict(title='x (nm)', tickvals=[0, w_crop],
                           ticktext=['0', f'{w_crop:.2f}'],
                           visible=show_axes),
                yaxis=dict(title='y (nm)', tickvals=[0, h_crop],
                           ticktext=['0', f'{h_crop:.2f}'],
                           visible=show_axes),
                annotations=[], shapes=[],
            )

        self._update_fig_width(h_crop, w_crop)
        self._apply_figure_title()
        if self.labelToggle.value:
            self._add_figure_labels(w_crop, h_crop)

    # ------------------------------------------------------------------

    def _compute_fig_size(self, h_crop: float, w_crop: float) -> tuple[int, int]:
        """Return (width_px, height_px) that fits the image aspect ratio within
        the figWidth × figHeight budget.  Falls back to a square if crop is zero.
        Minimum plot area is 200 × 200 px regardless of widget values or aspect ratio."""
        L, R, T, B = 60, 60 + self._CTX_LEGEND_W, 80, 60
        MIN_PLOT = 200
        max_plot_h = max(MIN_PLOT, self.figHeight.value - T - B)
        max_plot_w = max(MIN_PLOT, self.figWidth.value  - L - R)
        if h_crop <= 0 or w_crop <= 0:
            return L + max_plot_h + R, T + max_plot_h + B  # square fallback
        plot_h = max_plot_h
        plot_w = max(MIN_PLOT, int(plot_h * w_crop / h_crop))
        if plot_w > max_plot_w:                 # wide image: constrain by width
            plot_w = max_plot_w
            plot_h = max(MIN_PLOT, int(plot_w * h_crop / w_crop))
        return L + plot_w + R, T + plot_h + B

    def _update_fig_width(self, h_crop: float, w_crop: float) -> None:
        """Resize the figure to match the image aspect ratio (called on each image load)."""
        w_px, h_px = self._compute_fig_size(h_crop, w_crop)
        self.figure.update_layout(autosize=False, width=w_px, height=h_px)

    def _on_ctx_legend_w_change(self, change) -> None:
        """Resize the reserved right margin and re-render any active context markers.

        Guarded by _loading: template application sets figCtxLegendW.value while
        suppressing observers, so skip the layout cascade during those batch
        updates (otherwise it fires before the image/context list is ready).
        """
        if self._loading:
            return
        self._CTX_LEGEND_W = change['new']
        self._apply_figure_layout()
        if self.contextToggle.value and self.contextSelect.value:
            self._render_context_markers(self.contextSelect.value)

    def _apply_figure_layout(self, _=None) -> None:
        """'Apply Settings' callback: compute aspect-ratio size then apply all settings."""
        h_crop, w_crop = self._last_crop
        w_px, h_px = self._compute_fig_size(h_crop, w_crop)
        self._figure_layout_update(
            margin=dict(l=60, r=60 + self._CTX_LEGEND_W, t=80, b=60),
            autosize=False, width=w_px, height=h_px,
        )

    def _add_figure_labels(self, w: float, h: float) -> None:
        """Overlay corner text / scalebar annotations via plotly paper-coords."""
        color = self.labelColorSelect.value
        fs    = self.labelFontSize.value
        # Paper-fraction corners: xf, yf ∈ [0,1] map to axes area edges
        corners = {
            'upper left':  (self.upperLeftSelect.value,  0.03, 0.97),
            'upper right': (self.upperRightSelect.value, 0.97, 0.97),
            'lower left':  (self.lowerLeftSelect.value,  0.03, 0.03),
            'lower right': (self.lowerRightSelect.value, 0.97, 0.03),
        }
        text_map = {
            'channel':  str(self.channelSelect.value),
            'bias':     '{}{}'.format(*self.scan_dict.get('bias', ('N/A', ''))),
            'setpoint': '{} {}'.format(*self.scan_dict.get('setpoint', ('N/A', ''))),
            'feedback': str(self.scan_dict.get('feedback', 'N/A')),
            'date':     str(self.scan_dict.get('date', 'N/A')),
            'filename': str(self.scan_dict.get('filename', 'N/A')),
            'filters':  '+'.join(k for k, t in [('G', self.gaussianToggle),
                                                  ('M', self.medianToggle),
                                                  ('LP', self.laplacToggle)] if t.value),
        }
        annotations, shapes = [], []
        iw = self.image_info['width']

        for position, (selection, xf, yf) in corners.items():
            if selection == 'none':
                continue
            ha, va = self._ALIGN_PARAMS[position]
            xanchor = ha           # 'left' | 'right'
            yanchor = va           # 'top'  | 'bottom'

            if selection == 'scalebar':
                standards = [.1, .2, .5, 1, 2, 5, 10, 20, 50,
                             100, 200, 500, 1000, 2000, 5000]
                sb_len = min(standards, key=lambda x: abs(x - iw * 0.20))
                label  = f'{sb_len * 10:.0f} Å' if sb_len < 1 else f'{sb_len:.0f} nm'
                frac   = sb_len / iw  # bar length in paper-fraction of image width

                # Horizontal bar position in paper coords
                if ha == 'right':
                    bar_x1, bar_x0 = xf, xf - frac
                else:
                    bar_x0, bar_x1 = xf, xf + frac
                # Vertical position of bar and label (label always above bar)
                bar_y  = 0.08 if va == 'bottom' else 0.90
                lbl_y  = bar_y + 0.04

                shapes.append(dict(
                    type='line', x0=bar_x0, x1=bar_x1, y0=bar_y, y1=bar_y,
                    xref='x domain', yref='y domain',
                    line=dict(color=color, width=3)))
                annotations.append(dict(
                    x=(bar_x0 + bar_x1) / 2, y=lbl_y, text=label,
                    xref='x domain', yref='y domain',
                    xanchor='center', yanchor='bottom', showarrow=False,
                    font=dict(size=fs, color=color)))
                continue

            annotations.append(dict(
                x=xf, y=yf, text=text_map.get(selection, ''),
                xref='x domain', yref='y domain',
                xanchor=xanchor, yanchor=yanchor,
                showarrow=False, font=dict(size=fs, color=color)))

        with self.figure.batch_update():
            self.figure.layout.annotations = annotations
            self.figure.layout.shapes      = shapes

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _apply_figure_title(self) -> None:
        """Set figure title text and font; automargin reserves the space (I5).

        Anchored top-left at y=0.925 in container coords; automargin=True lets
        plotly size the top margin to the title so no hand-tuned pixel margin
        is needed (mirrors DAT's approach for consistent whitespace).
        """
        self.figure.update_layout(title=dict(
            text=self.scan_info, font=dict(size=self.figTitleSize.value),
            x=0, y=0.925, yref='container', xref='paper',
            xanchor='left', yanchor='bottom', automargin=True))


    def _update_channel_selection(self) -> None:
        current = self.channelSelect.value
        self.channelSelect.options = self.img.channels
        self.channelSelect.value = (current if current in self.img.channels
                                    else self.img.channels[0])

    def _update_info_text(self) -> None:
        self.filenameText.value  = self.sxm_files[self.image_index]   # F3: sxm_files not all_files
        self.selectionList.value = self.sxm_files[self.image_index]

    def nextDisplay(self, a) -> None:
        if self.image_index < len(self.sxm_files) - 1:  # F4: bound on sxm_files, not all_files
            self.image_index += 1
            self._loading = True
            self._update_info_text()   # sync UI; handler_file_selection suppressed
            self._loading = False
            try:
                self.load_new_image()
                self._redraw()
            except Exception as err:
                self.updateErrorText('navigation error: ' + str(err))

    def previousDisplay(self, a) -> None:
        if self.image_index > 0:
            self.image_index -= 1
            self._loading = True
            self._update_info_text()
            self._loading = False
            try:
                self.load_new_image()
                self._redraw()
            except Exception as err:
                self.updateErrorText('navigation error: ' + str(err))

    def update_scan_direction(self, a) -> None:
        self.directionBtn.icon = ('caret-square-o-left' if self.directionBtn.value
                                  else 'caret-square-o-right')
        self.redraw_image(a)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def get_plot_data(self):
        """Return the processed image array currently displayed in the figure."""
        return self.image_data.copy()

    def get_file_selection(self) -> list:
        """Return the currently loaded SXM filename (list of one, or empty)."""
        if not self.sxm_files:
            return []
        return [self.sxm_files[self.image_index]]

    def get_parameter(self, key: str):
        """Return a scan parameter by key, or None if unavailable."""
        if self.img is None:
            return None
        return self._cache_scan_param(key)

    def get_channel(self, channel: str, direction: str = 'forward'):
        """Return raw channel data array, or None if unavailable."""
        if self.img is None:
            return None
        return self.img.get_channel(channel, direction=direction)[0]

    # ------------------------------------------------------------------
    # Code export (N15)
    # ------------------------------------------------------------------

    def _build_export_code(self) -> str:
        """Return a self-contained Python snippet that reproduces the current plot."""
        if not self.sxm_files:
            return '# No file loaded'
        fname = self.sxm_files[self.image_index]
        channel = self.channelSelect.value or 'Z'
        direction = 'backward' if self.directionBtn.value else 'forward'
        lbl = self.image_info
        unit = lbl.get('unit', 'nm')
        width  = lbl.get('width', 1)
        height = lbl.get('height', 1)
        cmap   = self._resolve_colorscale(self.cmapSelection.value)
        if isinstance(cmap, list):
            cmap_repr = repr(cmap)
        else:
            cmap_repr = repr(str(cmap))
        vmin_val = self.vmin.value
        vmax_val = self.vmax.value
        lines = [
            "import numpy as np",
            "import plotly.graph_objects as go",
            "from spmpy import Spm",
            "from simpleNFB.process_utils import remove_line_average",
            "",
            f"img = Spm({fname!r})",
            f"data, _ = img.get_channel({channel!r}, direction={direction!r})",
            "",
            "# Apply corrections (match browser state)",
        ]
        if self.linebylineBtn.value:
            lines.append("data = remove_line_average(data)")
        if self.flattenBtn.value:
            lines.append("# plane-fit subtraction (apply np.polyfit per row/col as needed)")
        if self.invertBtn.value:
            lines.append("data = -data")
        if self.fixZeroBtn.value:
            lines.append("data = data - np.nanmin(data)")
        lines += [
            "",
            "# Crop incomplete scan rows",
            "data = data[~np.isnan(data).any(axis=1)]",
            "",
            "import matplotlib.pyplot as plt",
            f"fig, ax = plt.subplots(figsize=({self._fig_width / 100:.1f}, {self._fig_height / 100:.1f}))",
            f"extent = [0, {width}, 0, {height}]",
            f"im = ax.imshow(data, origin='upper', extent=extent,",
            f"               cmap={cmap_repr}, vmin={vmin_val}, vmax={vmax_val})",
            f"fig.colorbar(im, label='{channel} ({unit})')",
            f"ax.set_xlabel('x ({unit})')",
            f"ax.set_ylabel('y ({unit})')",
            "plt.tight_layout()",
            "plt.show()",
        ]
        return '\n'.join(lines)

    def _export_code_snippet(self, _event=None) -> None:
        """Export reproducible code: auto-copy to clipboard and show panel in browser."""
        code = self._build_export_code()

        try:
            import win32clipboard
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(code, win32clipboard.CF_UNICODETEXT)
            win32clipboard.CloseClipboard()
            status_html = ('<span style="font-size:11px;color:#2a9d2a;">'
                           '<b>Copied to clipboard.</b> Paste into a new cell to run.</span>')
        except ImportError:
            status_html = ('<span style="font-size:11px;color:#888;">'
                           'pywin32 not installed &mdash; select all &amp; copy '
                           '(Ctrl+A, Ctrl+C) below.</span>')
        except Exception as e:
            status_html = f'<span style="font-size:11px;color:#c0392b;">Clipboard error: {e}</span>'

        status    = widgets.HTML(value=status_html)
        close_btn = widgets.Button(
            description='Close',
            layout=widgets.Layout(width='60px', height='26px'))
        code_area = widgets.Textarea(
            value=code,
            layout=widgets.Layout(width='100%', height='220px',
                                  font_family='monospace', font_size='11px'),
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

        self._inject_cell(code)

    def handler_configOptionsDisplay(self, a) -> None:
        self._set_settings_visibility(self.configOptionBtn.value)
        if self.configOptionBtn.value:
            self.depthSelection.layout.visibility = 'visible'

    # ------------------------------------------------------------------
    # Header inspector + session timeline context (Concern 4)
    # ------------------------------------------------------------------

    def _populate_header_inspector(self) -> None:
        """Fill the header key Select from the loaded image's header."""
        if self.img is None:
            return
        keys = list(self.img.header.keys())
        self.headerKeySelect.options = keys
        if keys:
            self.headerKeySelect.value = keys[0]
            self.headerValueText.value = str(self.img.header[keys[0]])

    def _on_header_key_select(self, change) -> None:
        """Show the selected header key's value in the read-only Textarea."""
        if self.img is None or change['new'] is None:
            return
        self.headerValueText.value = str(self.img.header.get(change['new'], ''))

    def _on_context_toggle(self, change) -> None:
        """Show/hide the .dat context list; clear markers when hidden."""
        self.contextSelect.layout.display = 'flex' if change['new'] else 'none'
        if change['new']:
            self._update_context_list()
        else:
            self._clear_context_traces()
            self._loading = True
            self.contextSelect.value = ()
            self._loading = False

    def _update_context_list(self) -> None:
        """List .dat files whose mtime falls in [this image, next image).

        Uses session_index mtimes captured during the folder scan — no extra
        file reads. Newer images are first in sxm_files, so the *next* image in
        time is the previous list entry.
        """
        if not self.contextToggle.value or not self.session_index:
            return
        sxm_times = {n: m for (n, k, m) in self.session_index if k == 'sxm'}
        cur_name = (self.sxm_files[self.image_index]
                    if 0 <= self.image_index < len(self.sxm_files) else None)
        if cur_name is None or cur_name not in sxm_times:
            self.contextSelect.options = []
            return
        cur_mt = sxm_times[cur_name]
        # Next image in time = smallest sxm mtime strictly greater than cur_mt.
        later = [m for m in sxm_times.values() if m > cur_mt]
        next_mt = min(later) if later else float('inf')
        dats = sorted((m, n) for (n, k, m) in self.session_index
                      if k == 'dat' and cur_mt <= m < next_mt)
        self._loading = True
        self.contextSelect.options = [n for _, n in dats]
        self._loading = False

    def _clear_context_traces(self) -> None:
        """Remove context-marker scatter traces (keep base Heatmap at index 0)
        and strip the paper-coord annotation legend, preserving corner labels."""
        if len(self.figure.data) > 1:
            self.figure.data = self.figure.data[:1]
        # Drop only our legend annotations (tagged name='ctx_legend'); keep others.
        kept = tuple(
            a for a in (self.figure.layout.annotations or ())
            if getattr(a, 'name', None) != 'ctx_legend'
        )
        self.figure.update_layout(annotations=kept)

    def _render_context_markers(self, selected_dats) -> None:
        """Plot one Scatter marker per selected .dat with a paper-coord annotation
        legend mapping filename → colour.

        A trace legend (showlegend=True) is intentionally NOT used: plotly docs
        confirm it grows the plot margins to fit the legend box, which shrinks the
        plot area and breaks the heatmap aspect ratio.  Paper-anchored annotations
        draw on top of the fixed plot area without any margin reflow.
        Positions are cached in _context_spm_cache to avoid re-reading from disk.
        """
        self._clear_context_traces()
        if not selected_dats or self.img is None:
            return
        directory = self.directories[self.directorySelection.index]
        if directory != self.active_dir:
            directory = self.active_dir / directory
        colors = px.colors.qualitative.Plotly
        traces = []
        legend_entries = []   # (color, filename) in draw order
        for i, dat in enumerate(selected_dats):
            if dat not in self._context_spm_cache:
                try:
                    self._context_spm_cache[dat] = Spm(str(directory / dat))
                except Exception as err:
                    self.updateErrorText(f'context load error ({dat}): {err}')
                    continue
            spec = self._context_spm_cache[dat]
            try:
                rx, ry = relative_position(self.img, spec)
            except Exception as err:
                self.updateErrorText(f'context position error ({dat}): {err}')
                continue
            color = colors[i % len(colors)]
            traces.append(go.Scatter(
                x=[rx], y=[ry], mode='markers',
                name=dat, showlegend=False,
                marker=dict(symbol='x', size=12, color=color, line_width=2)))
            legend_entries.append((color, dat))
        if not traces:
            return
        self.figure.add_traces(traces)
        # Compute annotation x past the colorbar right edge + 20 px clearance.
        # Colorbar defaults: x=1.02 paper (left-anchored), thickness=12 px set
        # explicitly, xpad=10 px each side (Plotly default).
        # Right edge of colorbar in paper = 1.02 + (thickness + 2*xpad) / plot_w.
        w_px, _ = self._compute_fig_size(*self._last_crop)
        plot_w = max(1, w_px - 60 - (60 + self._CTX_LEGEND_W))
        x_ann = 1.02 + (12 + 2 * 10 + 20) / plot_w   # +52 px: cb width + gap
        row_dy = 0.045   # vertical spacing between entries (paper fraction)
        y0     = 0.985   # top entry y (just inside upper edge)
        new_anns = tuple(
            dict(name='ctx_legend', xref='paper', yref='paper',
                 x=x_ann, y=y0 - idx * row_dy,
                 xanchor='left', yanchor='top', showarrow=False,
                 align='left',
                 text=f'<span style="color:{color}">✕</span> {name}',
                 font=dict(size=self.fontsize, color='black'),
                 bgcolor='rgba(255,255,255,0.6)', borderpad=1)
            for idx, (color, name) in enumerate(legend_entries)
        )
        existing = tuple(self.figure.layout.annotations or ())
        self.figure.update_layout(annotations=existing + new_anns)

    def _on_context_select(self, change) -> None:
        """Render markers for the selected .dat file(s) on this SXM image."""
        if self._loading:
            return
        self._render_context_markers(change['new'])

    def handler_folder_selection(self, a) -> None:
        self._set_busy(True, 'Scanning folder...')
        try:
            index = 0
            # I4: no refreshBtn branch — refreshBtn is wired to
            # handler_root_folder_update, never to this handler.
            directory = self.directories[self.directorySelection.index]
            # I3: scandir + endswith (no false matches like .sxm.bak); mtime sort
            self.sxm_files, self.dat_files = [], []
            sxm_mt, dat_mt = [], []
            with os.scandir(directory) as it:
                for entry in it:
                    if entry.name.endswith('.sxm'):
                        sxm_mt.append((entry.stat().st_mtime, entry.name))
                    elif entry.name.endswith('.dat'):
                        dat_mt.append((entry.stat().st_mtime, entry.name))
            sxm_mt.sort(); dat_mt.sort()
            # F3: reverse sxm (newest first) BEFORE building all_files
            self.sxm_files = [n for _, n in reversed(sxm_mt)]
            self.dat_files = [n for _, n in dat_mt]
            self.all_files = self.sxm_files + self.dat_files
            # Concern 4: session timeline index (free — mtimes already collected)
            self.session_index = ([(n, 'sxm', m) for m, n in sxm_mt]
                                  + [(n, 'dat', m) for m, n in dat_mt])
            # suppress handler_file_selection while syncing list widget state
            self._loading = True
            self.selectionList.options = self.sxm_files
            if self.sxm_files:
                self.filenameText.value = self.sxm_files[index]
                self.selectionList.value = self.sxm_files[index]
            self._loading = False
            if self.sxm_files:
                self.image_index = index
                try:
                    self.load_new_image()
                    self._redraw()
                except Exception as err:
                    self.updateErrorText('folder selection error: ' + str(err))
        finally:
            self._set_busy(False)

    def handler_file_selection(self, update) -> None:
        if self._loading or self.selectionList.value is None:
            return
        self._set_busy(True, 'Loading image...')
        try:
            self.image_index = self.sxm_files.index(self.selectionList.value)
            self.load_new_image()
            self._redraw()
        except Exception as err:
            self.updateErrorText('file selection error: ' + str(err))
            print(traceback.format_exc())
        finally:
            self._set_busy(False)


    def _template_extra_save(self) -> dict:
        """Capture SXM label settings and context legend width alongside the plotly template."""
        return {
            'labels': {
                'show':        self.labelToggle.value,
                'upper_left':  self.upperLeftSelect.value,
                'upper_right': self.upperRightSelect.value,
                'lower_left':  self.lowerLeftSelect.value,
                'lower_right': self.lowerRightSelect.value,
                'color':       self.labelColorSelect.value,
                'font_size':   self.labelFontSize.value,
            },
            'ctx_legend_w': self._CTX_LEGEND_W,
        }

    def _template_extra_apply(self, entry: dict) -> None:
        """Restore SXM label settings and context legend width from a saved template entry."""
        labels = entry.get('labels')
        if labels:
            self._loading = True
            try:
                self.labelToggle.value      = labels['show']
                self.upperLeftSelect.value  = labels['upper_left']
                self.upperRightSelect.value = labels['upper_right']
                self.lowerLeftSelect.value  = labels['lower_left']
                self.lowerRightSelect.value = labels['lower_right']
                self.labelColorSelect.value = labels['color']
                self.labelFontSize.value    = labels['font_size']
            except Exception:
                pass
            finally:
                self._loading = False
        if 'ctx_legend_w' in entry:
            # Set the widget under _loading suppression so _on_ctx_legend_w_change
            # does NOT run its layout cascade mid-template-apply (that ran before
            # the image/context list was ready and left the context list empty).
            # Sync the reserved margin and apply the layout once, explicitly.
            self._loading = True
            try:
                self.figCtxLegendW.value = entry['ctx_legend_w']
            finally:
                self._loading = False
            self._CTX_LEGEND_W = entry['ctx_legend_w']
            self._apply_figure_layout()

    def handler_channel_selection(self, update) -> None:
        if self._loading:
            return
        try:
            self.update_image_data()
            self._redraw()
        except Exception as err:
            self.updateErrorText('channel selection error: ' + str(err))


# Backward-compatible alias so __init__.py can `from .SXM_browser import imageBrowser`
imageBrowser = fileBrowser
