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
from .process_utils import remove_line_average
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
        self.saveBtn = Btn_Widget(
            '', layout=FLB(24), icon='file-image-o',
            tooltip='Save displayed image to \\browser_output folder')
        self.copyBtn = Btn_Widget(
            '', layout=FLB(24), icon='clipboard',
            tooltip='Save and copy displayed image to clipboard')

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
            VBox(children=[
                HBox(children=[self.refreshBtn, self.saveBtn, self.copyBtn, self.configOptionBtn]),
                widgets.Label('Note')]),
            self.saveNote],
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
            self._figure_settings_tab(),
        ], layout=FLH(98),
        titles=['Label Settings', 'Title Settings', 'Filter Settings', 'Figure Settings'])
        # FigureWidget is itself a widget — no Output wrapper needed.
        # align_items/justify_content centre the fixed-width figure in the flex column.
        # Column layout: figure on top, controls below, each centred horizontally.
        # align_items='center'  — centres each child on the cross (horizontal) axis
        #                         at its own natural width; no child is forced to fill.
        # justify_content='flex-start' — stacks children from the top so the figure
        #                         sits immediately above the controls with no gap.
        self.v_image_layout = VBox(
            children=[self.figure, self.h_user_layout],
            layout=widgets.Layout(display='flex', flex_flow='column',
                                  align_items='center',
                                  justify_content='flex-start',
                                  overflow='auto'))
        self._build_main_layout(self.v_file_layout, self.v_image_layout, 10)

    def _connect_observers(self) -> None:
        """Wire all observe() and on_click() callbacks."""
        # Tier 1 — layout update only (colormap, labels, fonts; no data reprocessing)
        for w in (self.labelToggle,
                  self.upperLeftSelect, self.upperRightSelect,
                  self.lowerLeftSelect, self.lowerRightSelect,
                  self.labelColorSelect, self.labelFontSize, self.titleFontSize,
                  self.reverseScaleToggle, self.cmapSelection):
            w.observe(self._redraw, names='value')

        # Tier 2 — rebuild title text then redraw (no filter pipeline)
        # titleToggle lives here so update_scan_info runs before update_axes reads scan_info
        for w in (self.titleToggle, self.channelToggle, self.setpointToggle,
                  self.feedbackToggle, self.locationToggle, self.depthSelection,
                  self.nameToggle, self.directionToggle, self.dateToggle):
            w.observe(self._refresh_info, names='value')

        # Tier 3 — full pipeline (filter/channel processing required)
        for w in (self.gaussianToggle, self.gaussianSize,
                  self.medianToggle,   self.medianSize,
                  self.laplacToggle,   self.laplaceSize,
                  self.linebylineBtn, self.flattenBtn,
                  self.invertBtn, self.fixZeroBtn):
            w.observe(self.redraw_image, names='value')

        self._connect_figure_settings_observers()
        self.saveBtn.on_click(self.save_figure)
        self.copyBtn.on_click(self.copy_figure)
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
        """Callback for vmin/vmax changes; skipped during batch limit updates."""
        if not self._updating_limits:
            self._redraw()

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
        self.filenameText.value = self.all_files[self.image_index]
        self._loading = True
        self._update_channel_selection()   # channel observer suppressed
        self._loading = False
        self.update_image_data()           # runs exactly once

    def _cache_scan_param(self, key: str):
        """Return img.get_param(key), caching the result until next file load."""
        if key not in self._scan_cache:
            self._scan_cache[key] = self.img.get_param(key)
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
        self.update_scan_info()

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
                label.append(f'Date: {self.img.header["rec_date"]} {self.img.header["rec_time"]}')
            self.scan_dict['date'] = f'{self.img.header["rec_date"]} {self.img.header["rec_time"]}'

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
        L, R, T, B = 60, 60, 80, 60
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

    def _apply_figure_layout(self, _=None) -> None:
        """'Apply Settings' callback: compute aspect-ratio size then apply all settings."""
        h_crop, w_crop = self._last_crop
        w_px, h_px = self._compute_fig_size(h_crop, w_crop)
        self._figure_layout_update(
            margin=dict(l=60, r=60, t=80, b=60),
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
        """Set figure title text, font and top margin from current scan_info."""
        n_lines = (self.scan_info.count('<br>') + 1) if self.scan_info else 0
        t_margin = max(80, int(n_lines * self.titleFontSize.value * 2.5) + 40)
        self.figure.update_layout(
            title=dict(text=self.scan_info, font=dict(size=self.figTitleSize.value), x=0,y=0.925,yref='container',xref='paper',xanchor='left',yanchor='bottom',automargin=True),)
            #margin=dict(t=t_margin),
        

    def _update_channel_selection(self) -> None:
        current = self.channelSelect.value
        self.channelSelect.options = self.img.channels
        self.channelSelect.value = (current if current in self.img.channels
                                    else self.img.channels[0])

    def _update_info_text(self) -> None:
        self.filenameText.value  = self.all_files[self.image_index]
        self.selectionList.value = self.all_files[self.image_index]

    def nextDisplay(self, a) -> None:
        if self.image_index < len(self.all_files) - 1:
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

    def mouse_click(self, event) -> None:
        """Placeholder — linescan not yet implemented."""
        pass

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def get_plot_data(self):
        """Return the processed image array currently displayed in the figure."""
        return self.image_data.copy()

    def handler_configOptionsDisplay(self, a) -> None:
        self._set_settings_visibility(self.configOptionBtn.value)
        if self.configOptionBtn.value:
            self.depthSelection.layout.visibility = 'visible'

    def handler_folder_selection(self, a) -> None:
        self._set_busy(True, 'Scanning folder...')
        try:
            index = 0
            if type(a) == type(self.refreshBtn):
                index = self.selectionList.index
            directory = self.directories[self.directorySelection.index]
            self.sxm_files, self.dat_files = [], []
            for entry in os.listdir(directory):
                if '.sxm' in entry:
                    self.sxm_files.append(entry)
                elif '.dat' in entry:
                    self.dat_files.append(entry)
            self.all_files = self.sxm_files + self.dat_files
            self.sxm_files = list(np.flip(self.sxm_files))
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
        """Capture SXM label settings alongside the plotly template."""
        return {
            'labels': {
                'show':        self.labelToggle.value,
                'upper_left':  self.upperLeftSelect.value,
                'upper_right': self.upperRightSelect.value,
                'lower_left':  self.lowerLeftSelect.value,
                'lower_right': self.lowerRightSelect.value,
                'color':       self.labelColorSelect.value,
                'font_size':   self.labelFontSize.value,
            }
        }

    def _template_extra_apply(self, entry: dict) -> None:
        """Restore SXM label settings from a saved template entry."""
        labels = entry.get('labels')
        if not labels:
            return
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
