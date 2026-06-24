'''
SXM_browser.py
--------------
imageBrowser widget for Nanonis SXM scan data in Jupyter notebooks.
'''

import os
import traceback
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from matplotlib.font_manager import FontProperties as fm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.ndimage import gaussian_filter, gaussian_laplace, median_filter

from spmpy import Spm
from .base_browser import BaseBrowser
from .process_utils import remove_line_average
from .widget_helpers import HBox, VBox, Btn_Widget, Text_Widget, Selection_Widget


class fileBrowser(BaseBrowser):
    '''
    Interactive browser for Nanonis SXM image data.

    Public attributes:
        figure      – matplotlib Figure
        axes        – matplotlib Axes
        image_data  – ndarray of current processed image
        img         – Spm object for the loaded file

    Key methods:
        update_axes()       – refresh the plot from current image_data
        update_image_data() – reload and process channel data
        save_figure(a)      – save to browser_outputs/
    '''

    _ALIGN_PARAMS: dict = {
        'upper left':  ('left',  'top'),
        'upper right': ('right', 'top'),
        'lower left':  ('left',  'bottom'),
        'lower right': ('right', 'bottom'),
    }

    def __init__(self, figsize=(6, 6), fontsize: int = 12, titlesize: int = 12,
                 cmap: str = 'Greys_r', home_directory: str = './') -> None:
        # --- state ---
        self.img = None
        self.figure, self.axes = plt.subplots(ncols=1, num='sxm')
        self.figure.canvas.header_visible = False
        self.fontsize = fontsize
        self.font = fm(size=fontsize, family='sans-serif')
        self.titlesize = titlesize
        self.cb = None
        self.image_data = np.zeros((64, 64))
        self.image_info = {'height': 1, 'width': 1, 'unit': 'nm'}
        self.scan_dict: dict = {}
        self.scan_info = ''
        self.errors: list = []
        self.image_index = 0
        self._cmap = cmap
        self._updating_limits = False
        self._scan_cache: dict = {}

        self.active_dir = Path(home_directory)
        self.sxm_files: list = []
        self.dat_files: list = []
        self.directories = [self.active_dir]

        self._build_widgets()
        self._build_layout()
        self._connect_observers()
        self.display()
        with plt.ioff():
            with self.figure_display:
                self.figure_display.clear_output(wait=True)
                plt.show(self.figure)

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
        self._L = L

        # selections
        self.rootFolder = widgets.Text(
            description='', layout=widgets.Layout(display='flex', width='90%'))
        self.directorySelection = widgets.Select(
            description='', options=self.directories, rows=8, layout=FL(98))
        self.selectionList = widgets.Select(
            description='', options=self.sxm_files, rows=27, layout=FL(98))
        self.directoryDisplayDepth = widgets.Dropdown(
            description='depth', value=1, options=['full', 1, 2, 3, 4, 5],
            tooltip='Depth of folder structure shown in selection menu',
            layout=FLB(75), style={'description_width': '40px'})
        self.channelSelect = widgets.Dropdown(description='', layout=L(165))
        self.refreshBtn = Btn_Widget(
            '', icon='refresh', tooltip='Reload file list', layout=FLB(24))

        # text display
        self.filenameText = Text_Widget('')
        self.indexText    = Text_Widget('0')
        self.errorText    = Selection_Widget([], 'Out:', rows=5)
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
        self.figure_display = widgets.Output(layout=FLB(99))

        # colormap
        self.vmin = widgets.FloatText(
            value=0, description='Min:', step=.1,
            layout=FL(50), style={'description_width': '40px'})
        self.vmax = widgets.FloatText(
            value=1, description='Max:', step=.1,
            layout=FL(50), style={'description_width': '40px'})
        self.cmapSelection = widgets.Dropdown(
            description='Color Map:', options=plt.colormaps(), value=self._cmap,
            layout=FL(99), style={'description_width': '77px'})

        # settings panel — visibility toggles
        self.configOptionBtn = widgets.ToggleButton(
            description='', icon='gear', value=False,
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

        self.filterLabel    = widgets.Label(value='Image Filter Settings', layout=FLH(98))
        self.gaussianToggle = widgets.ToggleButton(value=False, description='Gaussian', layout=FLH(60))
        self.gaussianSize   = widgets.BoundedIntText(
            value=2, min=0, max=10, step=1, tooltip='Gaussian kernel size', layout=FLH(40))
        self.medianToggle   = widgets.ToggleButton(value=False, description='Median',   layout=FLH(60))
        self.medianSize     = widgets.BoundedIntText(
            value=3, min=1, max=20, step=1, tooltip='Median kernel size',   layout=FLH(40))
        self.laplacToggle   = widgets.ToggleButton(value=False, description='Laplace',  layout=FLH(60))
        self.laplaceSize    = widgets.BoundedIntText(
            value=1, min=1, max=10, step=1, tooltip='Laplace kernel size',  layout=FLH(40))

    def _build_layout(self) -> None:
        """Assemble widgets into HBox/VBox containers."""
        FL  = lambda w: widgets.Layout(display='flex', width=f'{w}%')
        FLB = lambda w: widgets.Layout(display='flex', width=f'{w}%',
                                       align_items='center', justify_content='center')
        FLH = lambda w: widgets.Layout(visibility='hidden', display='flex', width=f'{w}%')

        self.h_process_btn_layout = HBox(children=[
            self.directionBtn, self.fixZeroBtn, self.linebylineBtn,
            self.flattenBtn, self.invertBtn])
        self.h_channel_layout = HBox(children=[
            widgets.Label('Channel'), self.channelSelect])
        self.v_color_layout = VBox(children=[
            HBox(children=[self.vmin, self.vmax]), self.cmapSelection])
        self.h_user_layout = HBox(
            children=[VBox(children=[self.h_channel_layout, self.h_process_btn_layout]),
                      self.v_color_layout],
            layout=FLB(100))
        self.v_file_layout = VBox(children=[
            HBox(children=[widgets.Label('Folder', layout=FL(24)),
                           self.directoryDisplayDepth], layout=FL(98)),
            self.directorySelection,
            widgets.Label('Images'), self.selectionList,
            VBox(children=[
                HBox(children=[self.refreshBtn, self.saveBtn, self.copyBtn, self.configOptionBtn]),
                widgets.Label('Note')]),
            self.saveNote],
            layout=FL(20))
        self.v_settings_layout = VBox(children=[
            self.titleToggle, self.labelToggle,
            self.labelLabel,
            self.upperLeftSelect, self.upperRightSelect,
            self.lowerLeftSelect, self.lowerRightSelect,
            self.labelColorSelect, self.labelFontSize,
            self.titleLabel,
            self.channelToggle, self.setpointToggle, self.feedbackToggle,
            self.locationToggle, self.depthSelection,
            self.nameToggle, self.directionToggle, self.dateToggle,
            self.titleFontSize, self.filterLabel,
            HBox(children=[self.gaussianToggle, self.gaussianSize], layout=FLH(98)),
            HBox(children=[self.medianToggle,   self.medianSize],   layout=FLH(98)),
            HBox(children=[self.laplacToggle,   self.laplaceSize],  layout=FLH(98)),
        ], layout=FLH(10))
        self.v_image_layout = VBox(
            children=[self.figure_display, self.h_user_layout], layout=FLB(70))
        self.mainlayout = VBox(children=[
            HBox(children=[
                widgets.Label('Session', layout=widgets.Layout(
                    display='flex', justify_content='flex-start', width='10%')),
                self.rootFolder], layout=FL(99)),
            HBox(children=[self.v_file_layout, self.v_image_layout,
                           self.v_settings_layout], layout=FL(99))],
            layout=FL(100))

        self.v_settings_layout.layout.min_width = '200px'
        self.v_file_layout.layout.min_width = '200px'

    def _connect_observers(self) -> None:
        """Wire all observe() and on_click() callbacks."""
        for child in self.v_settings_layout.children:
            if hasattr(child, 'children'):
                for ch in child.children:
                    ch.observe(self.handler_settingsChange, names='value')
            child.observe(self.handler_settingsChange, names='value')
        self.saveBtn.on_click(self.save_figure)
        self.copyBtn.on_click(self.copy_figure)
        self.configOptionBtn.observe(self.handler_configOptionsDisplay, names='value')

        self.directionBtn.observe(self.update_scan_direction, names='value')
        self.linebylineBtn.observe(self.redraw_image, names='value')
        self.flattenBtn.observe(self.redraw_image, names='value')
        self.invertBtn.observe(self.redraw_image, names='value')
        self.fixZeroBtn.observe(self.redraw_image, names='value')
        self.edgesBtn.observe(self.redraw_image, names='value')
        self.gaussianBtn.observe(self.redraw_image, names='value')
        self.vmin.observe(self._on_limits_change, names='value')
        self.vmax.observe(self._on_limits_change, names='value')

        self.directoryDisplayDepth.observe(self.update_directories, names='value')
        self.nextBtn.on_click(self.nextDisplay)
        self.previousBtn.on_click(self.previousDisplay)
        self.refreshBtn.on_click(self.handler_root_folder_update)
        self.directorySelection.observe(self.handler_folder_selection, names='value')
        self.selectionList.observe(self.handler_file_selection, names='value')
        self.channelSelect.observe(self.handler_channel_selection, names='value')
        self.cmapSelection.observe(self._redraw, names='value')
        self.rootFolder.observe(self.handler_root_folder_update, names='value')

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display(self) -> None:
        """Render the browser widget."""
        display.clear_output(wait=True)
        display.display(self.mainlayout)

    def _redraw(self, *_) -> None:
        """Update axes and redraw the canvas."""
        self.update_axes()
        self.figure.tight_layout(pad=1)
        self.figure.canvas.draw()

    def _on_limits_change(self, _) -> None:
        """Callback for vmin/vmax changes; skipped during batch limit updates."""
        if not self._updating_limits:
            self._redraw()

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def save_figure(self, a) -> None:
        """Save the current figure to browser_outputs/ at 500 dpi."""
        self.saveBtn.icon = 'hourglass-start'
        out_dir = self.active_dir / 'browser_outputs'
        out_dir.mkdir(exist_ok=True)
        stem = (f'{str(self.directorySelection.value).split(chr(92))[-1]}'
                f'_{self.img.name.split(".")[0]}_{self.channelSelect.value}')
        if self.saveNote.value:
            stem += f'_{self.saveNote.value}'
        self.last_save_fname = str(out_dir / f'{stem}.png')
        self.figure.savefig(self.last_save_fname, dpi=500, format='png',
                            transparent=True, bbox_inches='tight')
        self.updateErrorText('Figure Saved')
        self.saveNote.value = ''
        self.saveBtn.icon = 'file-image-o'

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
        self._update_channel_selection()
        self.update_image_data()

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
            mode = ('Constant Height $\\rightarrow$ z-offset: %.3f%s' % z_offset
                    if fb_enable == 'OFF' else 'Constant Current')
            if abs(bias[0]) < 0.1:
                bias = (bias[0] * 1000, 'mV')
            self.scan_dict['bias'] = bias

            if self.channelToggle.value:
                ch_str = f'channel: {self.channelSelect.value}'
                label.append(f'{ch_str} $\\rightarrow$ {mode}'
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
                    name_str += f' $\\rightarrow$ direction: {d}'
                label.append(name_str)
            self.scan_dict['filename'] = self.img.name
            if self.dateToggle.value:
                label.append(f'Date: {self.img.header["rec_date"]} {self.img.header["rec_time"]}')
            self.scan_dict['date'] = f'{self.img.header["rec_date"]} {self.img.header["rec_time"]}'

        self.scan_info = '\n'.join(label) if self.titleToggle.value else ''

    def update_axes(self) -> None:
        """Render image_data onto the figure axes."""
        if self.img is None:
            return
        ax = self.axes
        ax.clear()
        img_origin = ('upper' if self._cache_scan_param('scan_dir') == 'down'
                      else 'lower')
        height = self.image_info['height']
        width  = self.image_info['width']
        data   = self.image_data[~np.isnan(self.image_data).any(axis=1)]
        row, col = self.image_data.shape
        y, x = data.shape
        w, h = width * x / col, height * y / row

        axesImage = ax.imshow(data, aspect='equal', origin=img_origin,
                              extent=[0, w, 0, h],
                              cmap=self.cmapSelection.value,
                              vmin=self.vmin.value, vmax=self.vmax.value)
        if not self.cb:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.02)
            self.cb = self.figure.colorbar(axesImage, cax=cax)
        else:
            self.cb.update_normal(axesImage)
        self.cb.set_label(f'{self.channelSelect.value} ({self.image_info["unit"]})',
                          fontsize=8, labelpad=0, rotation=270)
        self.cb.set_ticks([self.vmin.value, self.vmax.value])
        self.cb.set_ticklabels([f'{self.vmin.value:.2f}', f'{self.vmax.value:.2f}'],
                               fontsize=8, rotation=270, verticalalignment='center')
        self.cb.ax.tick_params(length=0)
        tl = self.cb.ax.get_yticklabels()
        tl[0].set_verticalalignment('bottom')
        tl[1].set_verticalalignment('top')

        ax.set_title(self.scan_info, fontsize=self.titleFontSize.value, loc='left')
        ax.set_xlabel('x (nm)', fontsize=self.labelFontSize.value)
        ax.set_ylabel('y (nm)', fontsize=self.labelFontSize.value)
        ax.set_xticks([0, w])
        ax.set_xticklabels([0, round(w, 2)], fontsize=self.labelFontSize.value)
        ax.set_yticks([0, h])
        ax.set_yticklabels([0, round(h, 2)], fontsize=self.labelFontSize.value)
        if self.labelToggle.value:
            ax.axis('off')
            self._add_figure_labels()
        else:
            ax.axis('on')

    def _add_figure_labels(self) -> None:
        """Overlay corner text / scalebar annotations on the axes."""
        color = self.labelColorSelect.value
        ax    = self.axes
        w, h  = ax.get_xlim()[1], ax.get_ylim()[1]
        corners = {
            'upper left':  (self.upperLeftSelect.value,  0.03 * w, 0.97 * h),
            'upper right': (self.upperRightSelect.value, 0.97 * w, 0.97 * h),
            'lower left':  (self.lowerLeftSelect.value,  0.03 * w, 0.03 * h),
            'lower right': (self.lowerRightSelect.value, 0.97 * w, 0.03 * h),
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
        for position, (selection, x_pos, y_pos) in corners.items():
            if selection == 'none':
                continue
            ha, va = self._ALIGN_PARAMS[position]
            if selection == 'scalebar':
                iw = self.image_info['width']
                standards = [.1, .2, .5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
                sb_len    = min(standards, key=lambda x: abs(x - iw * 0.20))
                label     = f'{sb_len * 10:.0f} Å' if sb_len < 1 else f'{sb_len:.0f} nm'
                self.font = fm(size=self.labelFontSize.value, family='sans-serif')
                ax.add_artist(AnchoredSizeBar(
                    ax.transAxes, sb_len / iw, label, position,
                    frameon=False, color=color, label_top=True,
                    sep=1, pad=1, fontproperties=self.font, size_vertical=1e-2))
                continue
            ax.text(x_pos, y_pos, text_map.get(selection, ''),
                    fontsize=self.labelFontSize.value,
                    verticalalignment=va, horizontalalignment=ha, color=color)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _update_channel_selection(self) -> None:
        current = self.channelSelect.value
        self.channelSelect.options = self.img.channels
        self.channelSelect.value = (current if current in self.img.channels
                                    else self.img.channels[0])

    def _update_info_text(self) -> None:
        self.filenameText.value = self.all_files[self.image_index]
        self.selectionList.value = self.all_files[self.image_index]

    def nextDisplay(self, a) -> None:
        if self.image_index < len(self.all_files) - 1:
            self.image_index += 1
            self._update_info_text()

    def previousDisplay(self, a) -> None:
        if self.image_index > 0:
            self.image_index -= 1
            self._update_info_text()

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

    def handler_settingsChange(self, a) -> None:
        self.redraw_image(a)

    def handler_configOptionsDisplay(self, a) -> None:
        self._set_settings_visibility(self.configOptionBtn.value)
        if self.configOptionBtn.value:
            self.depthSelection.layout.visibility = 'visible'

    def handler_folder_selection(self, a) -> None:
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
        self.selectionList.options = self.sxm_files
        if self.sxm_files:
            self.filenameText.value = self.sxm_files[index]
            if self.filenameText.value in self.selectionList.options:
                self.selectionList.value = self.filenameText.value

    def handler_file_selection(self, update) -> None:
        if self.selectionList.value is None:
            return
        self.image_index = self.sxm_files.index(self.selectionList.value)
        try:
            self.load_new_image()
            self._redraw()
        except Exception as err:
            self.updateErrorText('file selection error: ' + str(err))
            print(traceback.format_exc())

    def handler_channel_selection(self, update) -> None:
        try:
            self.update_image_data()
            self._redraw()
        except Exception as err:
            self.updateErrorText('channel selection error: ' + str(err))
