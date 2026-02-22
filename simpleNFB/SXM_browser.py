'''
Created: 14.06.23
Author: amsp
Description: ipython inferface for viewing and SXM image data in jupyter notebook

Functions:
    HBox: easy horizontal layout
    VBox: easy vertical layout
    Btn_Widget: basic button widget
    Text_Widget: basic text widget
    Image_Widget: basic image widget
    Selection_Widget: basic selection list widget
Classes:
    imageBrowser: view SXM data and export images with information
'''

import ipywidgets as widgets
from IPython import display
from scipy.ndimage import gaussian_filter, median_filter, gaussian_laplace
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties as fm
import numpy as np
import traceback
import subprocess
from pathlib import Path
import os
import sys
sys.path.append(r'./spmpy')
from spmpy import Spm

# Layouts
def HBox(*pargs, **kwargs):
    box = widgets.Box(*pargs, **kwargs)
    box.layout.display = 'flex'
    box.layout.align_items = 'stretch'
    return box
def VBox(*pargs, **kwargs):
    box = widgets.Box(*pargs, **kwargs)
    box.layout.display = 'flex'
    box.layout.flex_flow = 'column'
    box.layout.align_items = 'stretch'
    return box
# Custom Widgets
def Btn_Widget(displayText:str,**kwargs):
    return widgets.Button(description=displayText,**kwargs)

def Text_Widget(text:str,**kwargs):
    return widgets.Text(value=text,**kwargs)

def Image_Widget(filename:str,format='png',width=720,height=720,**kwargs):
    return widgets.Image(value=open(filename,'rb').read(),format=format,width=width,height=height,**kwargs)

def Selection_Widget(selection_list:list,label:str,rows=30):
    return widgets.Select(options=selection_list, description=label,disabled=False,rows=rows)

# Classes
class imageBrowser():
    '''
    Info:
        figure = imageBrowser.figure
        axes = imageBrowser.axes
        image_data = imageBrowser.image_data

        - imageBrowser.update_axes() can be used to refresh the browser plot
        - image_data can be accessed for further analysis/processing
        - axes can be accessed for futher plot modification (axis limits, labels, etc)
    '''
    def __init__(self,figsize=(6,6),fontsize=12,titlesize=12,cmap='Greys_r',home_directory='./'):
        self.img = None
        self.figure,self.axes = plt.subplots(ncols=1,num='sxm') # simple default figure size
        self.figure.canvas.header_visible = False
        #self.figure.canvas.layout = widgets.Layout(display='flex',width=f'9%',align_items='center',justify_content='center')
        self.fontsize = fontsize
        self.font = fm(size=fontsize,family='sans-serif')
        self.titlesize = titlesize
        self.cb = None
        self.image_data = np.zeros((64,64)) # 64 x 64 pixel zeros
        self.image_info = {'height':1,'width':1,'unit':'nm'}
        self.scan_dict = {}
        self.scan_info = ''
        self.errors = []
        self.image_index = 0

        self.active_dir = Path(home_directory)
        self.sxm_files = []
        self.dat_files = []
        self.directories = [self.active_dir]

        # widget layouts
        smallLayout = widgets.Layout(visibility='visible',width='80px')
        mediumLayout = widgets.Layout(visibility='visible',width='120px')
        largeLayout = widgets.Layout(visibility='visible',width='160px')
        extraLargeLayout = widgets.Layout(visibility='visible',width='200px')
        layout = lambda x: widgets.Layout(visibility='visible',width=f'{x}px')
        layout_h = lambda x: widgets.Layout(visibility='hidden',width=f'{x}px')
        flex_layout = lambda x: widgets.Layout(display='flex',width=f'{x}%')
        flex_layout_btn = lambda x: widgets.Layout(display='flex',width=f'{x}%',align_items='center',justify_content='center')
        flex_layout_h = lambda x: widgets.Layout(visibility='hidden',display='flex',width=f'{x}%')
        self._layout = layout
        
        # selections
        self.rootFolder = widgets.Text(description='',layout=widgets.Layout(dispaly='flex',width='90%'))
        self.directorySelection = widgets.Select(description='',options=self.directories,rows=8,layout=flex_layout(98))
        self.selectionList = widgets.Select(description='',options=self.sxm_files,rows=27,layout=flex_layout(98))
        self.directoryDisplayDepth = widgets.Dropdown(description='depth',value=1,options=['full',1,2,3,4,5],tooltip='depth of the folder structure displayed in the selection menu',layout=flex_layout_btn(75),style={'description_width':'40px'})
        #self.channelSelect = Selection_Widget(['z'],'Channels:',rows=5)
        self.channelSelect = widgets.Dropdown(description='',layout=layout(165))
        self.refreshBtn = Btn_Widget('',icon='refresh',tooltip='Reload file list',layout=flex_layout_btn(24))
        
        # text display
        self.filenameText = Text_Widget('')
        self.indexText = Text_Widget('0')
        self.errorText = Selection_Widget([],'Out:',rows=5)
        self.saveNote = Text_Widget('',description='',tooltip='This text is appended to filename when the figure is saved',layout=flex_layout(99))
        
        # image display
        self.nextBtn = Btn_Widget('',layout=layout(40),icon='arrow-circle-down',tooltip='Load next image in list')
        self.previousBtn = Btn_Widget('',layout=layout(40),icon='arrow-circle-up',tooltip='Load previous image in list')
        self.linebylineBtn = widgets.ToggleButton(description='',value=False,layout=layout(40),icon='align-justify',tooltip='Line by line linear subtraction')
        self.flattenBtn = widgets.ToggleButton(description='',value=False,layout=layout(40),icon='square-o',tooltip='Apply plane fit and subtraction')
        self.edgesBtn = widgets.ToggleButton(description='',value=False,layout=layout(40),icon='dot-circle-o',tooltip='Apply laplace filter (edge detection)')
        self.gaussianBtn = widgets.ToggleButton(description='',value=False,layout=layout(40),icon='bullseye',tooltip='Apply a 3x3 Gaussian filter')
        self.invertBtn = widgets.ToggleButton(description='',value=False,layout=layout(40),icon='exchange',tooltip='Invert sign of the image data')
        self.directionBtn = widgets.ToggleButton(description='',value=False,layout=layout(40),icon='caret-square-o-right',tooltip='select scan direction, default is forward')
        self.fixZeroBtn = widgets.ToggleButton(description='',value=False,layout=layout(40),icon='neuter',tooltip='Rescale the image data so the minimum value is zero')
        
        # outputs
        self.saveBtn = Btn_Widget('',layout=flex_layout_btn(24),icon='file-image-o',tooltip='Save displayed image to \\browser_output folder\nText in the "note" is appended to figure filename')
        self.copyBtn = Btn_Widget('',layout=flex_layout_btn(24),icon='clipboard',tooltip='Save displayed image to \\browser_output folder\ncopy displayed image to clipboard')
        self.figure_display = widgets.Output(layout=flex_layout_btn(99))
        
        # cmap options
        self.vmin = widgets.FloatText(value=0,description='Min:',step=.1,layout=flex_layout(50),style={'description_width':'40px'})
        self.vmax = widgets.FloatText(value=1,description='Max:',step=.1,layout=flex_layout(50),style={'description_width':'40px'})
        self.cmapSelection = widgets.Dropdown(description='Color Map:',options=plt.colormaps(),value=cmap,layout=flex_layout(99),style={'description_width':'77px'})
        
        # figure display toggles
        ### show title
        self.configOptionBtn = widgets.ToggleButton(description='',icon='gear',value=False,tooltip='Display options panel',layout=flex_layout_btn(24))
        self.titleToggle = widgets.ToggleButton(value=True, description='Show Title',tooltip='Toggle figure title',layout=flex_layout_h(98))
        self.labelToggle = widgets.ToggleButton(value=False, description='Show Labels',tooltip='Toggle figure laels',layout=flex_layout_h(98))

        ### show info labels
        label_options = ['none','channel','bias','setpoint','feedback','date','filename','scalebar','filters']
        self.labelLabel = widgets.Label(value='Figure Label Settings',layout=flex_layout_h(98))
        self.upperLeftSelect = widgets.Dropdown(value='bias',options=label_options,description='UL:',layout=flex_layout_h(98),style={'description_width':'40px'})
        self.upperRightSelect = widgets.Dropdown(value='filename',options=label_options,description='UR:',layout=flex_layout_h(98),style={'description_width':'40px'})
        self.lowerLeftSelect = widgets.Dropdown(value='none',options=label_options,description='LL:',layout=flex_layout_h(98),style={'description_width':'40px'})
        self.lowerRightSelect = widgets.Dropdown(value='scalebar',options=label_options,description='LR:',layout=flex_layout_h(98),style={'description_width':'40px'})
        self.labelColorSelect = widgets.ColorPicker(concise=True,description='Color:',value='orange',layout=flex_layout_h(98),style={'description_width':'40px'})
        self.labelFontSize = widgets.IntText(description='size:',value=20,tooltip='change label font size',style={'description_width':'40px'},layout=flex_layout(98))
        
        # figure title options
        self.titleLabel = widgets.Label(value='Figure Title Settings',layout=flex_layout_h(98))
        self.channelToggle = widgets.ToggleButton(value=True,description='channel',layout=flex_layout_h(98))
        self.setpointToggle = widgets.ToggleButton(value=True,description='Setpoint',layout=flex_layout_h(98))
        self.feedbackToggle = widgets.ToggleButton(value=True,description='Feedback',layout=flex_layout_h(98))
        self.locationToggle = widgets.ToggleButton(value=True,description='file location',layout=flex_layout_h(98))
        self.depthSelection = widgets.Dropdown(value='full',options=['full',1,2,3,4,5],description='Depth:',tooltip='folder depth to display in location section of the image title',layout=flex_layout_h(98),style={'description_width':'40px'})
        self.nameToggle = widgets.ToggleButton(value=True,description='Filename',layout=flex_layout_h(98))
        self.directionToggle = widgets.ToggleButton(value=True,description='Direction',layout=flex_layout_h(98))
        self.dateToggle = widgets.ToggleButton(value=True,description='Date',layout=flex_layout_h(98))
        self.titleFontSize = widgets.IntText(description='size:',value=9,tooltip='change title font size',style={'description_width':'40px'},layout=flex_layout(98))

        # image filter settings
        self.filterLabel = widgets.Label(value='Image Filter Settings',layout=flex_layout_h(98))
        self.gaussianToggle = widgets.ToggleButton(value=False,description='Gaussian',layout=flex_layout_h(60))
        self.gaussianSize = widgets.BoundedIntText(value=2,min=0,max=10,step=1,tooltip='size of the gaussain kernel',layout=flex_layout_h(40))
        self.medianToggle = widgets.ToggleButton(value=False,description='Median',layout=flex_layout_h(60))
        self.medianSize = widgets.BoundedIntText(value=3,min=1,max=20,step=1,tooltip='size of the median kernel',layout=flex_layout_h(40))
        self.laplacToggle = widgets.ToggleButton(value=False,description='Laplace',layout=flex_layout_h(60))
        self.laplaceSize = widgets.BoundedIntText(value=1,min=1,max=10,step=1,tooltip='size of the laplace filter kernel',layout=flex_layout_h(40))
        # plane fit settings ### needs interactive plot functionality (select 3 points) --> new implementation of plane subtraction function
        self.planeFitToggle = widgets.ToggleButton(value=False,description='Plane Fit',tooltip='plane subtraction')

        # layouts
        self.h_process_btn_layout = HBox(children=[self.directionBtn,self.fixZeroBtn,self.linebylineBtn,self.flattenBtn,self.invertBtn])
        self.h_channel_layout = HBox(children=[widgets.Label('Channel'),self.channelSelect])
        self.v_color_layout = VBox(children=[HBox(children=[self.vmin,self.vmax]),self.cmapSelection])
        self.h_user_layout = HBox(children=[VBox(children=[self.h_channel_layout,self.h_process_btn_layout]),self.v_color_layout],layout=flex_layout_btn(100))
        self.v_file_layout = VBox(children=[HBox(children=[widgets.Label('Folder',layout=flex_layout(24)),self.directoryDisplayDepth],layout=flex_layout(98)),
                                            self.directorySelection,
                                            widgets.Label('Images'),self.selectionList,
                                            VBox(children=[HBox(children=[self.refreshBtn,self.saveBtn,self.copyBtn,self.configOptionBtn]),
                                                           widgets.Label('Note')]),
                                                           self.saveNote],layout=flex_layout(20))
        self.v_settings_layout = VBox(children=[self.titleToggle,
                                                self.labelToggle,
                                                self.labelLabel,
                                                self.upperLeftSelect,
                                                self.upperRightSelect,
                                                self.lowerLeftSelect,
                                                self.lowerRightSelect,
                                                self.labelColorSelect,
                                                self.labelFontSize,
                                                self.titleLabel,
                                             self.channelToggle,
                                             self.setpointToggle,
                                             self.feedbackToggle,
                                             self.locationToggle,
                                             self.depthSelection,
                                             self.nameToggle,
                                             self.directionToggle,
                                             self.dateToggle,
                                             self.titleFontSize,
                                             self.filterLabel,
                                             HBox(children=[self.gaussianToggle,self.gaussianSize],layout=flex_layout_h(98)),
                                             HBox(children=[self.medianToggle,self.medianSize],layout=flex_layout_h(98)),
                                             HBox(children=[self.laplacToggle,self.laplaceSize],layout=flex_layout_h(98))],layout=flex_layout_h(10))
        
        self.v_image_layout = VBox(children=[self.figure_display,self.h_user_layout],layout=flex_layout_btn(70))

        self.mainlayout = VBox(children=[HBox(children=[widgets.Label('Session',layout=widgets.Layout(display='flex',justify_content='flex-start',width='10%')),
                                                        self.rootFolder],layout=flex_layout(99)),
                                         HBox(children=[self.v_file_layout,self.v_image_layout,self.v_settings_layout],layout=flex_layout(99))],
                                         layout=flex_layout(100))

        ### configure scaling behavior
        self.v_settings_layout.layout.min_width = '200px'
        self.v_file_layout.layout.min_width = '200px'
        ## Display and output events
        #### connect config panel widgets to functions
        for child in self.v_settings_layout.children:
            if type(child) == type(self.v_settings_layout):
                for ch in child.children:
                    ch.observe(self.handler_settingsChange,names='value')
            child.observe(self.handler_settingsChange,names='value')
        self.saveBtn.on_click(self.save_figure)
        self.copyBtn.on_click(self.copy_figure)
        self.configOptionBtn.observe(self.handler_configOptionsDisplay,names='value')

        ## image processing events
        self.directionBtn.observe(self.update_scan_direction,names='value')
        self.linebylineBtn.observe(self.redraw_image,names='value')
        self.flattenBtn.observe(self.redraw_image,names='value')
        self.invertBtn.observe(self.redraw_image,names='value')
        self.fixZeroBtn.observe(self.redraw_image,names='value')
        self.edgesBtn.observe(self.redraw_image,names='value')
        self.gaussianBtn.observe(self.redraw_image,names='value')
        self.vmin.observe(self.updateDisplayImage,names='value')
        self.vmax.observe(self.updateDisplayImage,names='value')

        ## selection events
        self.directoryDisplayDepth.observe(self.update_directories,names=['value'])
        self.nextBtn.on_click(self.nextDisplay)
        self.previousBtn.on_click(self.previousDisplay)
        self.refreshBtn.on_click(self.handler_root_folder_update)
        self.directorySelection.observe(self.handler_folder_selection,names=['value'])
        self.selectionList.observe(self.handler_file_selection,names=['value'])
        self.channelSelect.observe(self.handler_channel_selection,names=['value'])
        self.cmapSelection.observe(self.updateDisplayImage,names='value')
        self.rootFolder.observe(self.handler_root_folder_update,names='value')

        # mpl events
        #self.figure.canvas.mpl_connect('button_press_event',self.mouse_click)

        self.display()
        with plt.ioff():
            with self.figure_display:
                self.figure_display.clear_output(wait=True)
                plt.show(self.figure)
        #self.find_directories(self.active_dir)
        #self.update_directories()
        #self.updateInfoText()
        #self.handler_file_selection('startup')
        #self.updateErrorText('finish startup')
    # show browser
    def display(self):
        display.clear_output(wait=True)
        display.display(self.mainlayout)
    def find_directories(self,_path):
        directories = []
        for _directory in os.listdir(_path):
            if _directory[-4:] in ['.dat','.sxm']: continue
            elif os.path.isdir(_path / _directory):
                if 'browser_outputs' in _directory or 'ipynb' in _directory or 'raw_stml_data' in _directory: continue
                directories.append(_path / _directory)
                self.find_directories(_path / _directory)
        else:
            pass
        self.directories.extend(directories)
        return directories
    def update_directories(self,a):
        if self.directoryDisplayDepth.value == 'full':
            depth = 0
        else:
            depth = -self.directoryDisplayDepth.value
        display_directories = ['\\'.join(str(directory).split('\\')[depth:]) for directory in self.directories]
        display_directories[0] = 'session folder'
        self.directorySelection.options = display_directories
    def copy_figure(self,a):
        self.save_figure(a)
        # Make powershell command
        powershell_command = r'$imageFilePaths = @("'
        for image_path in [self.last_save_fname]:
            powershell_command += image_path + '","'
        powershell_command = powershell_command[:-2] + '); '
        powershell_command += r'Set-Clipboard -Path $imageFilePaths;'
        self.copyBtn.icon = 'hourglass-half'
        # Execute Powershell
        completed = subprocess.run(["powershell", "-Command", powershell_command], capture_output=True)
        self.copyBtn.icon = 'clipboard'
    def save_figure(self,a):
        self.saveBtn.icon = 'hourglass-start'
        if os.path.exists(self.active_dir / 'browser_outputs'):
            pass
        else:
            os.mkdir(self.active_dir / 'browser_outputs')
        fname = 'browser_outputs/' + str(self.directorySelection.value).split("\\")[-1] + f'_{self.img.name.split(".")[0]}_{self.channelSelect.value}'
        if self.saveNote.value != '':
            fname += f'_{self.saveNote.value}'
        fname = self.active_dir / fname
        self.last_save_fname = f'{fname}.png'
        self.figure.savefig(f'{fname}.png',dpi=500,format='png',transparent=True,bbox_inches='tight')
        self.updateErrorText('Figure Saved')
        self.saveNote.value = ''
        self.saveBtn.icon = 'file-image-o'
    def remove_line_average(self):
        for i in range(self.image_data.shape[0]):
            if np.isnan(self.image_data[i,:]).any():
                continue
            try:
                y = self.image_data[i,:]
                x = np.arange(self.image_data.shape[1])
                coef = np.polyfit(x,y,1)
                self.image_data[i,:] -= np.polyval(coef,x)
            except:
                print('linebyline error: ',i)
                print( self.image_data[i,:])
    # linescan
    def mouse_click(self,event):
        x = event.xdata
        y = event.ydata
        #image_value = self.image_data[iy,ix]
        try:
            ix = np.round(ix,3)
            iy = np.round(iy,3)
        except:
            pass
        self.saveNote.value = f'{x}'
        print(ix,iy)
        #d = np.sqrt(ix**2+iy**2)

        #self.updateErrorText(str(round(event.xdata,2)) + ' ' + str(round(event.ydata,2)))
    # image generation
    def redraw_image(self,a):
        if self.img != None:
            self.update_image_data()
            self.updateDisplayImage()
    def load_new_image(self):
        #self.updateErrorText('load new image')
        directory = self.directories[self.directorySelection.index]
        if directory != self.active_dir:
            directory = os.path.join(self.active_dir,directory)
        file = os.path.join(directory,self.sxm_files[self.image_index])
        self.img = Spm(file)
        self.filenameText.value = self.all_files[self.image_index]
        self.updateChannelSelection()
        self.update_image_data()
        #self.updateErrorText('finish load new image')
    def update_image_data(self):
        #self.updateErrorText('update image data')
        channel = self.channelSelect.value
        if self.directionBtn.value:
            direction = 'backward'
        else:
            direction = 'forward'

        flatten = self.flattenBtn.value
        offset = False # look into how this works
        zero = False # look into how this works
        try:
            self.image_data,unit = self.img.get_channel(channel, direction = direction, flatten=flatten, offset=offset,zero=zero)
        except:
            self.updateErrorText('Error in flattening routine: setting flatten=False')
            self.image_data,unit = self.img.get_channel(channel, direction = direction, flatten=False, offset=offset,zero=zero)
        if direction == 'backward':
            self.image_data = np.flip(self.image_data,axis=1)
        self.image_info['unit'] = unit
        if self.invertBtn.value:
            self.image_data *= -1
        if self.linebylineBtn.value:
            self.remove_line_average()
        if self.fixZeroBtn.value:
            self.image_data -= np.nanmin(self.image_data)        
        if self.gaussianToggle.value:
            self.image_data = gaussian_filter(self.image_data,self.gaussianSize.value)
        if self.medianToggle.value:
            self.image_data = median_filter(self.image_data,size=self.medianSize.value)
        if self.laplacToggle.value:
            self.image_data = -1*gaussian_laplace(self.image_data,self.laplaceSize.value)

        # set vmin and vmax without triggers image update
        self.vmin.unobserve(self.updateDisplayImage,names='values')
        self.vmax.unobserve(self.updateDisplayImage,names='values')
        self.vmin.value = round(np.nanmin(self.image_data),3)
        self.vmax.value = round(np.nanmax(self.image_data),3)
        self.vmin.observe(self.updateDisplayImage,names='values')
        self.vmax.observe(self.updateDisplayImage,names='values')
        self.update_scan_info()
        #self.updateErrorText('finish update image data')
    def update_scan_info(self):
        #self.updateErrorText('update scan info')
        label = []
        if self.img.type == 'scan':
            fb_enable = self.img.get_param('z-controller>controller status')
            fb_ctrl = self.img.get_param('z-controller>controller name')
            bias = self.img.get_param('V')
            set_point = self.img.get_param('setpoint')
            height = self.img.get_param('height')
            width = self.img.get_param('width')
            angle = self.img.get_param('angle')
            z_offset = self.img.get_param('z_offset')
            comment = self.img.get_param('comments')

            self.image_info['width'] = width[0]
            self.image_info['height'] = height[0]
            self.scan_dict['feedback'] = fb_enable
            self.scan_dict['setpoint'] = set_point
            self.scan_dict['size'] = (width,height,angle)
            self.scan_dict['z_offset'] = z_offset
            self.scan_dict['comment'] = comment

            if fb_enable == 'OFF':
                mode = 'Constant Height $\\rightarrow$ z-offset: %.3f%s' % z_offset
                #label.append('mode: Constant Height $\\rightarrow$ z-offset: %.3f%s' % z_offset)
            else:
                mode = 'Constant Current'
                #label.append('mode: Constant Current')
                
            if np.abs(bias[0])<0.1:
                bias = list(bias)
                bias[0] = bias[0]*1000
                bias[1] = 'mV'
                bias = tuple(bias)
            self.scan_dict['bias'] = bias

            #label.append(f'channel: {self.channelSelect.value}')
            if self.channelToggle.value:
                if self.feedbackToggle.value:
                    label.append(f'channel: {self.channelSelect.value} $\\rightarrow$ {mode}')
                else:
                    label.append(f'channel: {self.channelSelect.value}')
            if self.setpointToggle.value:
                label.append(f'setpoint: I = {set_point[0]:.0f}{set_point[1]}, V = {bias[0]:.2f}{bias[1]}')

            if self.locationToggle.value:
                location = self.directories[self.directorySelection.index]
                if self.depthSelection.value == 'full':
                    location = location
                else:
                    location = '\\'.join(str(location).split('\\')[-int(self.depthSelection.value):])
                label.append(f'location: {location}') #dat files use header['Saved Date']

            if self.nameToggle.value:
                if self.directionToggle.value:
                    direction = {True:'backward',False:'forward'}[self.directionBtn.value]
                    name_str = f'filename: {self.img.name} $\\rightarrow$ direction: {direction}'
                else:
                    name_str = f'filename: {self.img.name}'
                label.append(name_str)
            self.scan_dict['filename'] = self.img.name

            if self.dateToggle.value:
                label.append(f'Date: {self.img.header["rec_date"]} {self.img.header["rec_time"]}')

            self.scan_dict['date'] = f'{self.img.header["rec_date"]} {self.img.header["rec_time"]}'

        if self.titleToggle.value:
            self.scan_info = '\n'.join(label)
        else:
            self.scan_info = ''
        #self.updateErrorText('finish update scan info')

    def update_axes(self):
        #self.updateErrorText('update axes')
        if not self.figure:
            self.updateErrorText('making new figure/axes')
            self.figure,self.axes = plt.subplots(ncols=1,figsize=(8,8))
            #self.figure.tight_layout(pad=2)
        ax = self.axes
        ax.clear()
        imgOrigin = 'lower'
        if self.img.get_param('scan_dir') == 'down':
            imgOrigin = 'upper'
        height = self.image_info['height']
        width = self.image_info['width']
        data = self.image_data[~np.isnan(self.image_data).any(axis=1)]
        row,col = self.image_data.shape
        y,x = data.shape
        w = width*x/col
        h = height*y/row
        axesImage = ax.imshow(data,aspect='equal',origin=imgOrigin,extent=[0,w,0,h],cmap=self.cmapSelection.value,vmin=self.vmin.value,vmax=self.vmax.value) # add cmap and vmin/max
        if not self.cb:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%",pad=0.05)
            #self.cb = self.figure.colorbar(axesImage,ax=ax,shrink=0.75,pad=.05)
            self.cb = self.figure.colorbar(axesImage,cax=cax)
        else:
            #self.cb.remove()
            self.cb.update_normal(axesImage)# = self.figure.colorbar(axesImage,ax=ax,shrink=0.75,pad=.01)
        self.cb.set_label(f'{self.channelSelect.value} ({self.image_info["unit"]})',fontsize=self.labelFontSize.value)
        ax.set_title(self.scan_info,fontsize=self.titleFontSize.value,loc='left')
        ax.set_xlabel('x (nm)',fontsize=self.labelFontSize.value)
        ax.set_ylabel('y (nm)',fontsize=self.labelFontSize.value)
        ax.set_xticks([0,w])
        ax.set_yticks([0,h])
        ax.set_xticklabels([0,round(w,2)],fontsize=self.labelFontSize.value)
        ax.set_yticklabels([0,round(h,2)],fontsize=self.labelFontSize.value)
        if self.labelToggle.value:
            ax.axis('off')
            self.addFigureLabels()
        else:
            ax.axis('on')

    def addFigureLabels(self):
        color = self.labelColorSelect.value
        ax = self.axes
        w = ax.get_xlim()[1]
        h = ax.get_ylim()[1]
        label_positions = {
            'upper left': (0.03*w,0.97*h,'upper left'),
            'upper right': (0.97*w,0.97*h,'upper right'),
            'lower left': (0.03*w,0.03*h,'lower left'),
            'lower right': (0.97*w,0.03*h,'lower right')
        }
        selections = {
            'upper left': self.upperLeftSelect.value,
            'upper right': self.upperRightSelect.value,
            'lower left': self.lowerLeftSelect.value,
            'lower right': self.lowerRightSelect.value
        }
        for position in label_positions.keys():
            x_pos,y_pos,align = label_positions[position]
            selection = selections[position]
            if selection == 'none':
                continue
            elif selection == 'channel':
                text = f'{self.channelSelect.value}'
            elif selection == 'bias':
                bias = self.scan_dict.get('bias',('N/A',''))
                text = f'{bias[0]}{bias[1]}'
            elif selection == 'setpoint':
                setpoint = self.scan_dict.get('setpoint','N/A')
                text = f'{setpoint[0]} {setpoint[1]}'
            elif selection == 'feedback':
                feedback = self.scan_dict.get('feedback','N/A')
                text = f'{feedback}'
            elif selection == 'date':
                date = self.scan_dict.get('date','N/A')
                text = f'{date}'
            elif selection == 'filename':
                filename = self.scan_dict.get('filename','N/A')
                text = f'{filename}'
            elif selection == 'scalebar':
                image_width = self.image_info['width'] # in nm
                target_length = 0.20 # % of image width
                initial_scalebar_length = image_width * target_length
                # round to nearest standard value
                standard_lengths = [.1,.2,.5,1,2,5,10,20,50,100,200,500,1000,2000,5000] # in nm
                scalebar_length = min(standard_lengths, key=lambda x: abs(x-initial_scalebar_length))
                actual_length_per = scalebar_length / image_width # percent
                if scalebar_length < 1:
                    scalebar_length *= 10
                    unit = 'Å'
                else:
                    unit = 'nm'
                label = f'{scalebar_length} {unit}'
                self.font = fm(size=self.labelFontSize.value,family='sans-serif')
                scalebar = AnchoredSizeBar(ax.transAxes, actual_length_per,label,align,frameon = False,color=color,label_top=True,sep=1,pad=1,fontproperties=self.font,size_vertical=1e-2)
                ax.add_artist(scalebar)
                continue
            elif selection == 'filters':
                text = []
                if self.gaussianToggle.value: text.append('G')
                if self.medianToggle.value: text.append('M')
                if self.laplacToggle.value: text.append('LP')
                text = '+'.join(text)
            # add text to axes
            if align == 'upper left':
                ax.text(x_pos,y_pos,text,fontsize=self.labelFontSize.value,verticalalignment='top',horizontalalignment='left',color=color) #,backgroundcolor='black'
            elif align == 'upper right':
                ax.text(x_pos,y_pos,text,fontsize=self.labelFontSize.value,verticalalignment='top',horizontalalignment='right',color=color)
            elif align == 'lower left':
                ax.text(x_pos,y_pos,text,fontsize=self.labelFontSize.value,verticalalignment='bottom',horizontalalignment='left',color=color)
            elif align == 'lower right':
                ax.text(x_pos,y_pos,text,fontsize=self.labelFontSize.value,verticalalignment='bottom',horizontalalignment='right',color=color)
        #self.updateErrorText('finish update axes')
    # display configuration
    def updateDisplayImage(self,*params):
        #self.updateErrorText('update display image')
        self.update_axes()
        #self.figure.canvas.set_window_title(self.img.name.split('.')[0])
        self.figure.tight_layout(pad=1)
        self.figure.canvas.draw()
        #self.updateErrorText('finish update display image')
    def updateInfoText(self):
        self.filenameText.value = self.all_files[self.image_index]
        self.selectionList.value = self.all_files[self.image_index]
    def updateChannelSelection(self):
        current_value = self.channelSelect.value
        self.channelSelect.options = self.img.channels
        if current_value in self.img.channels:
            self.channelSelect.value = current_value
        else:
            self.channelSelect.value = self.img.channels[0]
        #self.updateErrorText(self.channelSelect.value)
    def updateErrorText(self,text):
        self.errors.append(f'{len(self.errors)} {text}')
        self.errorText.options = self.errors

    def nextDisplay(self,a):
        if self.image_index == len(self.all_files)-1:
            return
        else:
            self.image_index += 1
            self.updateInfoText()
    def previousDisplay(self,a):
        if self.image_index == 0:
            return
        else:
            self.image_index -= 1
            self.updateInfoText()
    def update_scan_direction(self,a):
        if self.directionBtn.value:
            self.directionBtn.icon = 'caret-square-o-left'
        else:
            self.directionBtn.icon = 'caret-square-o-right'
        self.redraw_image(a)
### Selection update
    def handler_settingsChange(self,a):
        self.redraw_image(a)

    def handler_configOptionsDisplay(self,a):
        if self.configOptionBtn.value:
            self.v_settings_layout.layout.visibility = 'visible'
            for child in self.v_settings_layout.children:
                if type(child) == type(self.v_settings_layout):
                    for ch in child.children:
                        ch.layout.visibility = 'visible'
                child.layout.visibility = 'visible'
            self.depthSelection.layout.visibility = 'visible'
        else:
            self.v_settings_layout.layout.visibility = 'hidden'
            for child in self.v_settings_layout.children:
                if type(child) == type(self.v_settings_layout):
                    for ch in child.children:
                        ch.layout.visibility = 'hidden'
                child.layout.visibility = 'hidden'

    def handler_root_folder_update(self,a):
        new_root = self.rootFolder.value
        if type(a) == type(self.refreshBtn): 
            current_directory = self.directorySelection.value
            current_file = self.selectionList.value
            # check if filepath exists
        exists = os.path.exists(new_root)
        is_dir = os.path.isdir(new_root)
        if exists and is_dir:
            self.directorySelection.options = [self.active_dir]
            self.directories = [self.active_dir]
            self.active_dir = Path(new_root)
            self.find_directories(self.active_dir)
            self.update_directories(a)
        if type(a) == type(self.refreshBtn): 
            self.directorySelection.value = current_directory
            #self.selectionList.value = current_file
    def handler_folder_selection(self,a):
        index=0
        if type(a) == type(self.refreshBtn): 
            index = self.selectionList.index
        directory = self.directories[self.directorySelection.index]
        #if self.directorySelection.value != self.active_dir:
        #    directory = f'{self.active_dir}/{self.directorySelection.value}'
        #else:
        #    directory = self.active_dir
        self.sxm_files = []
        self.dat_files = []
        for file in os.listdir(directory):
            if '.sxm' in file:
                self.sxm_files.append(file)
            elif '.dat' in file:
                self.dat_files.append(file)
        self.all_files = self.sxm_files + self.dat_files
        self.sxm_files = list(np.flip(self.sxm_files))
        self.selectionList.options = self.sxm_files
        if len(self.sxm_files) != 0:
            self.filenameText.value = self.sxm_files[index]
            if self.filenameText.value in self.selectionList.options:
                self.selectionList.value = self.filenameText.value
    def handler_file_selection(self,update):
        #self.updateErrorText(str(update))
        if self.selectionList.value != None:
            self.image_index = self.sxm_files.index(self.selectionList.value)
        else:
            return
        try:
            self.load_new_image()
            self.updateDisplayImage()
        except Exception as err:
            self.updateErrorText('file selection error:' + str(err))
            print(traceback.format_exc())
    def handler_channel_selection(self,update):
        try:
            channel_number = self.img.channels.index(self.channelSelect.value)
            self.update_image_data()
            self.updateDisplayImage()
        except Exception as err:
            self.updateErrorText('channel selection error:' + str(err))
