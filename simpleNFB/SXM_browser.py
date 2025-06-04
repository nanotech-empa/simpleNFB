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
from skimage import filters
from scipy.ndimage import gaussian_filter, median_filter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
        self.figure,self.axes = plt.subplots(ncols=1,figsize=figsize,num='sxm') # simple default figure size
        self.fontsize = fontsize
        self.titlesize = titlesize
        self.cb = None
        self.image_data = np.zeros((64,64)) # 64 x 64 pixel zeros
        self.image_info = {'height':1,'width':1,'unit':'nm'}
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
        self._layout = layout
        # selections
        self.directorySelection = Selection_Widget(self.directories,'Folders:',rows=8)
        self.selectionList = Selection_Widget(self.sxm_files,'SXM Files:',rows=27)
        #self.channelSelect = Selection_Widget(['z'],'Channels:',rows=5)
        self.channelSelect = widgets.Dropdown(description='Channels:',layout=layout(200))
        self.refreshBtn = Btn_Widget('',icon='refresh',tooltip='Reload file list',layout=layout(30))
        # text display
        self.filenameText = Text_Widget('')
        self.indexText = Text_Widget('0')
        self.errorText = Selection_Widget([],'Out:',rows=5)
        self.saveNote = Text_Widget('',description='Note:',tooltip='This text is appended to filename when the figure is saved',layout=layout(200))
        # image display
        self.nextBtn = Btn_Widget('',layout=layout(30),icon='arrow-circle-down',tooltip='Load next image in list')
        self.previousBtn = Btn_Widget('',layout=layout(30),icon='arrow-circle-up',tooltip='Load previous image in list')
        self.linebylineBtn = widgets.ToggleButton(description='',value=False,layout=layout(30),icon='align-justify',tooltip='Line by line linear subtraction')
        self.flattenBtn = widgets.ToggleButton(description='',value=False,layout=layout(30),icon='square-o',tooltip='Apply plane fit and subtraction')
        self.edgesBtn = widgets.ToggleButton(description='',value=False,layout=layout(30),icon='dot-circle-o',tooltip='Apply laplace filter (edge detection)')
        self.gaussianBtn = widgets.ToggleButton(description='',value=False,layout=layout(30),icon='bullseye',tooltip='Apply a 3x3 Gaussian filter')
        self.invertBtn = widgets.ToggleButton(description='',value=False,layout=layout(30),icon='exchange',tooltip='Invert sign of the image data')
        self.fixZeroBtn = widgets.ToggleButton(description='',value=False,layout=layout(30),icon='neuter',tooltip='Rescale the image data so the minimum value is zero')
        self.saveBtn = Btn_Widget('',layout=layout(30),icon='file-image-o',tooltip='Save displayed image to \\browser_output folder\nText in the "note" is appended to figure filename')
        self.copyBtn = Btn_Widget('',layout=layout(30),icon='clipboard',tooltip='Save displayed image to \\browser_output folder\ncopy displayed image to clipboard')
        self.figure_display = widgets.Output()
        # cmap options
        self.vmin = widgets.FloatText(value=0,description='Min:',step=.1,layout=layout(180))
        self.vmax = widgets.FloatText(value=1,description='Max:',step=.1,layout=layout(180))
        self.cmapSelection = widgets.Dropdown(description='colormap:',options=plt.colormaps(),value=cmap,layout=layout(180))
        # figure title options
        self.titleOptionBtn = widgets.ToggleButton(description='',icon='gear',value=False,tooltip='Display figure title options panel',layout=layout(30))
        self.titleLabel = widgets.Label(value='Figure Title Settings',layout=layout_h(150))
        self.titleToggle = widgets.ToggleButton(value=True, description='Show Title',tooltip='Toggle figure title',layout=layout_h(150))
        self.channelToggle = widgets.ToggleButton(value=True,description='channel',layout=layout_h(150))
        self.setpointToggle = widgets.ToggleButton(value=True,description='Setpoint',layout=layout_h(150))
        self.feedbackToggle = widgets.ToggleButton(value=True,description='Feedback',layout=layout_h(150))
        self.locationToggle = widgets.ToggleButton(value=True,description='file location',layout=layout_h(150))
        self.depthSelection = widgets.Dropdown(value='full',options=['full',1,2,3,4,5],description='Depth:',tooltip='folder depth to display in location section of the image title',layout=layout_h(150))
        self.nameToggle = widgets.ToggleButton(value=True,description='Filename',layout=layout_h(150))
        self.dateToggle = widgets.ToggleButton(value=True,description='Date',layout=layout_h(150))
        # image filter settings
        self.filterLabel = widgets.Label(value='Image Filter Settings',layout=layout_h(150))
        self.gaussianToggle = widgets.ToggleButton(value=False,description='Gaussian',layout=layout_h(90))
        self.gaussianSize = widgets.BoundedIntText(value=2,min=0,max=10,step=1,tooltip='size of the gaussain kernel',layout=layout_h(60))
        self.medianToggle = widgets.ToggleButton(value=False,description='Median',layout=layout_h(90))
        self.medianSize = widgets.BoundedIntText(value=3,min=1,max=20,step=1,tooltip='size of the median kernel',layout=layout_h(60))
        self.laplacToggle = widgets.ToggleButton(value=False,description='Laplace',layout=layout_h(90))
        self.laplaceSize = widgets.BoundedIntText(value=3,min=3,max=10,step=1,tooltip='size of the laplace filter kernel',layout=layout_h(60))
        # plane fit settings ### needs interactive plot functionality (select 3 points) --> new implementation of plane subtraction function
        self.planeFitToggle = widgets.ToggleButton(value=False,description='Plane Fit',tooltip='plane subtraction')

        # layouts
        self.h_selection_btn_layout = HBox(children=[self.refreshBtn,self.previousBtn,self.nextBtn,self.saveBtn,self.copyBtn,self.titleOptionBtn])
        self.h_process_btn_layout = HBox(children=[self.fixZeroBtn,self.linebylineBtn,self.flattenBtn,self.invertBtn])
        self.v_text_layout = VBox(children=[self.channelSelect,self.saveNote])
        self.v_btn_layout = VBox(children=(self.h_selection_btn_layout,self.h_process_btn_layout))
        self.v_color_layout = VBox(children=(self.cmapSelection,self.vmin,self.vmax))
        self.h_user_layout = HBox(children=[self.v_text_layout,self.v_btn_layout,self.v_color_layout])
        self.v_file_layout = VBox(children=[self.directorySelection,self.selectionList])
        self.v_settings_layout = VBox(children=[self.titleLabel,
                                             self.titleToggle,
                                             self.channelToggle,
                                             self.setpointToggle,
                                             self.feedbackToggle,
                                             self.locationToggle,
                                             self.depthSelection,
                                             self.nameToggle,
                                             self.dateToggle,
                                             self.filterLabel,
                                             HBox(children=[self.gaussianToggle,self.gaussianSize],layout=layout_h(150)),
                                             HBox(children=[self.medianToggle,self.medianSize],layout=layout_h(150)),
                                             HBox(children=[self.laplacToggle,self.laplaceSize],layout=layout_h(150))],layout=layout_h(180))
        self.v_image_layout = VBox(children=[self.figure_display,self.h_user_layout])
        self.mainlayout = HBox(children=[self.v_file_layout,self.v_image_layout,self.v_settings_layout])

        # connect widgets to functions
        for child in self.v_settings_layout.children:
            if type(child) == type(self.v_settings_layout):
                for ch in child.children:
                    ch.observe(self.handler_settingsChange,names='value')
            child.observe(self.handler_settingsChange,names='value')
        self.nextBtn.on_click(self.nextDisplay)
        self.previousBtn.on_click(self.previousDisplay)
        self.saveBtn.on_click(self.save_figure)
        self.refreshBtn.on_click(self.handler_folder_selection)
        self.linebylineBtn.observe(self.redraw_image,names='value')
        self.flattenBtn.observe(self.redraw_image,names='value')
        self.invertBtn.observe(self.redraw_image,names='value')
        self.fixZeroBtn.observe(self.redraw_image,names='value')
        self.edgesBtn.observe(self.redraw_image,names='value')
        self.gaussianBtn.observe(self.redraw_image,names='value')
        self.directorySelection.observe(self.handler_folder_selection,names=['value'])
        self.selectionList.observe(self.handler_file_selection,names=['value'])
        self.channelSelect.observe(self.handler_channel_selection,names=['value'])
        self.cmapSelection.observe(self.updateDisplayImage,names='value')
        self.vmin.observe(self.updateDisplayImage,names='value')
        self.vmax.observe(self.updateDisplayImage,names='value')
        self.figure.canvas.mpl_connect('button_press_event',self.mouse_click)
        self.copyBtn.on_click(self.copy_figure)
        self.titleOptionBtn.observe(self.handler_titleOptionsDisplay,names='value')

        self.display()
        with self.figure_display:
            plt.show(self.figure)
        self.find_directories(self.active_dir)
        self.update_directories()
        #self.updateInfoText()
        #self.handler_file_selection('startup')
        #self.updateErrorText('finish startup')
    # show browser
    def display(self):
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
    def update_directories(self):
        display_directories = ['\\'.join(str(directory).split('\\')[-1:]) for directory in self.directories]
        display_directories[0] = f'(active){display_directories[0]}'
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
        ix = event.xdata
        iy = event.ydata
        d = np.sqrt(ix**2+iy**2)

        self.updateErrorText(str(round(event.xdata,2)) + ' ' + str(round(event.ydata,2)))
    # image generation
    def redraw_image(self,a):
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
        direction = 'forward' # add toggle switch to choose forward and backward
        flatten = self.flattenBtn.value
        offset = False # look into how this works
        zero = False # look into how this works
        try:
            self.image_data,unit = self.img.get_channel(channel, direction = direction, flatten=flatten, offset=offset,zero=zero)
        except:
            self.updateErrorText('Error in flattening routine: setting flatten=False')
            self.image_data,unit = self.img.get_channel(channel, direction = direction, flatten=False, offset=offset,zero=zero)
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
            self.image_data = filters.laplace(self.image_data,ksize=self.laplaceSize.value)

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
            #label.append(f'channel: {self.channelSelect.value}')
            if self.channelToggle.value:
                if self.feedbackToggle.value:
                    label.append(f'channel: {self.channelSelect.value} $\\rightarrow$ {mode}')
                else:
                    label.append(f'channel: {self.channelSelect.value}')
            if self.setpointToggle.value:
                label.append(f'setpoint: I = {set_point[0]:.0f}{set_point[1]}, V = {bias[0]:.2f}{bias[1]}')
            #label.append('I = %.0f%s' % set_point)  
            #label.append('bias = %.2f%s' % bias)
            #label.append('size: %.1f%s x %.1f%s (%.0f%s)' % (width+height+angle))
            #label.append('comment: %s' % comment)
            if self.locationToggle.value:
                location = self.directories[self.directorySelection.index]
                if self.depthSelection.value == 'full':
                    location = location
                else:
                    location = '\\'.join(str(location).split('\\')[-int(self.depthSelection.value):])
                label.append(f'location: {location}') #dat files use header['Saved Date']
            if self.nameToggle.value:
                label.append(f'filename: {self.img.name}')
            if self.dateToggle.value:
                label.append(f'Date: {self.img.header["rec_date"]} {self.img.header["rec_time"]}')
        #label.append('path: %s' % self.img.path)
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
        self.cb.set_label(f'{self.channelSelect.value} ({self.image_info["unit"]})',fontsize=self.fontsize)
        ax.set_title(self.scan_info,fontsize=self.titlesize,loc='left')
        ax.set_xlabel('x (nm)',fontsize=self.fontsize)
        ax.set_ylabel('y (nm)',fontsize=self.fontsize)
        ax.set_xticks([0,w])
        ax.set_yticks([0,h])
        ax.set_xticklabels([0,round(w,2)],fontsize=self.fontsize)
        ax.set_yticklabels([0,round(h,2)],fontsize=self.fontsize)
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

### Selection update
    def handler_settingsChange(self,a):
        self.redraw_image(a)

    def handler_titleOptionsDisplay(self,a):
        if self.titleOptionBtn.value:
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
        self.selectionList.options = self.sxm_files
        if len(self.sxm_files) != 0:
            self.filenameText.value = self.sxm_files[index]
            if self.filenameText.value in self.selectionList.options:
                self.selectionList.value = self.filenameText.value
    def handler_file_selection(self,update):
        #self.updateErrorText(str(update))
        self.image_index = self.all_files.index(self.selectionList.value)
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
