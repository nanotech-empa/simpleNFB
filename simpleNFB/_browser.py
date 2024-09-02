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
#from IPython.core import html
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.signal import savgol_filter
import traceback
import subprocess
from tkinter import Tk
from tkinter import filedialog
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
class _Browser():
    '''
    Info:
        figure = _Browser.figure
        axes = _Browser.axes

        - _Browser.update_axes() can be used to refresh the browser plot
        - axes can be accessed for futher plot modification (axis limits, labels, etc)
    '''
    def __init__(self,figsize=(8,8),fontsize=12,titlesize=12,viewerType='browser',cmap='Greys_r',home_directory='./',sxmBrowser=None):
        # label == 'browser' or 'dat' or 'sxm'
        self.viewerType = viewerType

        self.figure,self.axes = plt.subplots(ncols=1,figsize=figsize,num=viewerType) # simple default figure size

        self.img = None
        self.colormap_str = cmap
        self.fontsize = fontsize
        self.titlesize = titlesize
        self.labels = []
        self.errors = []
        self.active_dir = Path(home_directory)
        self.spm_files = []
        self.files = []
        self.directories = [self.active_dir]
        #self.update_directories()
        self.create_gui_objects()
        #self.sxmBrowser = None
        #self.datBrowser = None
        #if sxmBrowser == None:
        #    self.referenceLocBtn.disabled(True)

        self.display()
        with self.figure_display:
            plt.show(self.figure)
        #with self.wfFigure_display:
            #plt.show(self.wfFigure)
        self.find_directories(self.active_dir)
        self.update_directories()

##### Create GUI elements
    def create_gui_objects(self):
        # widget layouts
        layout = lambda x: widgets.Layout(visibility='visible',width=f'{x}px')
        # selections
        self.refreshBtn = Btn_Widget('',icon='refresh',tooltip='Reload file list',layout=layout(30))
        self.directorySelection = Selection_Widget(self.directories,'Folders:',rows=5)
        self.selectionList = widgets.SelectMultiple(options=self.files,value=[],description='Files:',rows=30)
        self.channelXSelect = widgets.Dropdown(options=['V'],value='V',description='X:')
        self.channelSelect = widgets.SelectMultiple(options=['I'],value=['I'],description='Y:',rows=3)

        # text display
        self.filenameText = Text_Widget('')
        self.indexText = Text_Widget('0')
        self.errorText = Selection_Widget([],'Out:',rows=5)
        self.saveNote = Text_Widget('',description='note:')
        # image display
        self.nextBtn = Btn_Widget('',layout=layout(30),icon='arrow-circle-down',tooltip='Load next file in list')
        self.previousBtn = Btn_Widget('',layout=layout(30),icon='arrow-circle-up',tooltip='Load previous file in list')
        self.flattenBtn = widgets.ToggleButton(description='',value=False,layout=layout(30),icon='barcode',tooltip='Normalize the data\nEach curve is rescaled to a maximum value of 1.0')
        self.invertBtn = widgets.ToggleButton(description='',value=False,layout=layout(30),icon='exchange',tooltip='Invert horizonatl direction of the data')
        self.fixZeroBtn = widgets.ToggleButton(description='',value=False,layout=layout(30),icon='neuter',tooltip='Remove baseline of the data\nSubstracts average of the data')
        self.referenceLocBtn = Btn_Widget('',layout=layout(30),icon='map-marker',tooltip='Plot tip location on image browser figure')
        self.plot2DBtn = Btn_Widget('',layout=layout(30),icon='area-chart',tooltip='Convert plot from 1D to 2D\nSpectrum names are shown on the vertical axis')
        self.saveBtn = Btn_Widget('',layout=layout(30),icon='file-image-o',tooltip='Save displayed image to \\browser_output folder\nText in the "note" is appended to figure filename')
        self.copyBtn = Btn_Widget('',layout=layout(30),icon='clipboard',tooltip='copy displayed image to clipboard\nSave displayed image to \\browser_output folder')
        self.csvBtn = Btn_Widget('',layout=layout(30),icon='file-excel-o',tooltip='Save displayed data to \\browser_output folder\ndata saved as .csv')
        self.generateWaterFallBtn = Btn_Widget('Waterfall',disabled=True)
        self.figure_display = widgets.Output()
        self.legendBtn = widgets.ToggleButton(description='',value=True,layout=layout(30),icon='tags',tooltip='Toggle display of the plot legend\nDefault is active')


        # analyis
        self.fileRefBtn = widgets.ToggleButton(description='Reference',value=False,layout=layout(120))
        self.fileRefSelect = widgets.Dropdown(description='file:',options=[None]+list(self.selectionList.options),value=None,layout=layout(160))
        # offset options
        self.offsetBtn = widgets.ToggleButton(description='',value=False,layout=layout(30),icon='navicon',tooltip='Apply vertical offset')
        self.offset_value = widgets.FloatText(value=0.1e-12,description='offset:',step=.1e-12,readout_format='.1e',layout=layout(160))
        # colormap
        self.cmapSelection = widgets.Dropdown(description='colormap:',options=plt.colormaps(),value=self.colormap_str,layout=layout(200))
        # smoothing options
        self.smoothBtn = widgets.ToggleButton(description='',value=False,layout=layout(30),icon='filter',tooltip='Apply savitzky-golay filter to plot data')
        self.windowParam = widgets.BoundedIntText(description='window:',value=3,min=3,max=101,step=2,layout=layout(160))
        self.orderParam = widgets.BoundedIntText(description='order:',value=1,min=1,max=5,step=1,layout=layout(160))

        # layouts
        self.v_text_layout = VBox(children=[self.saveNote,self.errorText])
        self.h_process_layout = HBox(children=[self.flattenBtn,self.fixZeroBtn,self.referenceLocBtn,self.plot2DBtn,self.offsetBtn,self.smoothBtn,self.legendBtn])
        self.h_selection_btn_layout = HBox(children=[self.refreshBtn,self.previousBtn,self.nextBtn,self.csvBtn,self.saveBtn,self.copyBtn])
        self.v_param_layout = VBox(children=[self.offset_value,self.windowParam,self.orderParam])
        self.v_channel_layout = VBox(children=[self.channelXSelect,self.channelSelect])
        self.v_file_select_layout = VBox(children=[self.directorySelection,self.selectionList,self.v_channel_layout])
        
        self.v_btn_layout = VBox(children=[self.h_selection_btn_layout,self.h_process_layout,self.cmapSelection])
        self.h_user_layout = HBox(children=[self.v_text_layout,self.v_btn_layout,self.v_param_layout])

        self.v_image_layout = VBox(children=[self.figure_display,self.h_user_layout])
        self.h_main_layout = HBox(children=[self.v_file_select_layout,self.v_image_layout])

        # connect widgets to functions
        #self.rootSelection.on_click(self.open_project)
        self.refreshBtn.on_click(self.handler_directory_changed)
        self.saveBtn.on_click(self.save_figure)
        self.copyBtn.on_click(self.copy_figure)
        self.csvBtn.on_click(self.save_data)
        self.generateWaterFallBtn.on_click(self.generateWaterFall)
        self.flattenBtn.observe(self.redraw_image,names='value')
        self.legendBtn.observe(self.redraw_image,names='value')
        self.referenceLocBtn.on_click(self.plotSpectrumLocations)
        self.plot2DBtn.on_click(self.plot2D)
        self.fixZeroBtn.observe(self.redraw_image,names='value')

        self.directorySelection.observe(self.handler_directory_changed,names=['value'])
        self.selectionList.observe(self.handler_file_changed,names=['value'])
        self.channelXSelect.observe(self.handler_channel_selection,names='value')
        self.channelSelect.observe(self.handler_channel_selection,names=['value'])

        self.smoothBtn.observe(self.handler_update_axes,names='value')
        self.windowParam.observe(self.handler_update_axes,names='value')
        self.orderParam.observe(self.handler_update_axes,names='value')
        self.offsetBtn.observe(self.handler_update_axes,names='value')
        self.offset_value.observe(self.handler_update_axes,names='value')
        self.fileRefBtn.observe(self.handler_update_axes,names='value')
        self.fileRefSelect.observe(self.changeReferenceSelection,names='value')
        self.cmapSelection.observe(self.handler_update_axes,names='value')
##### Handler Methods (GUI callbacks)
    def handler_directory_changed(self,a):
        index=0
        if type(a) == type(self.refreshBtn): 
            index = self.selectionList.index
        directory = self.directorySelection.value
        self.spm_files = [file for file in os.listdir(directory) if self.is_spm(file)]
        self.sxm_files = [file for file in self.spm_files if '.sxm' in file]
        self.dat_files = [file for file in self.spm_files if '.dat' in file]
        self.files = {'sxm':self.sxm_files,'dat':self.dat_files,'browser':self.spm_files}[self.viewerType]
        self.selectionList.options = self.files
        self.fileRefSelect.options = [None]+list(self.files)
        if len(self.spm_files) == 0: return
        if type(self.selectionList) == type(widgets.Select):
            self.selectionList.value = self.selectionList.options[index]
        else:
            self.selectionList.value = [self.selectionList.options[index]]
    def handler_file_changed(self,update:object):
        if len(self.spm_files) == 0: return
        if self.viewerType == 'sxm' and len(self.selectionList.value) > 1:
            self.selectionList.index = [self.selectionList.index[0]]
        self.file_index = [self.files.index(value) for value in self.selectionList.value]
        try:
            self.load_new_data()
            self.updateDisplayData()
        except Exception as err:
            self.updateErrorText('file selection error:' + str(err))
            print(traceback.format_exc())
    def handler_channel_selection(self,update):
        try:
            self.update_data()
            self.updateDisplayData()
        except Exception as err:
            self.updateErrorText('channel selection error:' + str(err))
    def handler_update_axes(self,a):
        self.udpate_info()
        self.update_axes()
##### Directory and File selections
    def find_directories(self,_path):
        directories = []
        for _directory in os.listdir(_path):
            if os.path.isdir(_path / _directory):
                if 'browser_outputs' in _directory or 'ipynb' in _directory: continue
                directories.append(_path / _directory)
                self.find_directories(_path / _directory)
        else:
            pass
        self.directories.extend(directories)
        return directories
    def update_directories(self):
        self.directorySelection.options = self.directories

        # setup in sub-class
        pass
##### image generation
    def load_new_data(self,filename=None):
        #self.updateErrorText('load new image')
        directory = self.directorySelection.value
        if directory != self.active_dir:
            directory = os.path.join(self.active_dir,directory)
        if filename == None:
            files = [os.path.join(directory,self.files[index]) for index in self.file_index]
            self.file = [Spm(f) for f in files]
            self.updateChannelSelection()
            self.update_data()
        else:
            return Spm(os.path.join(directory,filename))
        #self.updateErrorText('finish load new image')
    def smooth_data(self,data):
        if self.smoothBtn.value:
            window = self.windowParam.value
            order = self.orderParam.value
            return savgol_filter(data,window,order)
        else:
            return data
    def changeReferenceSelection(self,a):
        if self.fileRefBtn.value == True:
            self.handler_update_axes(a)
        else:
            pass
    # load new data and update axes

    def redraw_image(self,a):
        self.update_data()
        self.updateDisplayData()
##### Update functions
    def update_data(self,filename=None):
        # setup in sub-class
        pass
    def udpate_info(self):
        # setup in sub-class
        pass
    def update_axes(self):
        # setup in sub-class
        pass
    def updateDisplayData(self,*params):
        self.update_axes()
        self.figure.canvas.draw()
    def updateChannelSelection(self):
        current_value_X = self.channelXSelect.value
        current_value_Y = self.channelSelect.value[0]
        self.channelXSelect.options = self.file[0].channels
        self.channelSelect.options = self.file[0].channels
        if current_value_X in self.file[0].channels:
            self.channelXSelect.value = current_value_X
        else:
            self.channelXSelect.value = self.file[0].channels[0]
        if current_value_Y in self.file[0].channels:
            self.channelSelect.value = [current_value_Y]
        else:
            self.channelSelect.value = [self.file[0].channels[0]]
### alternate functions
    def generateWaterFall(self):
        positions = []
        distances = []
        for file in self.file:
            positions.append([file.get_param('x')[0],file.get_param('y')[0]])
        x_0,y_0 = positions[0]
        for p in positions:
            x_a = p[0]-x_0
            y_a = p[1]-y_0
            distances.append(np.sqrt(x_a**2 + y_a**2))
        dataMin = np.min(self.file_data)
        dataMax = np.max(self.file_data)
        xMin = distances[0]
        xMax = distances[-1]
        yMin = self.file_x[0].min()
        yMax = self.file_x[0].max()
        self.wfAxes.imshow(np.rot90(self.file_data),extent=[xMin,xMax,yMin,yMax],aspect='auto',cmap=self.cmapSelection.value)
        self.wfAxes.set_xlabel('distance (nm)')
        self.wfAxes.set_ylabel('Bias (V)')
        self.wfAxes.set_title('dI/dV')
        self.save_figure(self.generateWaterFallBtn)
    def relative_position(self,img,file,**params):
        [o_x,o_y] = img.get_param('scan_offset')
        width = img.get_param('width')[0]
        height = img.get_param('height')[0]
        [o_x,o_y] = [o_x*10**9,o_y*10**9]
        angle = float(img.get_param('scan_angle'))*-1* np.pi/180
        x_file = file.get_param('x')[0]
        y_file = file.get_param('y')[0]
        if angle != 0:
            #Transforming to relative coordinates with angle
            x_rel = (x_file-o_x)*np.cos(angle) + (y_file-o_y)*np.sin(angle)+width/2
            y_rel = -(x_file-o_x)*np.sin(angle) + (y_file-o_y)*np.cos(angle)+height/2
        else:
            x_rel = x_file-o_x+width/2
            y_rel = y_file-o_y+height/2
        return [x_rel,y_rel]
    def plotSpectrumLocations(self,a):
        fig = self.sxmBrowser.figure
        ax = self.sxmBrowser.axes
        if self.sxmBrowser.img == None: pass
        else:
            spx,spy = self.sxmBrowser.img.header['scan_pixels']
            ipy,ipx = self.sxmBrowser.image_data[~np.isnan(self.sxmBrowser.image_data).any(axis=1)].shape
            height = self.sxmBrowser.img.get_param('height')[0]
            rel_positions = []
            colors = plt.cm.jet(np.linspace(0,1,len(self.file)))
            colors = plt.cm.get_cmap(self.cmapSelection.value)(np.linspace(0,1,len(self.file)))
            for i,file in enumerate(self.file):
                rx,ry = self.relative_position(self.sxmBrowser.img,file)
                rel_positions.append([rx,ry])
                ax.plot(rx,ry,marker='o',markersize=10,color=colors[i],label=self.labels[i].split('.')[0][-3:])
            if self.fileRefBtn.value and self.fileRefSelect.value != None:
                pass
    def plot2D(self,a):
        if len(self.file) != 0:
            dataPoints = max([len(file_data) for file_data in self.file_data])
            xData = self.file_x[[len(file_data) for file_data in self.file_data].index(dataPoints)]
            yLabels = [int(''.join([s for s in file.name if s.isdigit()])) for file in self.file] #[int(name) for name in file.name.split() if name.isdigit()]
            ymin = 0
            ymax = len(yLabels)
            ylabel = 'file Number'
            dataArray = np.zeros((len(self.file),dataPoints))
            for i,file_data in enumerate(self.file_data):
                dataArray[i,:len(file_data)] = file_data
            vmin = np.min(dataArray)
            vmax = np.max(dataArray)
            if 'QtE' in self.active_dir:
                delayPositionsTHz1 = [float(file.header["Ext. VI 1>Position>PP1 (m)"]) for file in self.file]
                delayPositionsTHz2 = [float(file.header["Ext. VI 1>Position>PP2(m)"]) for file in self.file]
                if abs(sum(np.gradient(delayPositionsTHz1))) > abs(sum(np.gradient(delayPositionsTHz2))):
                    delayPositions = delayPositionsTHz1
                    ylabel = 'Delay THz1 (m)'
                else:
                    delayPositions = delayPositionsTHz2
                    ylabel = 'Delay THz2 (m)'
                ymin = delayPositions[0]
                ymax = delayPositions[-1]
                maxVal = np.max(np.abs(dataArray))
                vmin = -maxVal
                vmax = maxVal
            #print(yLabels)
            self.axes.clear()
            self.axes.imshow(dataArray,aspect='auto',origin='lower',extent=[xData[0],xData[-1],ymin,ymax],cmap=self.cmapSelection.value)
            if 'QtE' not in self.active_dir:
                self.axes.set_yticks(np.arange(len(yLabels))+.5)
                self.axes.set_yticklabels(yLabels)
            self.axes.set_ylabel(ylabel)
            self.axes.set_xlabel(f'{self.file_info[0]["x_label"]} ({self.file_info[0]["x_unit"]})')
        else:
            pass
        
### display configuration

        #self.updateErrorText(self.channelSelect.value)
    def updateErrorText(self,text):
        self.errors.append(f'{len(self.errors)} {text}')
        self.errorText.options = self.errors

    def nextDisplay(self,a):
        if self.file_index == len(self.files)-1:
            return
        else:
            self.file_index += 1
            self.selectionList.value = [self.files[self.file_index]]
    def previousDisplay(self,a):
        if self.file_index == 0:
            return
        else:
            self.file_index -= 1
            self.selectionList.value = [self.files[self.file_index]]

##### misc
    def display(self):
        display.display(self.h_main_layout)
    def is_spm(self,filename):
        spm_file = False
        if '.sxm' in filename or '.dat' in filename:
            spm_file = True
        for extension in ['png','jpeg','jpg','svg']:
            if extension in filename:
                spm_file = False
        return spm_file
    def make_figure(self,figsize=(7,5),cols=1):
        try:
            if self.figure:
                plt.close(self.figure)
        except Exception as err:
            print(err)
        self.figure,self.axes = plt.subplots(ncols=cols,figsize=figsize)
        if cols != 1:
            self.axs = self.axes[1:]
            self.axes = self.axes[0]
##### output functions
    def save_figure(self,a):
        self.saveBtn.icon = 'hourglass-start'
        if os.path.exists(self.active_dir / 'browser_outputs'):
            pass
        else:
            os.mkdir(self.active_dir / 'browser_outputs')
        fname = 'browser_outputs/' + str(self.directorySelection.value).split("\\")[-1] + f'_{self.file[0].name.split(".")[0]}_{self.channelSelect.value[0]}'
        if self.saveNote.value != '':
            fname += f'_{self.saveNote.value}'
        else:
            pass
        fname = self.active_dir / fname
        self.figure.savefig(f'{fname}.png',dpi=500,format='png',transparent=True,bbox_inches='tight')
        self.last_save_fname = f'{fname}.png'
        self.updateErrorText('Figure Saved')
        self.saveNote.value = ''
        self.saveBtn.icon = 'file-image-o'
    def copy_figure(self,a):
        self.save_figure(a)
        self.copyBtn.icon = 'hourglass-half'
        # Make powershell command
        powershell_command = r'$imageFilePaths = @("'
        for image_path in [self.last_save_fname]:
            powershell_command += image_path + '","'
        powershell_command = powershell_command[:-2] + '); '
        powershell_command += r'Set-Clipboard -Path $imageFilePaths;'
        # Execute Powershell
        completed = subprocess.run(["powershell", "-Command", powershell_command], capture_output=True)
        self.copyBtn.icon = 'clipboard'
    def save_data(self,a):
        # udpate in sub-class
        pass