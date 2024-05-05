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
class spectrumBrowser():
    '''
    Info:
        figure = specBrowser.figure
        axes = specBrowser.axes
        spec = specBrowser.spec --> list of spm objects selected in the browser

        - specBrowser.update_axes() can be used to refresh the browser plot
        - spec can be accessed for further analysis/processing
        - axes can be accessed for futher plot modification (axis limits, labels, etc)
    '''
    def __init__(self,figsize=(8,8),fontsize=12,titlesize=12,cmap='Greys_r',home_directory='./',sxmBrowser=None):
        self.img = None
        self.figure,self.axes = plt.subplots(ncols=1,figsize=figsize,num='dat') # simple default figure size
        #self.wfFigure,self.wfAxes = plt.subplots(ncols=1,figsize=figsize)
        self.sxmBrowser = sxmBrowser
        if sxmBrowser == None:
            self.referenceLocBtn.disabled(True)
        self.fontsize = fontsize
        self.titlesize = titlesize
        self.spec_x = [np.linspace(-2,2,64)]
        self.spec_data = [np.zeros(64)] # 64 x 64 pixel zeros
        self.spec_info = [{'x_unit':'N','y_unit':'a.u.','x_label':'Index'}]
        self.spec_label = ''
        self.labels = []
        self.errors = []
        self.spec_index = [0]
        self.axes.plot(self.spec_x[0],self.spec_data[0])
        self.active_dir = Path(home_directory)
        self.sxm_files = []
        self.dat_files = []
        self.directories = [self.active_dir]
        #self.update_directories()

        # widget layouts
        smallLayout = widgets.Layout(visibility='visible',width='80px')
        mediumLayout = widgets.Layout(visibility='visible',width='120px')
        largeLayout = widgets.Layout(visibility='visible',width='160px')
        extraLargeLayout = widgets.Layout(visibility='visible',width='200px')
        layout = lambda x: widgets.Layout(visibility='visible',width=f'{x}px')
        # selections
        #self.rootSelection = Btn_Widget('Open',disabled=True)
        self.directorySelection = Selection_Widget(self.directories,'Folders:',rows=5)
        self.selectionList = widgets.SelectMultiple(options=self.dat_files,value=[],description='DAT Files:',rows=30)
        self.channelXSelect = widgets.Dropdown(options=['V'],value='V',description='X:')
        self.channelYSelect = widgets.SelectMultiple(options=['I'],value=['I'],description='Y:',rows=3)
        #self.channelYSelect.add_class("left-spacing-class")
        #display(HTML("<style>.left-spacing-class {margin-left: 10px;}</style>"))

        self.refreshBtn = Btn_Widget('',icon='refresh',tooltip='Reload file list',layout=layout(30))
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
        self.copyBtn = Btn_Widget('',layout=layout(30),icon='clipboard',tooltip='Save displayed image to \\browser_output folder\ncopy displayed image to clipboard')
        self.csvBtn = Btn_Widget('',layout=layout(30),icon='list-ul',tooltip='Save displayed data to \\browser_output folder\ndata saved as .csv')
        self.generateWaterFallBtn = Btn_Widget('Waterfall',disabled=True)
        self.figure_display = widgets.Output()
        self.legendBtn = widgets.ToggleButton(description='',value=True,layout=layout(30),icon='tags',tooltip='Toggle display of the plot legend\nDefault is active')
        #self.wfFigure_display = widgets.Output()
        #self.figure_tabs = widgets.Tab(children=[self.figure_display])
        #self.figure_tabs.titles = ['Line Spectrum', 'Waterfall']

        # analyis
        self.specRefBtn = widgets.ToggleButton(description='Reference',value=False,layout=mediumLayout)
        self.specRefSelect = widgets.Dropdown(description='spec:',options=[None]+list(self.selectionList.options),value=None,layout=largeLayout)
        # offset options
        self.offsetBtn = widgets.ToggleButton(description='',value=False,layout=layout(30),icon='navicon',tooltip='Apply vertical offset')
        self.offset_value = widgets.FloatText(value=0.1e-12,description='offset:',step=.1e-12,readout_format='.1e',layout=largeLayout)
        # colormap
        self.cmapSelection = widgets.Dropdown(description='colormap:',options=plt.colormaps(),value=cmap,layout=layout(200))
        # smoothing options
        self.smoothBtn = widgets.ToggleButton(description='',value=False,layout=layout(30),icon='filter',tooltip='Apply savitzky-golay filter to plot data')
        self.windowParam = widgets.BoundedIntText(description='window:',value=3,min=3,max=101,step=2,layout=largeLayout)
        self.orderParam = widgets.BoundedIntText(description='order:',value=1,min=1,max=5,step=1,layout=largeLayout)

        # layouts
        self.v_text_layout = VBox(children=[self.saveNote,self.errorText])
        self.h_process_layout = HBox(children=[self.flattenBtn,self.fixZeroBtn,self.referenceLocBtn,self.plot2DBtn,self.offsetBtn,self.smoothBtn])
        self.h_selection_btn_layout = HBox(children=[self.refreshBtn,self.csvBtn,self.saveBtn,self.copyBtn,self.legendBtn])
        self.v_param_layout = VBox(children=[self.offset_value,self.windowParam,self.orderParam])
        self.v_channel_layout = VBox(children=[self.channelXSelect,self.channelYSelect])
        self.v_file_select_layout = VBox(children=[self.directorySelection,self.selectionList,self.v_channel_layout])
        
        self.v_btn_layout = VBox(children=[self.h_selection_btn_layout,self.h_process_layout,self.cmapSelection])
        self.h_user_layout = HBox(children=[self.v_text_layout,self.v_btn_layout,self.v_param_layout])

        self.v_image_layout = VBox(children=[self.figure_display,self.h_user_layout])
        self.h_main_layout = HBox(children=[self.v_file_select_layout,self.v_image_layout])

        # connect widgets to functions
        #self.rootSelection.on_click(self.open_project)
        self.saveBtn.on_click(self.save_figure)
        self.copyBtn.on_click(self.copy_figure)
        self.csvBtn.on_click(self.save_data)
        self.generateWaterFallBtn.on_click(self.generateWaterFall)
        self.refreshBtn.on_click(self.handler_folder_selection)
        self.flattenBtn.observe(self.redraw_image,names='value')
        self.legendBtn.observe(self.redraw_image,names='value')
        self.referenceLocBtn.on_click(self.plotSpectrumLocations)
        self.plot2DBtn.on_click(self.plot2D)
        self.fixZeroBtn.observe(self.redraw_image,names='value')

        self.directorySelection.observe(self.handler_folder_selection,names=['value'])
        self.selectionList.observe(self.handler_file_selection,names=['value'])
        self.channelXSelect.observe(self.handler_channel_selection,names='value')
        self.channelYSelect.observe(self.handler_channel_selection,names=['value'])

        self.smoothBtn.observe(self.handler_update_axes,names='value')
        self.windowParam.observe(self.handler_update_axes,names='value')
        self.orderParam.observe(self.handler_update_axes,names='value')
        self.offsetBtn.observe(self.handler_update_axes,names='value')
        self.offset_value.observe(self.handler_update_axes,names='value')
        self.specRefBtn.observe(self.handler_update_axes,names='value')
        self.specRefSelect.observe(self.changeReferenceSelection,names='value')
        self.cmapSelection.observe(self.handler_update_axes,names='value')

        self.display()
        with self.figure_display:
            plt.show(self.figure)
        #with self.wfFigure_display:
            #plt.show(self.wfFigure)
        self.find_directories(self.active_dir)
        self.update_directories()
    # show browser
    def display(self):
        display.display(self.h_main_layout)
    
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
    # output functions
    def save_figure(self,a):
        self.saveBtn.icon = 'hourglass-start'
        if os.path.exists(f'{self.active_dir}/browser_outputs'):
            pass
        else:
            os.mkdir(f'{self.active_dir}/browser_outputs')
        fname = f'{self.active_dir}/browser_outputs/{self.directorySelection.value}_{self.spec[0].name.split(".")[0]}_{self.channelYSelect.value[0]}'
        if self.saveNote.value != '':
            fname += f'_{self.saveNote.value}'
        if a.description == 'Waterfall':
            self.wfFigure.savefig(f'{fname}_waterfall.png',dpi=500,format='png',transparent=True,bbox_inches='tight')
        else:
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
        self.saveBtn.icon = 'hourglass-start'
        if os.path.exists(f'{self.active_dir}/browser_outputs'):
            pass
        else:
            os.mkdir(f'{self.active_dir}/browser_outputs')
        fname = f'{self.active_dir}/browser_outputs/{self.directorySelection.value}_{self.spec[0].name.split(".")[0]}_{self.channelYSelect.value[0]}'
        if self.saveNote.value != '':
            fname += f'_{self.saveNote.value}'

        header = ""
        for label in self.labels:
            header += f"{self.spec_info[0]['x_label']} ({self.spec_info[0]['x_unit']}), {label[:-4]},"
        #header = header.split(',')
        data = zip(self.spec_x,self.spec_data)
        out = []
        [[out.append(d) for d in D] for D in data]
        #for i in range(len(out)):
        #    new = [header[i]]
        #    [new.append(o) for o in out[i]]
        #    out[i] = new
        out = np.transpose(out)
        np.savetxt(f'{fname}.csv',out,delimiter=',',header=header)
        self.updateErrorText('Saved CSV')
        self.csvBtn.icon = 'list-ul'
    # image generation
    def redraw_image(self,a):
        self.update_image_data()
        self.updateDisplayImage()
    def load_new_image(self,filename=None):
        #self.updateErrorText('load new image')
        directory = self.directorySelection.value
        if directory != self.active_dir:
            directory = os.path.join(self.active_dir,directory)
        if filename == None:
            files = [os.path.join(directory,self.dat_files[index]) for index in self.spec_index]
            self.spec = [Spm(f) for f in files]
            self.filenameText.value = ''.join([f'{self.all_files[self.spec_index[i]]},' for i in range(len(self.spec_index))])
            self.updateChannelSelection()
            self.update_image_data()
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
        if self.specRefBtn.value == True:
            self.handler_update_axes(a)
        else:
            pass
    def update_image_data(self,filename=None):
        #self.updateErrorText('update image data')
        channelX = self.channelXSelect.value
        channelY = self.channelYSelect.value
        if filename == None:
            self.spec_data = []
            self.spec_info = []
            self.labels = []
            self.spec_x = []
            if len(self.selectionList.value) >= 1 and len(channelY) == 1:
                for spec in self.spec:
                    spec_data,yunit = spec.get_channel(channelY[0])
                    self.spec_data.append(spec_data)
                    spec_x,xunit = spec.get_channel(channelX)
                    self.spec_info.append({'x_unit':xunit,'y_unit':yunit,'x_label':channelX})
                    self.spec_x.append(spec_x)
                    self.labels.append(spec.name)
            if len(self.selectionList.value) == 1 and len(channelY) > 1:
                spec = self.spec[0]
                for i in range(len(channelY)):
                    spec_data,yunit = spec.get_channel(channelY[i])
                    self.spec_data.append(spec_data)
                    spec_x,xunit = spec.get_channel(channelX)
                    self.spec_info.append({'x_unit':xunit,'y_unit':yunit,'x_label':channelX})
                    self.spec_x.append(spec_x)
                    self.labels.append(channelY[i])
            self.update_scan_info()
        else:
            spec = self.load_new_image(filename=filename)
            spec_data,yunit = spec.get_channel(channelY[0])
            spec_x,xunit = spec.get_channel(channelX)

            return spec, spec_data, spec_x
        #self.updateErrorText('finish update image data')
    def update_scan_info(self):
        #self.updateErrorText('update scan info')
        experiments = [spec.header['Experiment'] for spec in self.spec]
        assert experiments.count(experiments[0]) == len(experiments), 'Please ensure all selections are the same measurement type'
        spec = self.spec[0]
        label = []
        experiment = experiments[0]
        label.append(f'Experiment: {experiment} $\\rightarrow$ filename: {spec.name}')
        if len(self.selectionList.value) > 1:
            if self.smoothBtn.value:
                label.append(f'Savitzky-Golay Filter $\\rightarrow$ Window: {self.windowParam.value}, Order: {self.orderParam.value}')
            self.spec_label = '\n'.join(label)
            return
        if 'STML' in experiment:
            fb_enable = spec.get_param('Z-Ctrl hold')
            set_point = spec.get_param('setpoint_spec')
            bias = spec.get_param('V_spec')
            if np.abs(bias[0])<0.1:
                bias = list(bias)
                bias[0] = bias[0]*1000
                bias[1] = 'mV'
                bias = tuple(bias)
            if fb_enable == 'FALSE':
                label.append('feedback on')
            elif fb_enable == 'TRUE':
                label.append('feedback off')
            label.append(f'Exposure Time (s): {int(spec.header["Spectrometer Exposure Time (ms)"])/1000}, $\lambda_c$: {spec.header["Spectrometer Selected Grating Center Wavelength (nm)"]}, grating: {spec.header["Spectrometer Selected Grating Density"]}')
            label.append('setpoint: I = %.0f%s, V = %.1f%s' % (set_point+bias))    
        if 'bias spectroscopy' in experiment:
            fb_enable = spec.get_param('Z-Ctrl hold')
            set_point = spec.get_param('setpoint_spec')
            bias = spec.get_param('V_spec')
            #lockin_status = self.get_param('Lock-in>Lock-in status')
            lockin_amplitude = float(spec.header['Lock-in>Amplitude'])*1e3
            lockin_phase= float(spec.header['Lock-in>Reference phase D1 (deg)'])
            lockin_frequency= float(spec.header['Lock-in>Frequency (Hz)'])
            if np.abs(bias[0])<0.1:
                bias = list(bias)
                bias[0] = bias[0]*1000
                bias[1] = 'mV'
                bias = tuple(bias)
            #if lockin_status == 'ON':
            label.append(f'lockin: A = {lockin_amplitude:.0f} mV, θ = {lockin_phase:.1f} deg, f = {lockin_frequency:.0f} Hz')
            if fb_enable == 'FALSE':
                label.append('feedback on')
            elif fb_enable == 'TRUE':
                label.append('feedback off')
            label.append('setpoint: I = %.0f%s, V = %.1f%s' % (set_point+bias))    
        if 'THz amplitude sweep' in experiment:
            label.append(f'Laser Rep. Rate: {spec.header["Ext. VI 1>Laser>PP Frequency (MHz)"]}')
            label.append(f'Pulse Polarity: THz1;{spec.header["Ext. VI 1>THzPolarity>THz1"]}, THz2;{spec.header["Ext. VI 1>THzPolarity>THz2"]}')
            label.append(f'Delay Positions: THz1;{spec.header["Ext. VI 1>Position>PP1 (m)"]}, THz2;{spec.header["Ext. VI 1>Position>PP2(m)"]}')
        if 'Z spectroscopy' in experiment:
            label.append(f'Spec Points: {len(self.spec_data)}')
            label.append(f'Integration time (s): {spec.header["Integration time (s)"]}')
            label.append(f'z-sweep (m): {spec.header["Z sweep distance (m)"]}')
        if 'History Data' in experiment:
            label.append(f'Bias (V): {spec.header["Bias>Bias (V)"]}')
            label.append(f'Feedback: {spec.header["Z-Controller>Controller status"]}')
            label.append(f'Sample Period (ms): {spec.header["Sample Period (ms)"]}')
        else:
            pass
        label.append(f'location: {self.directorySelection.value}')
        label.append(f'Date: {spec.header["Saved Date"]}')
        if self.smoothBtn.value:
            label.append(f'Savitzky-Golay Filter $\\rightarrow$ Window: {self.windowParam.value}, Order: {self.orderParam.value}')
        #label.append('comment: %s' % comment)
        self.spec_label = '\n'.join(label)
        #self.updateErrorText('finish update scan info')
    def update_axes(self):
        #self.updateErrorText('update axes')
        if not self.figure:
            self.updateErrorText('making new figure/axes')
            self.figure,self.axes = plt.subplots(ncols=1,figsize=(8,8))
            #self.figure.tight_layout(pad=2)
        ax = self.axes
        ax.clear()
        colors = plt.cm.get_cmap(str(self.cmapSelection.value))(np.linspace(0,1,len(self.spec_data)))
        #print(len(self.spec_data),len(self.labels))
        if self.specRefBtn.value and self.specRefSelect.value != None:
            rSpec,rSpec_data,rSpec_x = self.update_image_data(filename=self.specRefSelect.value)
        for i in range(len(self.spec_data)):
            y_values = self.spec_data[i]
            offset = 0
            if self.fixZeroBtn.value:
                offset = np.mean(self.spec_data[i][np.where(abs(self.spec_x[i])<0.1)[0]])
            y_values = self.smooth_data(self.spec_data[i]-offset)
            if self.flattenBtn.value:
                y_values = y_values / np.max(y_values)
            if self.offsetBtn.value:
                y_values = y_values + i*self.offset_value.value
            if self.specRefBtn.value and self.specRefSelect.value != None:
                y_values = y_values / rSpec_data
            ax.plot(self.spec_x[i],y_values,color=colors[i],label=self.labels[i])
        if self.specRefBtn.value and self.specRefSelect.value != None:
            ax.plot(rSpec_x,rSpec_data/np.max(rSpec_data)-1,color='grey',label='reference')
        ax.set_title(self.spec_label,fontsize=self.titlesize,loc='left')
        ax.set_xlabel(f'{self.spec_info[0]["x_label"]} ({self.spec_info[0]["x_unit"]})',fontsize=self.fontsize)
        ax.set_ylabel(f'{self.channelYSelect.value[0]} ({self.spec_info[0]["y_unit"]})',fontsize=self.fontsize)
        if self.legendBtn.value:
            if i > 3:
                ax.legend(bbox_to_anchor=(1.01, 1))
            else:
                ax.legend()
        else:
            pass
        self.figure.tight_layout(pad=2)
        #self.updateErrorText('finish update axes')

### alternate functions
    def generateWaterFall(self):
        positions = []
        distances = []
        for spec in self.spec:
            positions.append([spec.get_param('x')[0],spec.get_param('y')[0]])
        x_0,y_0 = positions[0]
        for p in positions:
            x_a = p[0]-x_0
            y_a = p[1]-y_0
            distances.append(np.sqrt(x_a**2 + y_a**2))
        dataMin = np.min(self.spec_data)
        dataMax = np.max(self.spec_data)
        xMin = distances[0]
        xMax = distances[-1]
        yMin = self.spec_x[0].min()
        yMax = self.spec_x[0].max()
        self.wfAxes.imshow(np.rot90(self.spec_data),extent=[xMin,xMax,yMin,yMax],aspect='auto',cmap=self.cmapSelection.value)
        self.wfAxes.set_xlabel('distance (nm)')
        self.wfAxes.set_ylabel('Bias (V)')
        self.wfAxes.set_title('dI/dV')
        self.save_figure(self.generateWaterFallBtn)
    def relative_position(self,img,spec,**params):
        #width = ref.get_param('width')
        #height = ref.get_param('height')
        #[px_x,px_y] = ref.get_param('scan_pixels')
        [o_x,o_y] = img.get_param('scan_offset')
        width = img.get_param('width')[0]
        height = img.get_param('height')[0]
        [o_x,o_y] = [o_x*10**9,o_y*10**9]
        angle = float(img.get_param('scan_angle'))*-1* np.pi/180
        x_spec = spec.get_param('x')[0]
        y_spec = spec.get_param('y')[0]
        if angle != 0:
            #Transforming to relative coordinates with angle
            x_rel = (x_spec-o_x)*np.cos(angle) + (y_spec-o_y)*np.sin(angle)+width/2
            y_rel = -(x_spec-o_x)*np.sin(angle) + (y_spec-o_y)*np.cos(angle)+height/2
        else:
            x_rel = x_spec-o_x+width/2
            y_rel = y_spec-o_y+height/2
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
            colors = plt.cm.jet(np.linspace(0,1,len(self.spec)))
            colors = plt.cm.get_cmap(self.cmapSelection.value)(np.linspace(0,1,len(self.spec)))
            for i,spec in enumerate(self.spec):
                rx,ry = self.relative_position(self.sxmBrowser.img,spec)
                rel_positions.append([rx,ry])
                ax.plot(rx,ry,marker='o',markersize=10,color=colors[i],label=self.labels[i].split('.')[0][-3:])
            if self.specRefBtn.value and self.specRefSelect.value != None:
                pass
    def plot2D(self,a):
        if len(self.spec) != 0:
            dataPoints = max([len(spec_data) for spec_data in self.spec_data])
            xData = self.spec_x[[len(spec_data) for spec_data in self.spec_data].index(dataPoints)]
            yLabels = [int(''.join([s for s in spec.name if s.isdigit()])) for spec in self.spec] #[int(name) for name in spec.name.split() if name.isdigit()]
            ymin = 0
            ymax = len(yLabels)
            ylabel = 'Spec Number'
            dataArray = np.zeros((len(self.spec),dataPoints))
            for i,spec_data in enumerate(self.spec_data):
                dataArray[i,:len(spec_data)] = spec_data
            vmin = np.min(dataArray)
            vmax = np.max(dataArray)
            if 'QtE' in self.active_dir:
                delayPositionsTHz1 = [float(spec.header["Ext. VI 1>Position>PP1 (m)"]) for spec in self.spec]
                delayPositionsTHz2 = [float(spec.header["Ext. VI 1>Position>PP2(m)"]) for spec in self.spec]
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
            self.axes.set_xlabel(f'{self.spec_info[0]["x_label"]} ({self.spec_info[0]["x_unit"]})')
        else:
            pass
        
### display configuration
    def updateDisplayImage(self,*params):
        #self.updateErrorText('update display image')
        self.update_axes()
        #self.figure.canvas.set_window_title(self.img.name.split('.')[0])
        self.figure.canvas.draw()
        #self.updateErrorText('finish update display image')
    def updateInfoText(self):
        self.filenameText.value = self.dat_files[self.spec_index]
        self.selectionList.value = self.all_files[self.spec_index]
    def updateChannelSelection(self):
        current_value_X = self.channelXSelect.value
        current_value_Y = self.channelYSelect.value[0]
        self.channelXSelect.options = self.spec[0].channels
        self.channelYSelect.options = self.spec[0].channels
        if current_value_X in self.spec[0].channels:
            self.channelXSelect.value = current_value_X
        else:
            self.channelXSelect.value = self.spec[0].channels[0]
        if current_value_Y in self.spec[0].channels:
            self.channelYSelect.value = [current_value_Y]
        else:
            self.channelYSelect.value = [self.spec[0].channels[0]]
        #self.updateErrorText(self.channelSelect.value)
    def updateErrorText(self,text):
        self.errors.append(f'{len(self.errors)} {text}')
        self.errorText.options = self.errors

    def nextDisplay(self,a):
        if self.spec_index == len(self.all_files)-1:
            return
        else:
            self.spec_index += 1
            self.updateInfoText()
    def previousDisplay(self,a):
        if self.spec_index == 0:
            return
        else:
            self.spec_index -= 1
            self.updateInfoText()

### Selection update
    def handler_folder_selection(self,a):
        index=0
        if type(a) == type(self.refreshBtn): 
            index = self.selectionList.index
        directory = self.directorySelection.value
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
        self.selectionList.options = self.dat_files
        self.specRefSelect.options = [None]+list(self.dat_files)
        if type(index) == int:
            self.filenameText.value = self.dat_files[index]
        else:
            self.filenameText.value = self.dat_files[index[0]]
        if self.filenameText.value in self.selectionList.options:
            self.selectionList.value = [self.filenameText.value]
        #print(self.dat_files)
    def handler_file_selection(self,update:object):
        #self.updateErrorText(str(update))
        self.spec_index = [self.dat_files.index(value) for value in self.selectionList.value]#self.all_files.index(self.selectionList.value)
        try:
            self.load_new_image()
            self.updateDisplayImage()
        except Exception as err:
            self.updateErrorText('file selection error:' + str(err))
            print(traceback.format_exc())
    def handler_channel_selection(self,update):
        try:
            #channel_number = self.spec[0].channels.index(self.channelSelect.value[0])
            self.update_image_data()
            self.updateDisplayImage()
        except Exception as err:
            self.updateErrorText('channel selection error:' + str(err))
### data presentation update
    def handler_update_axes(self,a):
        self.update_scan_info()
        self.update_axes()
### misc
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