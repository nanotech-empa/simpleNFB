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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import numpy as np
from scipy.signal import savgol_filter,medfilt
from scipy.interpolate import interp1d
import traceback
import subprocess
from tkinter import Tk
from tkinter import filedialog
from pathlib import Path
import os
import sys
import time
sys.path.append(r'./spmpy')
from spmpy import Spm

### matplotlib default settings ###
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['axes.linewidth'] = 0.8 # Previous Setting
mpl.rcParams['font.family'] = ['Microsoft Sans Serif']
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.labelpad'] = 3
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7
mpl.rcParams['axes.labelsize'] = 7
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
    def __init__(self,figsize=(3.5,2.8),fontsize=8,titlesize=5,cmap='Greys_r',home_directory='./',sxmBrowser=None):
        self.img = None
        self.figure,self.axes = plt.subplots(ncols=1,figsize=figsize,num='dat',dpi=150) # simple default figure size
        self.axes2 = None
        self.cb = None
        self.sxmBrowser = sxmBrowser
        if sxmBrowser == None:
            self.referenceLocBtn.disabled(True)
        self.fontsize = fontsize
        self.titlesize = titlesize
        self.spec_x = [np.linspace(-2,2,64)]
        self.spec_data = [np.zeros(64)] # 64 x 64 pixel zeros
        self.spec_info = [{'x_unit':'N','y_unit':'a.u.','x_label':'Index'}]
        self.spec_label = ''
        self.plasmonInfo = {'file':None,'spm':None,'interp':None}
        self.labels = []
        self.legendFontsize = [8,6,4,3]
        self.errors = []
        self.spec_index = [0]
        self.axes.plot(self.spec_x[0],self.spec_data[0])
        self.default_channel = {'STML':['Wavelength', 'Intensity'],'bias spectroscopy':['V','dIdV'],'Z spectroscopy':['zrel','I'],'History Data': ['Index','I']}
        self.loaded_experiments = None
        self.current_experiment = None
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
        layout_h = lambda x: widgets.Layout(visibility='hidden',width=f'{x}px')
        flex_layout = lambda x: widgets.Layout(display='flex',width=f'{x}%')
        flex_layout_btn = lambda x: widgets.Layout(display='flex',width=f'{x}%',align_items='center',justify_content='center')
        flex_layout_h = lambda x: widgets.Layout(visibility='hidden',display='flex',width=f'{x}%')
        
        ### selections ###
        self.rootFolder = widgets.Text(description='',layout=widgets.Layout(dispaly='flex',width='90%'))
        self.directorySelection = Selection_Widget(self.directories,'Folders:',rows=5)
        self.selectionList = widgets.SelectMultiple(options=self.dat_files,value=[],description='DAT Files:',rows=30)
        self.filterSelection = widgets.SelectMultiple(options=['all','dIdV','Z-Spectroscopy','stml','History'],value=['all'],description='Filter',rows=5)
        self.newFilterText = widgets.Text(description='New Filter',tooltip='user defined string to use for file filtering',layout=layout(200))
        self.addFilterBtn = widgets.Button(description='+',tooltip='click to add new filter to selection',layout=layout(30))
    
        self.channelXSelect = widgets.Dropdown(options=['Index'],value=None,description='X:')
        self.channelYSelect = widgets.SelectMultiple(options=[None],value=[None],description='Y:',rows=5)

        #self.channelXSelect = widgets.Dropdown(options=['V'],value='V',description='X:')
        #self.channelYSelect = widgets.SelectMultiple(options=['I'],value=['I'],description='Y:',rows=3)
        
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

        # offset options
        self.offsetBtn = widgets.ToggleButton(description='',value=False,layout=layout(30),icon='navicon',tooltip='Apply vertical offset')
        self.offset_value = widgets.FloatText(value=0.1e-12,description='offset:',step=.1e-12,readout_format='.1e',layout=largeLayout)
        # colormap
        self.cmapSelection = widgets.Dropdown(description='colormap:',options=plt.colormaps(),value=cmap,layout=layout(200))
        self.markerSelection = widgets.Dropdown(description='plotMarker:',options=['N','o','*','s','^','X'],value='o',layout=layout(200))

        # smoothing options
        self.smoothBtn = widgets.ToggleButton(description='',value=False,layout=layout(30),icon='filter',tooltip='Apply savitzky-golay filter to plot data')
        self.windowParam = widgets.BoundedIntText(description='window:',value=3,min=3,max=101,step=2,layout=largeLayout)
        self.orderParam = widgets.BoundedIntText(description='order:',value=1,min=1,max=5,step=1,layout=largeLayout)
        # settings menu toggle
        self.settingsBtn = widgets.ToggleButton(description='',icon='gear',value=False,tooltip='Display figure title options panel',layout=layout(30))
        # figure title options
        self.titleLabel = widgets.Label(value='Figure Title Settings',layout=layout(150))
        self.titleToggle = widgets.ToggleButton(value=True, description='Show Title',tooltip='Toggle figure title',layout=layout(150))
        self.setpointToggle = widgets.ToggleButton(value=True,description='Setpoint',layout=layout(150))
        self.feedbackToggle = widgets.ToggleButton(value=True,description='Feedback',layout=layout(150))
        self.locationToggle = widgets.ToggleButton(value=True,description='file location',layout=layout(150))
        self.depthSelection = widgets.Dropdown(value='full',options=['full',1,2,3,4,5],description='Depth:',tooltip='folder depth to display in location section of the image title',layout=layout(150),style={'description_width':'initial'})
        self.nameToggle = widgets.ToggleButton(value=True,description='Filename',layout=layout(150))
        self.dateToggle = widgets.ToggleButton(value=True,description='Date',layout=layout(150))
        # legend settings
        self.legendLabel = widgets.Label(value='Legend Settings',layout=layout_h(150))
        self.defaultLegendToggle = widgets.ToggleButton(value=True,description='default',tooltip='Toggle to enable the default Legend using filenames as labels',layout=layout(150))
        self.customLegendToggle = widgets.ToggleButton(value=False,description='custom',tooltip='Toggle to enable a custom Legend with user defined labels',layout=layout(150))
        self.legendText = widgets.Select(value='',options=[''],rows=5,layout=layout(150),disabled=False)
        # The above code is accessing the `legendEntry` attribute of the `self` object in Python.
        self.legendEntry = widgets.Text(description='',tooltip='enter new legend text here',layout=layout(150),disabled=False)
        self.legendToggle = widgets.ToggleButton(value=True,description='legend',layout=layout(74))
        self.legendUpdate = widgets.Button(description='Update',tooltip='Press to update selected legend entry with new text',layout=layout(74),disabled=False)
        # data filter settings
        self.filterLabel = widgets.Label(value='Data Filter Settings',layout=layout(150))
        self.offsetToggle = widgets.ToggleButton(value=False,description='Offset',tooltip='Apply a vertical offset to each line in dataset',layout=layout(150))
        self.offsetSize = widgets.FloatText(value=0.1e-12,description='amount:',step=.1e-12,readout_format='.1e',layout=layout(150),style={'description_width':'initial'})
        self.svgToggle = widgets.ToggleButton(value=False,description='Savitsky-Golay',layout=layout(150))
        self.svgSize = widgets.BoundedIntText(description='window:',value=3,min=3,max=101,step=2,layout=layout(150))
        self.svgOrder = widgets.BoundedIntText(description='order:',value=1,min=1,max=5,step=1,layout=layout(150))
        self.medFiltBtn = widgets.ToggleButton(description='Median',value=False,layout=layout(90),tooltip='Apply median filter to plot data')
        self.medFiltSize = widgets.BoundedIntText(description='',value=3,min=3,max=21,step=2,layout=layout(60))
        self.thresholdToggle = widgets.ToggleButton(value=False,description='Threshold',tooltip='enable to cut out values above threshold value',layout=layout(150))
        self.thresholdValue = widgets.FloatText(value=100,description='value:',layout=layout(150))
        self.averageToggle = widgets.ToggleButton(value=False,description='Average',tooltip='enable to plot average of all selected spectra, uses np.quantile to remove outliers',layout=layout(150))
        self.groupSize = widgets.BoundedIntText(description='Group:',value=3,min=3,max=20,step=1,tooltip='defines the group size used in batched averaging',layout=layout(150))
        # plotting modes settings
        self.stmlToggle = widgets.ToggleButton(value=False,description='STML Mode',tooltip='Convert bottom axis to energy\nadd top axis in wavelength\nscale intensity to current x time',layout=layout(150))
        self.normalizeTimeBtn = widgets.ToggleButton(value=True,description='Norm. Time',tooltip='Normalize intensity to current x time',layout=layout(150))
        self.normalizeCurrentBtn = widgets.ToggleButton(value=True,description='Norm. Current',tooltip='Normalize intensity to current x current',layout=layout(150))
        self.normalizeEnergyBtn = widgets.ToggleButton(value=True,description='Norm. Energy',tooltip='Normalize intensity to current x energy',layout=layout(150)) 
        self.normalizePlasmonBtn = widgets.ToggleButton(value=False,description='Norm. Plasmon',tooltip='Normalize intensity to plasmon intensity',layout=layout(150))
        self.plasmonReference = widgets.Dropdown(options=['None'],value='None',description='Plasmon:',layout=layout(150),style={'description_width':'40px'})
        # axes controls
        self.xLimitsBtn = widgets.Button(description='Update X',tooltip='Set X axis limits',layout=layout(110))
        self.xLimitsMin = widgets.FloatText(value=-1,description='Min.',layout=layout(150),style={'description_width':'40px'})
        self.xLimitsMax = widgets.FloatText(value=1,description='Max.',layout=layout(150),style={'description_width':'40px'})
        self.yLimitsBtn = widgets.Button(description='Update Y',tooltip='Set Y axis limits',layout=layout(110))
        self.yLimitsMin = widgets.FloatText(value=-1,description='Min.',layout=layout(150),style={'description_width':'40px'})
        self.yLimitsMax = widgets.FloatText(value=1,description='Max.',layout=layout(150),style={'description_width':'40px'})
        self.xLimitLock = widgets.ToggleButton(value=False,description='',icon='lock',tooltip='Lock X axis limits when loading new data',layout=layout(35))
        self.yLimitLock = widgets.ToggleButton(value=False,description='',icon='lock',tooltip='Lock X axis limits when loading new data',layout=layout(35))

        # layouts
        self.h_new_filter_layout = HBox(children=[self.newFilterText,self.addFilterBtn])
        self.v_filter_layout = VBox(children=[self.filterSelection,self.h_new_filter_layout])
        self.v_text_layout = VBox(children=[self.saveNote,self.errorText])
        self.h_process_layout = HBox(children=[self.flattenBtn,self.fixZeroBtn,self.referenceLocBtn,self.plot2DBtn])
        self.h_selection_btn_layout = HBox(children=[self.refreshBtn,self.csvBtn,self.saveBtn,self.copyBtn,self.settingsBtn])

        self.v_channel_layout = VBox(children=[self.channelXSelect,self.channelYSelect,self.saveNote])
        self.v_file_select_layout = VBox(children=[self.directorySelection,self.selectionList,self.v_filter_layout])
        
        self.v_btn_layout = VBox(children=[self.h_selection_btn_layout,self.h_process_layout,self.cmapSelection,self.markerSelection])
        self.h_user_layout = HBox(children=[self.v_channel_layout,self.v_btn_layout])

        self.v_settings_layout = widgets.Accordion(children=[
                                                    VBox(children=[
                                                        HBox(children=[
                                                            self.defaultLegendToggle,
                                                            self.customLegendToggle],
                                                            layout=layout(150)),
                                                        self.legendText,
                                                        self.legendEntry,
                                                        HBox(children=[
                                                            self.legendToggle,
                                                            self.legendUpdate],
                                                            layout=layout(150)),],
                                                        layout=layout_h(180)),
                                                    VBox(children=[
                                                        self.titleToggle,
                                                        self.nameToggle,
                                                        self.setpointToggle,
                                                        self.feedbackToggle,
                                                        self.locationToggle,
                                                        self.depthSelection,
                                                        self.dateToggle,],
                                                        layout=layout_h(180)),
                                                    VBox(children=[
                                                        self.offsetToggle,
                                                        self.offsetSize,
                                                        self.svgToggle,
                                                        self.svgSize,
                                                        self.svgOrder,
                                                        HBox(children=[
                                                            self.medFiltBtn,
                                                            self.medFiltSize],
                                                            layout=layout(150)),
                                                        self.thresholdToggle,
                                                        self.thresholdValue,
                                                        self.averageToggle,
                                                        self.groupSize],
                                                        layout=layout_h(180)),
                                                    VBox(children=[
                                                        self.stmlToggle,
                                                        self.normalizeTimeBtn,
                                                        self.normalizeCurrentBtn,
                                                        self.normalizeEnergyBtn,
                                                        self.normalizePlasmonBtn,
                                                        self.plasmonReference],
                                                        layout=layout_h(180)),
                                                    VBox(children=[
                                                        self.xLimitsMin,
                                                        self.xLimitsMax,
                                                        HBox(children=[
                                                            self.xLimitsBtn,self.xLimitLock],
                                                            layout=layout(150)),
                                                        self.yLimitsMin,
                                                        self.yLimitsMax,
                                                        HBox(children=[
                                                            self.yLimitsBtn,
                                                            self.yLimitLock],
                                                            layout=layout(150)),],
                                                        layout=layout_h(180)),
                                                    ],
                                                    layout=layout_h(220),
                                                    titles=['Legend Settings','Title Settings','Filter Settings','STML Mode','Axes Controls'])

        self.v_image_layout = VBox(children=[self.figure_display,self.h_user_layout])
        self.h_main_layout = VBox(children=[HBox(children=[widgets.Label('Session',layout=widgets.Layout(display='flex',justify_content='flex-start',width='10%')),
                                                        self.rootFolder],layout=flex_layout(99)),HBox(children=[self.v_file_select_layout,self.v_image_layout,self.v_settings_layout],layout=flex_layout(99))],layout=flex_layout(100))

        # connect widgets to functions
        self.groupSize.observe(self.update_legend_settings,names='value')
        self.averageToggle.observe(self.update_legend_settings,names='value')
        self.groupSize.observe(self.update_legend_mode,names='value')
        for child in self.v_settings_layout.children[:-1]: # exclude axes control sliders
            if type(child) == type(self.v_btn_layout):
                for ch in child.children:
                    ch.observe(self.handler_settingsChange,names='value')
            child.observe(self.handler_settingsChange,names='value')

        self.medFiltBtn.observe(self.redraw_image,names='value')
        self.medFiltSize.observe(self.redraw_image,names='value')
        
        self.defaultLegendToggle.observe(self.update_legend_mode,names='value')
        self.customLegendToggle.observe(self.update_legend_mode,names='value')
        self.legendToggle.observe(self.handler_settingsChange,names='value')
        self.legendUpdate.on_click(self.update_legend_entry)

        self.settingsBtn.observe(self.handler_settingsDisplay,names='value')
        
        self.yLimitsBtn.on_click(self.handler_update_axes_limits)
        self.xLimitsBtn.on_click(self.handler_update_axes_limits)

        self.saveBtn.on_click(self.save_figure)
        self.copyBtn.on_click(self.copy_figure)
        self.csvBtn.on_click(self.save_data)
        self.generateWaterFallBtn.on_click(self.generateWaterFall)
        self.refreshBtn.on_click(self.handler_root_folder_update)
        self.flattenBtn.observe(self.redraw_image,names='value')
        self.legendBtn.observe(self.redraw_image,names='value')
        self.referenceLocBtn.on_click(self.plotSpectrumLocations)
        self.plot2DBtn.on_click(self.plot2D)
        self.fixZeroBtn.observe(self.redraw_image,names='value')

        self.rootFolder.observe(self.handler_root_folder_update,names='value')
        self.directorySelection.observe(self.handler_folder_selection,names=['value'])
        self.selectionList.observe(self.handler_file_selection,names=['value'])
        self.filterSelection.observe(self.handler_folder_selection,names='value')
        self.addFilterBtn.on_click(self.handler_update_filters)
        self.channelXSelect.observe(self.handler_channel_selection,names='value')
        self.channelYSelect.observe(self.handler_channel_selection,names=['value'])

        self.smoothBtn.observe(self.handler_update_axes,names='value')
        self.windowParam.observe(self.handler_update_axes,names='value')
        self.orderParam.observe(self.handler_update_axes,names='value')
        self.offsetBtn.observe(self.handler_update_axes,names='value')
        self.offset_value.observe(self.handler_update_axes,names='value')
        self.cmapSelection.observe(self.handler_update_axes,names='value')

        self.stmlToggle.observe(self.handler_settingsChange,names='value')
        self.plasmonReference.observe(self.handler_update_plasmonic_reference,names='value')


        with self.figure_display:
            self.figure_display.clear_output(wait=True)
            plt.show(self.figure)

        self.display()
        #with self.wfFigure_display:
            #plt.show(self.wfFigure)
        self.find_directories(self.active_dir)
        #self.update_directories()
    # show browser
    def display(self):
        #display.clear_output(wait=True)
        display.display(self.h_main_layout)
    
    def find_directories(self,_path):
        directories = []
        for _directory in os.listdir(_path):
            if _directory[-4:] in ['.dat','.sxm']: continue
            elif os.path.isdir(_path / _directory):
                if 'browser_outputs' in _directory or 'ipynb' in _directory or 'spmpy' in _directory or '__pycache__' in _directory or 'raw_stml_data' in _directory: continue
                directories.append(_path / _directory)
                self.find_directories(_path / _directory)
        else:
            pass
        self.directories.extend(directories)
        return directories
    def update_directories(self):
        display_directories = ['\\'.join(str(directory).split('\\')[-1:]) for directory in self.directories]
        display_directories[0] = 'session folder'
        self.directorySelection.options = display_directories
    # output functions
    def save_figure(self,a):
        self.saveBtn.icon = 'hourglass-start'
        if os.path.exists(self.active_dir / 'browser_outputs'):
            pass
        else:
            os.mkdir(self.active_dir / 'browser_outputs')
        fname = 'browser_outputs/' + str(self.directorySelection.value).split("\\")[-1] + f'_{self.spec[0].name.split(".")[0]}_{self.channelYSelect.value[0]}'
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
        self.saveBtn.icon = 'hourglass-start'
        if os.path.exists(f'{self.active_dir}/browser_outputs'):
            pass
        else:
            os.mkdir(self.active_dir / 'browser_outputs')
        fname = 'browser_outputs/' + str(self.directorySelection.value).split("\\")[-1] + f'_{self.spec[0].name.split(".")[0]}_{self.channelYSelect.value[0]}'
        if self.saveNote.value != '':
            fname += f'_{self.saveNote.value}'
        fname = self.active_dir / fname
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
        if self.loaded_experiments != None:
            self.update_image_data()
            self.updateDisplayImage()
    def load_new_image(self,filename=None):
        #self.updateErrorText('load new image')
        directory = self.directories[self.directorySelection.index]
        if directory != self.active_dir:
            directory = os.path.join(self.active_dir,directory)
        if filename == None:
            files = [os.path.join(directory,self.dat_files[index]) for index in self.spec_index]
            self.spec = [Spm(f) for f in files]
            self.filenameText.value = ''.join([f'{self.all_files[self.spec_index[i]]},' for i in range(len(self.spec_index))])
            self.loaded_experiments = [spec.header['Experiment'] for spec in self.spec]
            self.updateChannelSelection()
            self.update_image_data()
        else:
            return Spm(os.path.join(directory,filename))

        #self.updateErrorText('finish load new image')
    def smooth_data(self,data):
        window = self.svgSize.value
        order = self.svgOrder.value
        return savgol_filter(data,window,order)
    def rebin_intensity_nm_to_ev(self,wavelengths,intensities):
        center_energies = 1240 / wavelengths
        delta_wavelengths = np.abs(np.diff(wavelengths))
        delta_wavelengths = np.insert(delta_wavelengths,0,delta_wavelengths[0])
        delta_energies = [1240/(w-dw)-1240/(w+dw) for w,dw in zip(wavelengths,delta_wavelengths)]
        return center_energies,np.array([intensity/de for intensity,de in zip(intensities,delta_energies)]) # intensity / dE --> Counts / eV

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
                    if channelX == 'Index':
                        spec_x = np.arange(len(spec_data))
                        xunit = 'N'
                    else:
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

        ## adjusting data for STML plotting mode
        if self.stmlToggle.value and 'stml' in self.loaded_experiments[0].lower():
            data = []
            xx = []
            for i,spec in enumerate(self.spec):
                time = float(spec.header['Exposure Time [ms]'])/1000 # convert to seconds
                normfactor = 1
                if self.normalizeTimeBtn.value:
                    normfactor *= time
                if self.normalizeCurrentBtn.value:
                    current = abs(np.average(spec.get_channel('I')[0])) # in pA
                    normfactor *= current
                if self.normalizeEnergyBtn.value:
                    energies,intensities = self.rebin_intensity_nm_to_ev(self.spec_x[i],self.spec_data[i])
                else:
                    energies = 1240/self.spec_x[i]
                    intensities = self.spec_data[i]
                if self.plasmonReference.value != 'None' and self.plasmonInfo['file'] != None and self.normalizePlasmonBtn.value:
                    plasmon = abs(self.plasmonInfo['interp'](energies))+.1

                    data.append(intensities/normfactor/plasmon*(plasmon>0)*(self.spec_data[i]>5)) # remove counts below 35 to avoid noise amplification
                else:
                    data.append(intensities/normfactor)
                xx.append(energies) # convert from nm to eV
            self.spec_data = data
            self.spec_x = xx
            self.spec_info[0]['x_unit'] = 'eV'
            factor_list = ['cts']
            if self.normalizeCurrentBtn.value and self.normalizeTimeBtn.value:
                factor_list += ['pC']
            if self.normalizeCurrentBtn.value and not self.normalizeTimeBtn.value:
                factor_list += ['pA']
            if self.normalizeTimeBtn.value and not self.normalizeCurrentBtn.value:
                factor_list += ['s']
            if self.normalizeEnergyBtn.value:
                factor_list += ['eV']
            self.spec_info[0]['y_unit'] = '/'.join(factor_list) #cts/pC/eV'
            self.spec_info[0]['x_label'] = 'Energy'

            return spec, spec_data, spec_x
        #self.updateErrorText('finish update image data')
    def update_scan_info(self):
        #self.updateErrorText('update scan info')
        experiments = self.loaded_experiments#[spec.header['Experiment'] for spec in self.spec]
        assert experiments.count(experiments[0]) == len(experiments), 'Please ensure all selections are the same measurement type'
        spec = self.spec[0]
        label = []
        experiment = experiments[0]
        experiment_str = f'Experiment: {experiment} $\\rightarrow$ filename: {spec.name}'
        feedback_str = ''
        if self.nameToggle.value:
            label.append(experiment_str)
        if 'STML' in experiment:
            fb_enable = spec.header['Z-Controller>Controller status']
            set_point = spec.get_param('setpoint_spec')
            bias = spec.get_param('V_spec')
            if np.abs(bias[0])<0.1:
                bias = list(bias)
                bias[0] = bias[0]*1000
                bias[1] = 'mV'
                bias = tuple(bias)
            if fb_enable == 'ON':
                feedback_str = 'feedback on'
            elif fb_enable == 'OFF':
                feedback_str = 'feedback off'
            label.append(fr'Exposure Time (s): {float(spec.header["Exposure Time [ms]"])/1000:.0f}, $\lambda_c$: {spec.header["Center Wavelength [nm]"]}, grating: {spec.header["Selected Grating"]}')
            setpoint_str = 'setpoint: I = %.0f%s, V = %.1f%s' % (set_point+bias)  
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
                feedback_str = 'feedback on'
            elif fb_enable == 'TRUE':
                feedback_str = 'feedback off'
            setpoint_str = 'setpoint: I = %.0f%s, V = %.1f%s' % (set_point+bias)  
        if 'THz amplitude sweep' in experiment:
            label.append(f'Laser Rep. Rate: {spec.header["Ext. VI 1>Laser>PP Frequency (MHz)"]}')
            label.append(f'Pulse Polarity: THz1;{spec.header["Ext. VI 1>THzPolarity>THz1"]}, THz2;{spec.header["Ext. VI 1>THzPolarity>THz2"]}')
            label.append(f'Delay Positions: THz1;{spec.header["Ext. VI 1>Position>PP1 (m)"]}, THz2;{spec.header["Ext. VI 1>Position>PP2(m)"]}')
        if 'Z spectroscopy' in experiment:
            set_point = spec.get_param('setpoint_spec')
            bias = spec.get_param('V_spec')
            setpoint_str = 'setpoint: I = %.0f%s, V = %.1f%s' % (set_point+bias)  
            label.append(f'Spec Points: {len(self.spec_data)}')
            label.append(f'Integration time (s): {spec.header["Integration time (s)"]}')
            label.append(f'z-sweep (m): {spec.header["Z sweep distance (m)"]}')
        if 'History Data' in experiment:
            set_point = spec.get_param('setpoint_spec')
            bias = spec.get_param('V_spec')
            setpoint_str = 'setpoint: I = %.0f%s, V = %.1f%s' % (set_point+bias)  
            label.append(f'Bias (V): {spec.header["Bias>Bias (V)"]}')
            label.append(f'Feedback: {spec.header["Z-Controller>Controller status"]}')
            label.append(f'Sample Period (ms): {spec.header["Sample Period (ms)"]}')
        else:
            pass

        if len(self.spec) > 1:
            d1 = self.spec[0].header["Saved Date"]
            d2 = self.spec[-1].header["Saved Date"]
            date_str = f'Date: {d1} $\\rightarrow$ {d2}'
        else:
            date_str = f'Date: {spec.header["Saved Date"]}'

        # construct label
        if self.setpointToggle.value:
            if self.feedbackToggle.value:
                setpoint_str += " $\\rightarrow$ "
                setpoint_str += feedback_str
            label.append(setpoint_str)
        if self.locationToggle.value:
            location = self.directories[self.directorySelection.index]
            if self.depthSelection.value == 'full':
                label.append(f'location: {location}')
            else:
                location = '\\'.join(str(location).split('\\')[-int(self.depthSelection.value):])
                label.append(f'location: {location}')
        if self.dateToggle.value:
            label.append(date_str)
        if self.svgToggle.value:
            label.append(f'Savitzky-Golay Filter $\\rightarrow$ Window: {self.svgSize.value}, Order: {self.svgOrder.value}')
        #label.append('comment: %s' % comment)
        if self.titleToggle.value:
            self.spec_label = '\n'.join(label)
        else:
            self.spec_label = ''
        #self.updateErrorText('finish update scan info')
    #########################
    ### plotting function ###
    #########################
    def update_axes(self):
        print('update axes called')
        #self.updateErrorText('update axes')
        if not self.figure:
            self.updateErrorText('making new figure/axes')
            self.figure,self.axes = plt.subplots(ncols=1,figsize=(8,8))
            #self.figure.tight_layout(pad=2)
        ax = self.axes
        if self.cb:
            self.cb.remove()
            self.cb = None
        ax.clear()

        if self.axes2 != None:
            self.axes2.remove()
            self.axes2 = None

        y_values = self.spec_data
        x_values = self.spec_x
        if self.averageToggle.value and len(self.spec_data) % self.groupSize.value == 0:
            x_values,y_values = self.group_average()

        for i in range(len(y_values)):
            y = y_values[i]
            offset = 0
            if self.fixZeroBtn.value:
                offset = np.mean(self.spec_data[i][np.where(abs(x_values[i])<0.1)[0]])
            if self.thresholdToggle.value:
                y *= (y < self.thresholdValue.value)
            if self.svgToggle.value:
                y = self.smooth_data(y)
            if self.medFiltBtn.value:
                y = medfilt(y,self.medFiltSize.value)
            if self.flattenBtn.value:
                y = y / np.max(y)
            if self.offsetToggle.value:
                y = y + i*self.offsetSize.value
            y_values[i] = y

        if self.defaultLegendToggle.value:
            if self.averageToggle.value and len(self.spec_data) % self.groupSize.value == 0:
                labels = [self.labels[i] for i in range(0, len(self.labels), self.groupSize.value)]
            else:
                labels = self.labels
        elif self.customLegendToggle.value:
            labels = self.legendText.options

        colors = plt.cm.get_cmap(str(self.cmapSelection.value))(np.linspace(0,1,len(y_values)))
        for i in range(len(y_values)):
                ax.plot(x_values[i],y_values[i],color=colors[i],label=labels[i])

        ax.set_title(self.spec_label,fontsize=self.titlesize,loc='left')
        ax.set_xlabel(f'{self.spec_info[0]["x_label"]} ({self.spec_info[0]["x_unit"]})',fontsize=self.fontsize)
        ax.set_ylabel(f'{self.channelYSelect.value[0]} ({self.spec_info[0]["y_unit"]})',fontsize=self.fontsize)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="y",which='both',direction="in",)
        ax.tick_params(axis="x",which='both',direction="in",)

        if self.legendToggle.value:
            if len(self.axes.lines) < 4:
                fontsize = self.legendFontsize[0]
            elif len(self.axes.lines) < 16:
                fontsize = self.legendFontsize[1]
            elif len(self.axes.lines) >= 16:
                fontsize = self.legendFontsize[2]
            ax.legend(draggable=True,fontsize=fontsize,frameon=False)
        else:
            pass

        # add secondary axis for STML mode
        if self.stmlToggle.value and 'stml' in self.loaded_experiments[0].lower():
            xmin, xmax, ymin, ymax = self.axes.axis()
            if self.axes2 != None:
                self.axes2.remove()
                self.axes2 = None
            self.axes2 = ax.twiny()
            self.axes2.set_xlabel('Wavelength (nm)',fontsize=self.fontsize)
            self.axes2.set_xscale('function',functions=(lambda en: 1240/(en+1e-9),lambda lam: 1240/(lam+1e-9)))
            for i in range(len(self.spec_data)):
                energy = self.spec_x[i][np.where((self.spec_x[i]>=xmin) & (self.spec_x[i]<=xmax))]
                self.axes2.plot(1240/energy,energy,alpha=0)
            self.axes2.xaxis.set_minor_locator(AutoMinorLocator(2))
            self.axes2.tick_params(axis="x",which='both',direction="in",)
            #self.axes2.tick_params(axis="y",direction="in",)
            self.axes2.set_zorder(ax.get_zorder()-1)
            self.axes2.xaxis.set_zorder(ax.xaxis.get_zorder()+1)
            self.axes.set_ylim(ymin,ymax)

        self.updateAxesLimitSliders()

        if self.xLimitLock.value:
            self.axes.set_xlim(self.xLimitsMin.value,self.xLimitsMax.value)
            self.axes2.set_xlim(1240/self.xLimitsMax.value,1240/self.xLimitsMin.value)
        if self.yLimitLock.value:
            self.axes.set_ylim(self.yLimitsMin.value,self.yLimitsMax.value)

        self.figure.tight_layout(pad=1)
        ## secondardy function calls
        
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
            colors = plt.cm.get_cmap(self.cmapSelection.value)(np.linspace(0,1,len(self.axes.lines)))
            k = 0
            for i,spec in enumerate(self.spec):
                # skip spectra for averaging display
                if self.averageToggle.value and len(self.spec_data) % self.groupSize.value == 0:
                    indices = list(range(0, len(self.spec), self.groupSize.value))
                    if i not in indices:
                        continue
                
                rx,ry = self.relative_position(self.sxmBrowser.img,spec)
                rel_positions.append([rx,ry])
                if self.markerSelection.value != 'N':
                    ax.plot(rx,ry,marker=self.markerSelection.value,markersize=10,color=colors[k],label=self.labels[k].split('.')[0][-3:])
                elif self.markerSelection.value == 'N':
                    start = np.max(self.spec_index)-np.min(self.spec_index)+1
                    ax.text(rx,ry,f'{start-i}',color='r',fontsize=10,ha='center',va='center')
                k += 1

    def plot2D(self,a):
        if len(self.spec) != 0:
            dataPoints = max([len(spec_data) for spec_data in self.spec_data])
            xData = self.spec_x[[len(spec_data) for spec_data in self.spec_data].index(dataPoints)]
            yLabels = [int(''.join([s for s in spec.name if s.isdigit()])) for spec in self.spec] #[int(name) for name in spec.name.split() if name.isdigit()]
            ymin = 0
            ymax = len(yLabels)
            ylabel = 'Spec Number'
            dataArray = np.zeros((len(self.spec),dataPoints))
            directory = self.directories[self.directorySelection.index]
            if directory != self.active_dir:
                directory = os.path.join(self.active_dir,directory)
            files = [(self.spec_data[i], os.path.getmtime(os.path.join(directory, f))) for i,f in enumerate(self.selectionList.value)]
            data = sorted(files, key=lambda x: x[1], reverse=False)
            direction = np.sign(xData[0]-xData[-1])
            for i,spec_data in enumerate([d[0] for d in data]):
                if direction == 1:
                    dataArray[i,:len(spec_data)] = np.flip(spec_data)
                else:
                    dataArray[i,:len(spec_data)] = spec_data
            vmin = np.min(dataArray)
            vmax = np.max(dataArray)
            #print(yLabels)
            self.axes.clear()
            axesImage = self.axes.imshow(dataArray,aspect='auto',origin='lower',extent=[np.min(xData),np.max(xData),ymin,ymax],cmap=self.cmapSelection.value,interpolation='none')
            self.axes.set_ylabel(ylabel)
            self.axes.set_xlabel(f'{self.spec_info[0]["x_label"]} ({self.spec_info[0]["x_unit"]})')
            if not self.cb:
                divider = make_axes_locatable(self.axes)
                cax = divider.append_axes("right", size="5%",pad=0.05)
                #self.cb = self.figure.colorbar(axesImage,ax=ax,shrink=0.75,pad=.05)
                self.cb = self.figure.colorbar(axesImage,cax=cax)
            else:
                #self.cb.remove()
                self.cb.update_normal(axesImage)# = self.figure.colorbar(axesImage,ax=ax,shrink=0.75,pad=.01)
            self.cb.set_label(f'{self.channelYSelect.value[0]} ({self.spec_info[0]["y_unit"]})',fontsize=self.fontsize)
            
        else:
            pass
    def group_average(self):
        group_size = self.groupSize.value
        data = self.spec_data # list of arrays
        xx = self.spec_x
        grouped_data = [data[i:i + group_size] for i in range(0, len(data), group_size)]
        grouped_x = [xx[i] for i in range(0, len(xx), group_size)]
        group_averaged = []
        for group in grouped_data:
            median_average = []
            for element_group in zip(*group):
                medians = np.sort(medfilt(element_group,3))
                medians = medians[0:-1] # remove max
                median_average.append(np.average(medians))
            group_averaged.append(np.array(median_average))
        return grouped_x,group_averaged

### display configuration
    def updateDisplayImage(self,*params):
        #self.updateErrorText('update display image')
        self.update_axes()
        #self.figure.canvas.set_window_title(self.img.name.split('.')[0])
        self.figure_display.clear_output(wait=True)
        self.figure.canvas.draw()
        #self.updateErrorText('finish update display image')
    def updateInfoText(self):
        self.filenameText.value = self.dat_files[self.spec_index]
        self.selectionList.value = self.all_files[self.spec_index]
    def updateChannelSelection(self):
        if len(self.spec) > 0:
            default_channels = [None,None]
            if self.loaded_experiments[0] in self.default_channel.keys():
                default_channels = self.default_channel[self.loaded_experiments[0]]
            #print(default_channels)
            current_value_X = self.channelXSelect.value
            current_value_Y = self.channelYSelect.value[0]
            self.channelXSelect.options = ['Index'] + self.spec[0].channels
            self.channelYSelect.options = self.spec[0].channels
            if self.current_experiment == self.loaded_experiments[0] and current_value_X in self.spec[0].channels:
                self.channelXSelect.value = current_value_X
            elif default_channels[0] != None:
                self.channelXSelect.value = default_channels[0]
            else:
                self.channelXSelect.value = 'Index'
            if self.current_experiment == self.loaded_experiments[0] and current_value_Y in self.spec[0].channels:
                self.channelYSelect.value = [current_value_Y]
            elif default_channels[1] != None:
                self.channelYSelect.value = [default_channels[1]]
            else:
                self.channelYSelect.value = [self.spec[0].channels[0]]
        self.current_experiment = self.loaded_experiments[0]
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
    def updateAxesLimitSliders(self):
        xmin,xmax,ymin,ymax = self.axes.axis()
        if not self.xLimitLock.value:
            self.xLimitsMin.value = xmin
            self.xLimitsMax.value = xmax
        if not self.yLimitLock.value:   
            self.yLimitsMin.value = ymin
            self.yLimitsMax.value = ymax
### settings handler
    def handler_settingsChange(self,a):
        if a['owner'] != self.selectionList:
            self.redraw_image(a)

    def handler_settingsDisplay(self,a):
        if self.settingsBtn.value:
            self.v_settings_layout.layout.visibility = 'visible'
            for child in self.v_settings_layout.children:
                if type(child) == type(self.v_settings_layout):
                    for ch in child.children:
                        ch.layout.visibility = 'visible'
                child.layout.visibility = 'visible'
        else:
            self.v_settings_layout.layout.visibility = 'hidden'
            for child in self.v_settings_layout.children:
                if type(child) == type(self.v_settings_layout):
                    for ch in child.children:
                        ch.layout.visibility = 'hidden'
                child.layout.visibility = 'hidden'

    def update_legend_mode(self,a):
        print(a)
        if a['owner'] == self.defaultLegendToggle and a['new'] == True:
            self.customLegendToggle.value = False
        elif a['owner'] == self.customLegendToggle and a['new'] == True:
            self.defaultLegendToggle.value = False
        elif a['owner'] == self.groupSize:
            # set legend mode to default if group averaging is not configured properly
            if self.averageToggle.value and len(self.spec_data) % self.groupSize.value != 0:
                self.defaultLegendToggle.value = True
                self.customLegendToggle.value = False
        self.updateDisplayImage(a)
    def update_legend_entry(self,a):
        entry = self.legendText.value
        options = list(self.legendText.options)
        index = options.index(entry)
        new_text = self.legendEntry.value
        options[index] = new_text
        self.legendText.options = options
        self.updateDisplayImage(a)
    def update_legend_settings(self,a):
        if a['owner'] == self.selectionList or a['owner'] == self.averageToggle or a['owner'] == self.groupSize:
            new_selection = self.selectionList.value
            if self.averageToggle.value and len(self.spec_data) % self.groupSize.value == 0:
                new_selection = self.selectionList.value
                grouped_selection = [new_selection[i] for i in range(0, len(new_selection), self.groupSize.value)]
                new_selection = grouped_selection
            self.legendText.options = new_selection
            self.legendText.value = new_selection[0]
#        if a['owner'] == self.legendText:
#            self.legendEntry.value = a['new']
### Selection update
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
            self.update_directories()
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
                if 'all' in self.filterSelection.value or any(filt in file for filt in self.filterSelection.value):
                    self.dat_files.append(file)
        files = [(f, os.path.getmtime(os.path.join(directory, f))) for f in self.dat_files]
        self.dat_files = [f[0] for f in sorted(files, key=lambda x: x[1], reverse=True)]
        self.all_files = self.sxm_files + self.dat_files
        self.selectionList.options = self.dat_files
        if len(self.dat_files) != 0:
            if type(index) == int:
                self.filenameText.value = self.dat_files[index]
            else:
                self.filenameText.value = self.dat_files[index[0]]
            if self.filenameText.value in self.selectionList.options:
                self.selectionList.value = [self.filenameText.value]
            self.plasmonReference.options = ['None'] + [f for f in self.dat_files if 'stml' in f.lower()]
            self.plasmonReference.value = 'None'

        #print(self.dat_files)
    def handler_file_selection(self,update:object):
        #self.updateErrorText(str(update))
        self.spec_index = [self.dat_files.index(value) for value in self.selectionList.value]#self.all_files.index(self.selectionList.value)
        try:
            if len(self.selectionList.value) > 0:
                self.load_new_image()
                self.update_legend_settings(update)
                self.updateDisplayImage()
                #
        except Exception as err:
            self.updateErrorText('file selection error:' + str(err))
            print(traceback.format_exc())

        #self.update_legend_settings(update)

    def handler_channel_selection(self,update):
        try:
            #channel_number = self.spec[0].channels.index(self.channelSelect.value[0])
            if len(self.spec) > 0:
                self.update_image_data()
                self.updateDisplayImage()
        except Exception as err:
            self.updateErrorText('channel selection error:' + str(err))
    def handler_update_filters(self,update):
        if self.newFilterText.value != '' and self.newFilterText.value not in self.filterSelection.options:
            options = list(self.filterSelection.options)
            options.append(self.newFilterText.value)
            self.filterSelection.options = options
            self.newFilterText.value = ''
    def handler_update_plasmonic_reference(self,update):
        if self.plasmonReference.value != 'None':
            directory = self.directories[self.directorySelection.index]
            if directory != self.active_dir:
                directory = os.path.join(self.active_dir,directory)
            if self.plasmonReference.value in self.dat_files:
                reference_spec = Spm(os.path.join(directory,self.plasmonReference.value))
                reference_y, yunit = reference_spec.get_channel('Intensity')
                reference_x, xunit = reference_spec.get_channel('Wavelength')
            
                reference_y = abs(savgol_filter(reference_y,21,1))
                reference_y *= (reference_y > 35) # 2x noise floor + a little extra
                #reference_y += 1e-2*np.max(reference_y)
                reference_y /= np.max(reference_y)
                ref_data_interp = interp1d(1240/reference_x,reference_y,bounds_error=False,fill_value=0.0) # interpolated on energy axis
                self.plasmonInfo['spm'] = reference_spec
                self.plasmonInfo['file'] = self.plasmonReference.value
                self.plasmonInfo['interp'] = ref_data_interp
                
### data presentation update
    def handler_update_axes(self,a):
        self.update_scan_info()
        self.update_axes()
        #self.updateDisplayImage(a)
    def handler_update_axes_limits(self,a):
        if a == self.xLimitsBtn:
            self.axes.set_xlim(self.xLimitsMin.value,self.xLimitsMax.value)
            self.axes2.set_xlim(1240/self.xLimitsMax.value,1240/self.xLimitsMin.value)
        elif a == self.yLimitsBtn:
            self.axes.set_ylim(self.yLimitsMin.value,self.yLimitsMax.value)
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

# execution pattern for certain features
# 