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
class _fileBrowser():
    '''
    Info:
        figure = _Browser.figure
        axes = _Browser.axes

        - _Browser.update_axes() can be used to refresh the browser plot
        - axes can be accessed for futher plot modification (axis limits, labels, etc)
    '''
    def __init__(self,selection_type='sxm',root_directory='./'):
        # selection_type == 'all' or 'sxm' or 'dat'
        self.selection_type = selection_type
        self.root = Path(root_directory)
        self.directories = [self.root]
        self.spm_files = []
        self.files = []
        self.selected_files = []

        self.create_gui_objects()

        self.find_directories(self.root)
        self.update_directories()

##### Create GUI elements
    def create_gui_objects(self):
        # widget layouts
        layout = lambda x: widgets.Layout(visibility='visible',width=f'{x}px')
        self.rootTextEntry = widgets.Text(description='FilePath:',tooltip='Enter the root file directory here',layout=layout(500))
        self.rootBtn = widgets.Button(description='Update',tooltip='click to set new root directory',layout=layout(60))
        self.refreshBtn = Btn_Widget('',icon='refresh',tooltip='Reload file list',layout=layout(30))

        self.directorySelection = Selection_Widget(self.directories,'Folders:',rows=5)
        if self.selection_type == 'sxm':
            self.selectionList = widgets.Select(options=self.files,description='Files:',rows=30)
        if self.selection_type == 'dat':
            self.selectionList = widgets.SelectMultiple(options=self.files,value=[],description='Files:',rows=30)
        

        self.nextBtn = Btn_Widget('',layout=layout(30),icon='arrow-circle-down',tooltip='Load next file in list')
        self.previousBtn = Btn_Widget('',layout=layout(30),icon='arrow-circle-up',tooltip='Load previous file in list')

        # layouts
        self.h_root_layout = HBox(children=[self.rootTextEntry,self.rootBtn])
        self.h_btn_layout = HBox(children=[self.refreshBtn,self.previousBtn,self.nextBtn])
        self.main_layout = VBox(children=[self.h_root_layout,self.directorySelection,self.selectionList,self.h_btn_layout])
        # connect widgets to functions
        self.nextBtn.on_click(self.handler_next_file)
        self.previousBtn.on_click(self.handler_previous_file)
        self.refreshBtn.on_click(self.handler_directory_changed)
        self.rootBtn.on_click(self.handler_update_root_directory)
        self.directorySelection.observe(self.handler_directory_changed,names=['value'])
        self.selectionList.observe(self.handler_file_changed,names=['value'])

##### Handler Methods (GUI callbacks)
    def handler_update_root_directory(self,a):
        if os.path.isdir(self.rootTextEntry.value):
            self.root = Path(self.rootTextEntry.value)
            self.directories = [self.root]
            self.find_directories(self.root)
            self.update_directories()

    def handler_directory_changed(self,a):
        index=0
        if type(a) == type(self.refreshBtn): 
            index = self.selectionList.index
        directory = self.directories[self.directorySelection.index]
        print(directory)
        self.spm_files = [file for file in os.listdir(directory) if self.is_spm(file)]
        self.sxm_files = [file for file in self.spm_files if '.sxm' in file]
        self.dat_files = [file for file in self.spm_files if '.dat' in file]
        self.files = {'sxm':self.sxm_files,'dat':self.dat_files,'browser':self.spm_files}[self.selection_type]
        print(self.files)
        self.update_selection_list(index=index,updateOptions=True)

    def update_selection_list(self,index=0,updateOptions=False):
        if updateOptions:
            self.selectionList.options = self.files
        if len(self.selectionList.options) > 0:
            if self.selection_type in ['sxm','browser']:
                self.selectionList.value = self.selectionList.options[index]
            if self.selection_type == 'dat':
                pass
                self.selectionList.value = [self.selectionList.options[index]]

    def handler_file_changed(self,update:object):
        if len(self.spm_files) == 0: 
            return
        if self.selection_type == 'dat':
            #self.selectionList.index = [self.selectionList.index[0]]
            self.selected_files = [file for file in self.selectionList.value]
        if self.selection_type == 'sxm':
            self.selected_files = [self.selectionList.value]
        if len(self.selected_files) > 0:
            self.files = []
            [self.load_new_file(file) for file in self.selected_files]
            print(self.files)
    def handler_next_file(self,a):
        if self.selection_type in ['sxm','browser']:
            file_index = self.selectionList.index + 1
            if file_index == len(self.selectionList.options):
                file_index = 0
            self.selectionList.index = file_index
        if self.selection_type in ['dat']:
            file_index = [self.selectionList.index[0] + 1]
            if file_index[0] == len(self.selectionList.options):
                file_index = [0]
            self.selectionList.index = file_index

    def handler_previous_file(self,a):
        if self.selection_type in ['sxm','browser']:
            file_index = self.selectionList.index - 1
            if file_index == len(self.selectionList.options):
                file_index = len(self.selectionList.options)-1
            self.selectionList.index = file_index
        if self.selection_type == 'dat':
            file_index = [self.selectionList.index[0] - 1]
            if file_index[0] == 0:
                file_index = [len(self.selectionList.options)-1]
            self.selectionList.index = file_index

##### Directory and File selections
    def find_directories(self,_path):
        directories = []
        for _directory in os.listdir(_path):
            if os.path.isdir(_path / _directory):
                if 'browser_outputs' in _directory or 'ipynb' in _directory or '__pycache__' in _directory: continue
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

##### image generation
    def load_new_file(self,filename):
        directory = self.directories[self.directorySelection.index]
        if directory != self.root:
            directory = self.root / directory
            #directory = os.path.join(self.root,directory)
        filepath = directory / filename
        #filepath = os.path.join(directory,filename)
        print(filepath)
        if Path.is_file(filepath):
            self.files.append(Spm(filepath))

##### misc
    def display(self):
        display.display(self.main_layout)

    def is_spm(self,filename):
        spm_file = False
        if '.sxm' in filename or '.dat' in filename:
            spm_file = True
        for extension in ['png','jpeg','jpg','svg']:
            if extension in filename:
                spm_file = False
        return spm_file
##### output functions