import os
import sys
import spe_loader
from IPython import display
from IPython.display import HTML
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt,savgol_filter
import importlib
import subprocess
from pathlib import Path
importlib.reload(spe_loader)

class Spe_Browser():
    def __init__(self,standalone=True):
        self.figure, self.ax = plt.subplots(figsize=(7,6),num='spe')
        self.ax2 = self.ax.twiny()
        self.ax2.set_navigate(False)
        self.ax2.set_xlabel('Photon Energy (eV)')

        # set default values
        self.directories = ['./']
        self.spe_filenames = []
        self.spe_files = {}
        # plot variables
        self.headers = []
        self.markers = []
        self.marker = None

        # create widgets
        self.create_widgets()
        # display widgets
        self.handler_update_files('a')            
  
        if standalone:
            display.display(self.h_layout_main)
        with self.figure_display:
            plt.show(self.figure)

        self.find_directories(Path('./'))
        self.handler_update_directory(self.directories)
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
    def handler_update_directory(self,directory_list):
        self.directorySelection.options = self.directories
        self.handler_update_files('a')

    def handler_update_files(self,a):
        self.spe_filenames = []
        for f in os.listdir(self.directorySelection.value):
            if '.spe' in f:
                if 'raw' not in f:
                    self.spe_filenames.append(f)
                    self.spe_files[f] = spe_loader.SpeFile(str(self.directorySelection.value.absolute()),f)
        self.selectionList.options = self.spe_filenames
        self.selectionList.value = []

    def handler_update_filter(self,a):
        if self.filterBtn.value:
            self.handler_update_data(a)

    def cosmic_filter(self,y):
        width = int(self.filterWidth.value)
        threshold = int(self.filterThreshold.value)
        bad_pixels = y > threshold
        inds = [i for i in range(len(bad_pixels)) if bad_pixels[i]]
        for ind in inds:
            y[ind-width//2:ind+width//2] = np.average([y[ind-width//2],y[ind+width//2]])
        return y

    def update_ax2_ticks(self):
        self.figure.canvas.draw()
        # match the ax2 ticks with the ax ticks
        self.ax2.set_xticks(self.ax.get_xticks())
        self.ax2.set_xlim(self.ax.get_xlim())
        # grab ticklabels and convert to photon energy
        ax_ticklabels = self.ax.get_xticklabels()
        ax2_ticklabels = [round(1240/float(ticklabel.get_text()),3) for ticklabel in ax_ticklabels]
        self.ax2.set_xticklabels([str(ticklabel) for ticklabel in ax2_ticklabels])

    def update_plot_bounds(self,xmin,xmax,ymin,ymax): 
        self.xmin.value = xmin
        self.xmax.value = xmax
        self.ymin.value = ymin
        self.ymax.value = ymax
        self.handler_update_axes('a')

    def handler_update_ax_title(self,a):
        detail_labels = list(self.headers[0].keys())
        detail_labels.remove('frames')
        detail_labels.remove('date')
        detail_labels.remove('time')
        details = {}
        for key in detail_labels:
            details[key] = []
        for head in self.headers:
            for key in detail_labels:
                details[key].append(head[key])
        detail_string = [f'{key}: {item}' for key,item in details.items()]
        detail_string.append(f'date: {self.headers[0]["date"]}')
        title = '\n'.join(detail_string)
        title += f'\nBias (V): {self.biasNote.value}'
        title += f'\nCurrent (pA): {self.currentNote.value}'
        self.ax.set_title(title,loc='left',fontsize=10)
        #self.update_ax2_ticks()

    def handler_update_data(self,a):
        if self.selectionList.value == []:
            pass
        else:
            self.headers = []
            if len(self.selectionList.value) >= 1:
                self.ax.clear()
                ymin = 0
                ymax = 0
                for value in self.selectionList.value:
                    spe_file = self.spe_files[value]
                    x = spe_file.wavelengths
                    y = spe_file.intensity_avg
                    if max(y) > ymax:
                        ymax = max(y)
                    self.headers.append(spe_file.info)
                    #if self.filterBtn.value:
                    #    y = self.cosmic_filter(y)
                    self.ax.plot(x,y,'grey',alpha=0.5)
                    self.ax.plot(x,savgol_filter(y,21,2),label=f'{spe_file.info["date"]} [{spe_file.info["spec_number"]}]')
                self.ax.legend()
                self.ax.set_xlabel('wavelength (nm)')
                self.ax.set_ylabel('Intensity (counts)')
                self.figure.canvas.draw()
                self.update_plot_bounds(min(x),max(x),ymin,ymax)
                self.handler_update_ax_title(a)
                self.figure.tight_layout(pad=2)
                #headerNote.value = ''

    def handler_update_data_csv(self,a):
        self.headers = []
        if len(self.selectionList.value) >= 1:
            header_rows = 3
            self.ax.clear()
            for value in self.selectionList.value:
                x,y = np.loadtxt(f'{self.directorySelection.value}/{value}',delimiter=',',skiprows=4,unpack=True)
                with open(f'{self.directorySelection.value}/{value}') as f:
                    self.headers.append([next(f).strip()[2:] for i in range(header_rows)])
                if self.filterBtn.value:
                    y = self.cosmic_filter(y)
                self.ax.plot(x,y,label=value)
            for header in self.headers:
                header[2] = f'acquition time (s): {int(header[2].split(" ")[-1])//1000}'
            self.ax.legend()
            self.ax.set_xlabel('wavelength (nm)')
            self.ax.set_ylabel('Intensity (counts)')
            self.handler_update_axes(a)
            self.handler_update_ax_title(a)
            self.figure.tight_layout(pad=2)
            #headerNote.value = ''
    def handler_update_vline(self,a):
        if self.marker:
            self.marker.remove()
        self.marker = self.ax.axvline(self.vlinePosition.value,linestyle=':',color='k')
    def handler_add_vline(self,a):
        self.markers.append(self.marker)
        self.marker = None
    def handler_remove_vlines(self,a):
        for i in range(len(self.markers)):
            self.ax.lines.pop(-1)
        self.markers = []
        if self.marker:
            self.marker.remove()
            self.marker = None
        self.figure.canvas.draw()
    def handler_update_axes(self,a):
        self.ax.set_xlim(self.xmin.value,self.xmax.value)
        self.ax.set_ylim(self.ymin.value,self.ymax.value)
        self.update_ax2_ticks()

    def copy_figure(self,a):
        self.save_figure(a)
        # Make powershell command
        powershell_command = r'$imageFilePaths = @("'
        for image_path in [self.last_save_fname]:
            powershell_command += image_path + '","'
        powershell_command = powershell_command[:-2] + '); '
        powershell_command += r'Set-Clipboard -Path $imageFilePaths;'
        # Execute Powershell
        completed = subprocess.run(["powershell", "-Command", powershell_command], capture_output=True)

    def save_figure(self,a):
        fname = f'./browser_outputs/{self.selectionList.value[0]}'
        if self.saveNote.value != '':
            fname += f'_{self.saveNote.value}'
        self.last_save_fname = f'{fname}.png'
        self.figure.savefig(f'{fname}.png',dpi=500,format='png',transparent=True,bbox_inches='tight')
        self.saveNote.value = ''
        print('Figure Saved')

    def export_csv(self,a):
        for f in self.selectionList.value:
            file = self.spe_files[f]
            exportDir = file.fileDir + '//' + '_exported'
            if not os.path.exists(exportDir):
                os.makedirs(exportDir)
            file.export_data(exportDir)
        print('Data Exported')

    def create_widgets(self):
        self.figure_display = widgets.Output()

        self.xmin = widgets.IntText(value=600,step=10,description='wavelength:',layout=widgets.Layout(width="160px",height="30px"))
        self.xmax = widgets.IntText(value=1000,step=10,layout=widgets.Layout(width="80px",height="30px"))
        self.xbounds = widgets.HBox(children=[self.xmin,self.xmax])
        self.ymin = widgets.BoundedIntText(value=0,step=0,min=0,max=10,description='Intensity:',layout=widgets.Layout(width="160px",height="30px"))
        self.ymax = widgets.BoundedIntText(value=500,step=100,min=100,max=10000,layout=widgets.Layout(width="80px",height="30px"))
        self.ybounds = widgets.HBox(children=[self.ymin,self.ymax])

        self.biasNote = widgets.Text('',description='Bias:',layout=widgets.Layout(width="240px",height="30px"))
        self.currentNote = widgets.Text('',description='Current:',layout=widgets.Layout(width="240px",height="30px"))
        self.saveNote = widgets.Text('',description='SaveNote:',layout=widgets.Layout(width="240px",height="30px"))
        self.saveBtn = widgets.Button(description='Save',layout=widgets.Layout(width="78px",height="30px"))
        self.exportBtn = widgets.Button(description='Export',layout=widgets.Layout(width="78px",height="30px"))
        self.copyBtn = widgets.Button(description='Copy',layout=widgets.Layout(width="78px",height="30px"))
        self.vlineBtn = widgets.Button(description='Add Marker',layout=widgets.Layout(width="160px",height="30px"))
        self.removeLines = widgets.Button(description='Remove Markers',layout=widgets.Layout(width="160px",height="30px"))
        self.vlinePosition = widgets.BoundedFloatText(value=855,step=0.1,min=400,max=1200,description='Position:',layout=widgets.Layout(width="160px",height="30px"))

        self.directorySelection = widgets.Select(options=self.directories,description='Folders:',rows=4)
        self.selectionList = widgets.SelectMultiple(options=[],value=[],description='spe Files:',rows=25)

        self.h_output_layout = widgets.HBox(children=[self.saveBtn,self.exportBtn])
        self.h_output_layout_2 = widgets.HBox(children=[self.copyBtn])
        self.v_bounds_layout = widgets.VBox(children=[self.xbounds,self.ybounds,self.biasNote,self.currentNote,self.saveNote],layout=widgets.Layout(border='solid 1px gray',margin='10px 10px 10px 10px',padding='5px 5px 5px 5px'))
  
        self.vline_v_layout = widgets.VBox(children=[self.vlineBtn,self.vlinePosition,self.removeLines,self.h_output_layout,self.h_output_layout_2],layout=widgets.Layout(border='solid 1px gray',margin='10px 10px 10px 10px',padding='5px 5px 5px 5px'))
        self.h_control_layout = widgets.HBox(children=[self.v_bounds_layout,self.vline_v_layout])
        self.v_select_layout = widgets.VBox(children=[self.directorySelection,self.selectionList])
        self.v_display_layout = widgets.VBox(children=[self.figure_display,self.h_control_layout])
        self.h_layout_main = widgets.HBox(children=[self.v_select_layout,self.v_display_layout])
        

        self.xmin.observe(self.handler_update_axes,names='value')
        self.xmax.observe(self.handler_update_axes,names='value')
        self.ymin.observe(self.handler_update_axes,names='value')
        self.ymax.observe(self.handler_update_axes,names='value')

        self.vlinePosition.observe(self.handler_update_vline,names='value')
        self.vlineBtn.on_click(self.handler_add_vline)
        self.removeLines.on_click(self.handler_remove_vlines)

        #self.filterWidth.observe(self.handler_update_filter,names='value')
        #self.filterThreshold.observe(self.handler_update_filter,names='value')
        #self.filterBtn.observe(self.handler_update_data,names='value')

        self.directorySelection.observe(self.handler_update_files,names='value')
        self.selectionList.observe(self.handler_update_data,names='value')

        self.biasNote.observe(self.handler_update_ax_title,names='value')
        self.currentNote.observe(self.handler_update_ax_title,names='value')
        self.saveBtn.on_click(self.save_figure)
        self.exportBtn.on_click(self.export_csv)
        self.copyBtn.on_click(self.copy_figure)

if __name__ == '__main__':
    browser = Spe_Browser()
