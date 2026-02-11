### Single Point STML Acquisition & Tools for Line and Grid Acquisition ###

# Import
import math
import numpy as np
import os
import importlib
import subprocess
import time

# Import Custom Modules
import datalogger_base
importlib.reload(datalogger_base)
import spe_read
importlib.reload(spe_read)

class STMLBase():

    def __init__(self,nbase,lf,filename_increment:bool=True,additional_metadata_dict:dict={},datalogger_test_is_performed:bool=False):

        """
        STMLBase Class for Single Point STML Acquisition & Tools for Line and Grid Acquisition.
        
        Input:
        ----------
        nbase: initialized nanonis_connect_base class
        lf: initialized lightfield module
        filename_increment: If True, the filename will be incremented by 1 for each new measurement.
        additional_metadata_dict: Dictionary with additional metadata to be added to the new file.
        datalogger_test_is_performed: If True, the datalogger channel test is performed at the beginning.
        """

        # Initialized Module Base Classes
        self.nbase = nbase
        self.lf = lf

        # Initalize DataLogger_Base
        self.dlbase = datalogger_base.DataLogger_Base(nbase)
        # Open DataLogger Module
        if not self.dlbase._datalog_open:
            self.dlbase._open()

        # Internal Variables
        self.HARDWARE_DELAY_SEC = 5
        self.filename_increment = filename_increment
        self.additional_metadata_dict = additional_metadata_dict
        self.datalogger_test_is_performed = datalogger_test_is_performed
        self.current_session_path = self.nbase.utilities.SessionPathGet()
        self.AQUIRE_LOCK = False

        ### Run DataLogger Channel Test ###
        if not self.datalogger_test_is_performed:
            try:
                self.datalogger_channel_test()
                self.datalogger_test_is_performed = True
                print('Datalogger channel test successfully performed.')
            except Exception as e:
                self.datalogger_test_is_performed = False
                print('Error in datalogger channel test:', e)

        ### Active Process Check ###
        # Check if "ids_peak_cockpit.exe" is running
        if self.does_process_exist("ids_peak_cockpit.exe"):
            print('Warning: "ids_peak_cockpit.exe" is running. When camera readout is active this might overtax the network connection.\
                  Please make sure to pause camera readout before starting the measurement.')

    ### Main Measurement Functions ###

    def aquire_stml(self,filename:str,spe_data_return=False,print_out=True,one_time_additional_metadata_dict:dict={},logger=None):
        """
        Aquires single spectrum, while running datalogger and creates a joined output file. 

        Input:
        ----------
        filename: str
            Filename for the output files. If the filename already exists, a new filename will be generated.
        spe_data_return: bool
            If True, the function returns the data as a dictonary from the .spe file.
        """

        # Check if acquisition is locked
        if self.AQUIRE_LOCK:
            print('Warning: Acquisition (" STMLBase.aquire_stml) is locked. Please wait until the previous acquisition is finished.')
            return None
        self.AQUIRE_LOCK = True

        # Current session path
        self.current_session_path = self.nbase.utilities.SessionPathGet()

        # Create 'raw_stml_data' folder in current session path if it does not exist
        raw_stml_data_path = self.current_session_path + r'\raw_stml_data'
        if not os.path.exists(raw_stml_data_path):
            os.makedirs(raw_stml_data_path)
        
        # Update datadirectory in lightfield
        self.lf.set_data_directory_external(dataDirectory=self.current_session_path)

        # Change filename increment in lightfield
        self.lf.change_file_save_increment_bool(False)

        # Make sure that the datalogger is open & is not running
        if not self.dlbase._datalog_open:
            self.dlbase._open()
        self.dlbase.stop()

        # Get module parameters
        module_parameters = self.collect_module_parameters(filename)

        # Update module parameters with one_time_additional_metadata_dict
        if one_time_additional_metadata_dict is None:
            one_time_additional_metadata_dict = {}
        if one_time_additional_metadata_dict != {}:
            module_parameters.update(one_time_additional_metadata_dict)

        # Set DataLogger Properties
        last_sxm = module_parameters['Last aquired sxm file']
        comment = "Last sxm file before spectrum: "+ str(last_sxm) + " SPE file: " + filename + ".spe"

        # Make sure that filename is unique
        if filename + '.dat' in os.listdir(self.current_session_path):
            filename = self.interative_filename_generator(filebase='stml_',fileending='.dat',other_expected_filebases_endings=None)
            print('Warning: File already exists. Creating new filename: ',filename)

        # Make the number of recorded points equal to the number of points in the spectrum
        exposure_time = module_parameters['Exposure Time [ms]'] / 1000 # in seconds
        number_of_desired_datapoints = 1340
        RTOversampl = self.nbase.utilities.RTOversamplGet()
        RTFreq = self.nbase.utilities.RTFreqGet()
        ave =int(RTFreq / RTOversampl / (number_of_desired_datapoints / exposure_time)/4) # to increase the number of points
        self.dlbase.set_prop(filename,comment=comment,averaging=ave)
        
        # Starting Acquisition
        if print_out:
            print("Starting Acquisition of "+ filename + " ...")

        # Start DataLogger
        self.dlbase.start()
        if logger is not None:
            logger(f"DataLogger started for file: '{filename}'")
        # Acquire Lightfield Data
        if not self.lf.acquiring:
            self.lf.acquire_data(filename,print_out=print_out)
        time.sleep(0.3)
        # Wait until Lightfield acquisition is finished running
        #start_time = time.time()

        while self.lf.experiment.IsRunning:
            time.sleep(0.3)
        #print('Lightfield acquisition finished.')
        # Wait until it is ready to run again
        while not self.lf.experiment.IsReadyToRun:
            time.sleep(0.3)
        #print('Lightfield is ready to run again.')
        time.sleep(0.1) # give the data logger a few extra points
        # Stop DataLogger
        self.dlbase.stop()
        if logger is not None:
            logger(f"DataLogger stopped for file: '{filename}'")

        ### Wait until files are written ###
        # first: SPE file from lightfield
        if logger is not None:
            logger('Waiting for LightField and DataLogger files')
        FOUND_SPE_FILE = False
        FOUND_DAT_FILE = False
        try:
            for i in range(int(5//.1)):
                if os.path.exists(self.current_session_path+r"\\"+filename+'.spe'):
                    spe_dict = spe_read.load_spe_file_to_dic(self.current_session_path+r"\\"+filename+'.spe')
                    if logger is not None:
                        logger('Lightfield .spe file found.')
                    FOUND_SPE_FILE = True
                    break
                time.sleep(0.1)
        except Exception as e:
            print('Error: ',e)
            if logger is not None:
                logger('Error while waiting for .spe file: '+str(e))
        # second: dat file from Nanonis DataLogger
        try:
            datalogger_file_filepath = f'{self.current_session_path}\{filename}00001.dat'
            for i in range(int(5//.1)):
                dlstatus = self.nbase.datalog.StatusGet() # start_time, acquisition_elapsed_hours, acquisition_elapsed_minutes, acquisition_elapsed_seconds, stop_time, saved_file_path, saved_points
                if dlstatus[5] == datalogger_file_filepath and os.path.getsize(os.path.join(self.current_session_path,datalogger_file_filepath)) > 0:
                    if logger is not None:
                        logger('Datalogger .dat file found.')
                    FOUND_DAT_FILE = True
                    break
                time.sleep(0.1)
        except Exception as e:
            print('Error: ',e)
            if logger is not None:
                logger('Error while waiting for .dat file: '+str(e))

        # Create .txt file with module parameters
        if logger is not None:
            logger('Creating .txt file with module parameters')
        txt_str = ''
        for key in list(module_parameters.keys()):
            txt_str += key + ': ' + str(module_parameters[key]) + '\n'
        with open(self.current_session_path + '\\' + filename + '.txt', 'w') as f:
            f.write(txt_str)
            f.close()

        time.sleep(0.1) 

        ### Create joined .dat file ###
        if logger is not None:
            logger('Creating joined .dat file')
        # Load spe file & datalogger file
        assert FOUND_SPE_FILE, "Lightfield .spe file was not found."
        assert FOUND_DAT_FILE, "Datalogger .dat file was not found."
        new_filepath = self.current_session_path + r"\\"+filename+'.dat'
        spe_data = self.joined_file_creator(spe_dict,datalogger_file_filepath,new_filepath,additional_metadata=module_parameters,new_title='Experiment\tSTML',return_data=spe_data_return)
        
        ### clean up files and reset settings ###
        # Move .spe, datalogger.dat and metadata.txt file and .csv (if exists) to 'raw_stml_data' folder
        files_to_move = ['.spe', '-raw.spe', '00001.dat', '.txt', '.csv']
        for extension in files_to_move:
            source_file = os.path.join(self.current_session_path, filename + extension)
            destination_file = os.path.join(self.current_session_path, 'raw_stml_data', filename + extension)
            if os.path.exists(source_file):
                os.rename(source_file,destination_file)

        # Wait until all operations are finished
        time.sleep(0.1)

        # Delete DataLogger Filename & Comment
        self.dlbase.set_prop('default',comment='')

        # Reset filename, increment and time in lightfield
        self.lf.set_file_save_name('spec')
        self.lf.reset_filename_increment()
        self.lf.change_file_save_add_time_bool(True)
        if not self.lf.get_file_save_increment_bool():
            self.lf.change_file_save_increment_bool(True)

        # Unlock Acquisition
        self.AQUIRE_LOCK = False

        # Aquisition finished
        if print_out:
            print("Acquisition of "+ filename + " finished")
            if logger is not None:
                logger(f"Acquisition of '{filename}' finished.")
        # Return spe data if requested
        if spe_data_return:
            return spe_data
        else:
            return None

    ### Helper Functions ###
    
    def does_process_exist(self,process_name):
        """Checks if a process with the name process_name is running."""
        progs = str(subprocess.check_output('tasklist'))
        if process_name in progs:
            return True
        else:
            return False

    def last_acquired_sxm(self,filepath):
        """Returns the filename of the last sxm file in the session folder."""
        files = os.listdir(filepath)
        files = [file for file in files if file.endswith('.sxm')]
        files_sorted = sorted(files, key=lambda x: os.path.getctime(os.path.join(filepath, x)))
        if len(files_sorted) > 0:
            return files_sorted[-1]
        else:
            return None
    
    def datlogger_file_finder(self,filepath,filename):
        """Finds the datalogger file with the filename in the filepath."""
        files = os.listdir(filepath)
        files = [file for file in files if file.startswith(filename) and file.endswith('.dat')]
        files_sorted = sorted(files, key=lambda x: os.path.getmtime(os.path.join(filepath, x)))
        if len(files_sorted) > 1 or len(files_sorted) == 0:
            print('Warning: More than one or no datalogger file found.')
            return None
        return files_sorted[0]
    
    def joined_file_creator(self,spe_dict,datalogger_file_filepath,new_filepath,additional_metadata=None,new_title=None,return_data=False):
        """
        Creates a joined dat files from a datalogger file and a spe file.
        Bin averages the datalogger file to the number of points in the spe file and cuts off the last bins if the number of points is too large.

        Input:
        ----------
        spe_dict: Dictionary with the data from the spe file.
        datalogger_file_filepath: Path to the datalogger file.
        new_filepath: Path to the new joined file.
        additional_metadata: Dictionary with additional metadata to be added to the new file.
        new_title: New title for the dat file
        return_data: If True, returns the dictionary with 'wavelength' and 'intensity' data.

        Output:
        ----------
        If return_data is True, returns the dictionary with 'wavelength' and 'intensity' data.
        """

        # Create metadata dictionary
        new_metadata = {'Spectrometer Exposure Time (ms)':spe_dict['exp_time_ms'],
                    'Spectrometer Selected Grating Center Wavelength (nm)':spe_dict['selected_grating_center_wavelength'],
                    'Spectrometer Selected Grating Density':spe_dict['selected_grating_density']}
        if additional_metadata is not None:
            new_metadata.update(additional_metadata)

        # Create new columns dictionary
        new_columns_data = {'Intensity': spe_dict['intensity'],
                            'Wavelength': spe_dict['wavelength']}
        
        # Load the datalogger file from datalogger_file_filepath
        for i in range(int(5//.1)):
            try:
                with open(datalogger_file_filepath, 'r') as file:
                    lines = file.readlines()
            except PermissionError:
                time.sleep(.1)
                continue
            except Exception as e:
                print('Error:',e)

        # Overwrite the title if new_title is provided
        if new_title is not None:
            lines[0] = new_title + '\n'
        
        # Find the index of [DATA] section
        data_index = lines.index('[DATA]\n')

        # Extract column_names and column_data from datalogger file
        column_names = lines[data_index + 1].strip().split('\t')
        column_data = {}
        for name in column_names:
            column_data[name] = []
        for line in lines[data_index + 2:]:
            values = line.strip().split('\t')
            for i, value in enumerate(values):
                column_name = column_names[i]
                column_data[column_name].append(float(value))

        # Bin averaging
        column_data_binned = {}
        for name in column_names:
            column_data_binned[name] = []
            desired_num_points = 1340
            bin_size = len(column_data[name]) // desired_num_points
            if bin_size != 0:
                current_bin = 0
                bin_sum = 0
                for value in column_data[name]:
                    bin_sum += value
                    current_bin += 1
                    if current_bin == bin_size:
                        column_data_binned[name].append(bin_sum / bin_size)
                        current_bin = 0
                        bin_sum = 0
                if current_bin > 0:
                    average_value = bin_sum / current_bin
                    column_data_binned[name].append(average_value)
                # Cut off the last bins if the number of points is too large
                if len(column_data_binned[name]) > desired_num_points:
                    column_data_binned[name] = column_data_binned[name][:desired_num_points]
            else:
                if len(column_data[name]) > 0:
                    column_data_binned[name] = [np.average(column_data[name])] * desired_num_points
                    print('Warning: The number of points in the datalogger file is less than the necessary number of points for the spectrometer.\
                        Extending .dat file data with averages of existing data.')
                # The number of points in the datalogger file is less than the necessary number of points for the spectrometer
                # In this case, only wavelength and intensity are used from the spectrometer data
                print('Warning: The number of points in the datalogger file is less than the necessary number of points for the spectrometer.\
                    New .dat file only contains wavelength and intensity.')
                column_data_binned = {}
                break
        
        # Add additional Time Step column
        if list(column_data_binned.keys()) != []:
            time_step = [i * spe_dict['exp_time_ms'] for i in range(0,len(column_data_binned[list(column_data_binned.keys())[0]]))]
            column_data_binned['Time'] = time_step
        
        # Update metadata
        metadata_lines = []
        for key, value in new_metadata.items():
            metadata_lines.append(f"{key}\t{value}\n")
        lines = lines[:data_index-1] + metadata_lines + ["\n"] +[str(lines[data_index])]

        # Create a new dictioanry with all column data
        all_column_data = {}
        all_column_data.update(column_data_binned)
        all_column_data.update(new_columns_data)

        # Create a new column names line
        all_column_data_names = list(all_column_data.keys())
        all_column_data_line = '\t'.join(all_column_data_names) + '\n'
        lines = lines+ [str(all_column_data_line)]

        # Append new column data
        # Make sure that all columns have the same length
        column_data_len = []
        for key in all_column_data.keys():
            column_data_len.append(len(all_column_data[key]))
        if all(x == column_data_len[0] for x in column_data_len):
            pass
        else:
            if any(value == 1 for value in column_data_len):
                
                for key in all_column_data.keys():
                    if len(all_column_data[key]) == 1:
                        all_column_data[key] = all_column_data[key] * column_data_len[0]
                print('Warning: One column has only one value.')
            else:
                print('all_column_data',all_column_data)
                raise ValueError('All columns must have the same length')

        for i in range(0,len(all_column_data[list(all_column_data.keys())[0]])):
            values = []
            for column_name in all_column_data_names:
                values.append(str(all_column_data[column_name][i]))
            lines.append('\t'.join(values) + '\n')

        # Write the updated data to the file
        with open(new_filepath, 'w') as file:
            file.writelines(lines)
        
        if return_data:
            return new_columns_data
    
    def datalogger_channel_test_subroutine(self,set_custom_channels_to_record=False):

        # Desired Channels: Current (A)	Bias (V)	X (m)	Y (m)	Z (m)
        desired_channels = ['Current (A)','Bias (V)','X (m)','Y (m)','Z (m)']
            
        # Make sure that the datalogger is open & is not running
        if not self.dlbase._datalog_open:
            self.dlbase._open()
        self.dlbase.stop()

        # Set the properties of the datalogger
        filename = time.strftime("%Y%m%d_%H%M%S") + '_datalogger_test'
        if type(set_custom_channels_to_record) == bool:
            if not set_custom_channels_to_record:
                self.dlbase.set_prop(basename=filename,
                                    comment="Test file for datalogger channels.",
                                    averaging=1)
        elif type(set_custom_channels_to_record) == list:
            self.dlbase.set_prop(basename=filename,
                                comment="Test file for datalogger channels.",
                                averaging=1,
                                set_custom_channels_to_record=set_custom_channels_to_record)

        # Start the datalogger
        self.dlbase.start()

        # Wait for one second
        time.sleep(0.5)

        # Stop the datalogger
        self.dlbase.stop()

        # Wait for the file to be written
        time.sleep(self.HARDWARE_DELAY_SEC)

        # Load the datalogger file
        datalogger_file_filepath = os.path.join(self.current_session_path, filename + '00001.dat')

        try:
            with open(datalogger_file_filepath, 'r') as file:
                lines = file.readlines()
        except FileNotFoundError:
            print('Error: Datalogger file ',datalogger_file_filepath,' not found.')
            raise FileNotFoundError

        # Find the index of [DATA] section
        data_index = lines.index('[DATA]\n')

        # Extract column_names and column_data from datalogger file
        column_names = lines[data_index + 1].strip().split('\t')

        # Check if the desired channels are in the column_names
        all_channels_found = True
        missing_channels = []
        for channel in desired_channels:
            if channel not in column_names:
                all_channels_found = False
                missing_channels.append(channel)
            
        # Delete the datalogger file
        os.remove(datalogger_file_filepath)

        # Reset datalogger basename and comment
        self.dlbase.set_prop('default',comment='default',averaging=1)

        return all_channels_found, missing_channels

    def datalogger_channel_test(self):
        """Aquires a short .dat file and checks if the channels are working."""

        print('Running DataLogger Channel Test...')

        all_channels_found, missing_channels = self.datalogger_channel_test_subroutine(set_custom_channels_to_record=False)

        if not all_channels_found:
            print('Warning: The following channels are missing in the datalogger file:', missing_channels)
            print('Might be necessary to check the channel configuration in datalogger_base.py/dataLogger_Base.set_prop() function.')
            print('Testing second set of channel indexes since not channel index not aways persistent...')
            alternative_channels_to_record_indexes = [0, 81, 87, 88, 89]
            all_channels_found, missing_channels = self.datalogger_channel_test_subroutine(set_custom_channels_to_record=alternative_channels_to_record_indexes)
            if not all_channels_found:
                print('Warning: The following channels are missing in the datalogger file:', missing_channels)
                print('Might be necessary to check the channel configuration in datalogger_base.py/dataLogger_Base.set_prop() function.')
            else:
                print('All channels found in the datalogger file.')
        else:
            print('All channels found in the datalogger file.')

    def interative_filename_generator(self,filebase:str="stml_",fileending:str=".dat",other_expected_filebases_endings:list[list[str,str]]=None,other_expected_filebases_endings_stml:bool=True):
        """
        Searches the current_session_path for the next available filename with the filebase and fileending and adds the next number 5 digit number. Ex. stml_00001

        Returns
        ----------
        new_filename: str
            Next available filename.
        fileending: str
            Fileending of the file.
        other_expected_filebases_endings: list[list[str,str]]
            List of other expected filebases and endings. If these are found, the next number is increased by 1.
        other_expected_filebases_endings_stml: bool
            If True, the expected filebases and endings for the STML measurements are included.
        """

        if other_expected_filebases_endings_stml:
            other_fnames = [["stml_",".csv"],["stml_",".spe"],["stml_",".txt"],["stml_","-raw.spe"]]
            if other_expected_filebases_endings is not None:
                other_expected_filebases_endings = other_expected_filebases_endings + other_fnames
            else:
                other_expected_filebases_endings = other_fnames

        def list_files_in_folder_and_subfolders(folder_path):
            files_in_folder_and_subfolders = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    files_in_folder_and_subfolders.append(file)
            return files_in_folder_and_subfolders

        files_in_session_folder = os.listdir(self.current_session_path)
        # files_with_filebase = [f for f in files_in_session_folder if f.startswith(filebase) and f.endswith(fileending) and len(f) == len(filebase)+5+len(fileending)] #<-- Does not consider _repXXX files
        files_with_filebase = [f for f in files_in_session_folder if f.startswith(filebase) and f.endswith(fileending)]
        files_with_filebase.sort()

        if len(files_with_filebase) == 0:
            next_file_number = 1
        else:
            # next_file_number = int(files_with_filebase[-1].split('.')[0].split('_')[-1]) + 1 #<-- Does not consider _repXXX files
        
            last_file = files_with_filebase[-1].split('.')[0]
            if '_rep' in last_file:
                last_file_main_part = last_file.split('_rep')[0]
                next_file_number = int(last_file_main_part.split('_')[-1]) + 1
            else:
                next_file_number = int(files_with_filebase[-1].split('.')[0].split('_')[-1]) + 1
            
            if other_expected_filebases_endings is not None:
                need_to_increase_file_number = False
                for other_expected_filebase_ending in other_expected_filebases_endings:
                    files_in_session_folder_and_subfolders = list_files_in_folder_and_subfolders(self.current_session_path)

                    other_filebase = other_expected_filebase_ending[0]
                    other_fileending = other_expected_filebase_ending[1]
                    other_filename = other_filebase + str(next_file_number).zfill(5) + other_fileending

                    if other_filename in files_in_session_folder_and_subfolders:
                        if not need_to_increase_file_number:
                            need_to_increase_file_number = True

                if need_to_increase_file_number:
                        next_file_number = next_file_number + 1

        next_filename = filebase + str(next_file_number).zfill(5)

        return next_filename

    def collect_module_parameters(self,filename):

        module_parameters = {}

        # Nanonis Parameters
        current_session_path = self.nbase.utilities.SessionPathGet()
        tip_position_x, tip_position_y = self.nbase.followme.XYPosGet(Wait_for_newest_data=True)
        tip_position_z = self.nbase.zcontroller.ZPosGet()
        current_bias = self.nbase.biasmodule.Get()
        current_current = self.nbase.currentmodule.Get()

        module_parameters.update({'X (m)': tip_position_x,
                        'Y (m)': tip_position_y,
                        'Z (m)': tip_position_z,
                        'Bias [V]': current_bias,
                        'Current [A]': current_current})

        # Lightfield Parameters
        exposure_time = self.lf.get_exposure_time()
        grating_index = self.lf.get_grating()
        gratings = {'0': '150 g/mm', # Density: 150 g/mm Blaze: 800 nm
                    '1': '600 g/mm', # Density: 600 g/mm Blaze: 750 nm
                    '2': '1200 g/mm', # Density: 1200 g/mm Blaze: 750 nm
                    }
        selected_grating = gratings[str(grating_index)]
        center_wavelength = self.lf.get_grating_center_wavelength()
        sensor_temperature = self.lf.get_temperature()

        module_parameters.update({'Exposure Time [ms]': exposure_time,
                        'Selected Grating': selected_grating,
                        'Center Wavelength [nm]': center_wavelength,
                        'Sensor Temperature [C]': sensor_temperature})

        # Related Filenames
        last_sxm = self.last_acquired_sxm(current_session_path)
        spe_filename = filename + '.spe'
        dat_filename = filename + '00001.dat'

        module_parameters.update({'Last aquired sxm file': last_sxm,
                        'SPE Filename': spe_filename,
                        'DAT Filename': dat_filename})

        # Additional Metadata
        if self.additional_metadata_dict is not None:
            if len(self.additional_metadata_dict) > 0 and type(self.additional_metadata_dict) == dict:
                module_parameters.update(self.additional_metadata_dict)

        return module_parameters

    ### Pattern Functions ###

    def aquire_optical_spectrum_on_point(self,point_x,point_y,filename):
        # Go to the point
        self.nbase.followme.XYPosSet(point_x,point_y,Wait_end_of_move=True)
        time.sleep(2)

        # Acquire spectrum
        self.aquire_stml(filename)

    def calculate_grid_points(self,number_x_points, number_y_points, center_x, center_y, width, height, angle):

        grid_points = []
        
        # Calculate the step size for x and y directions
        step_x = width / (number_x_points)
        step_y = height / (number_y_points)
        
        # Convert the angle to radians
        angle_rad = math.radians(angle)
        
        # Calculate the starting position of the grid
        start_x = center_x - (width / 2)
        start_y = center_y - (height / 2)
        
        for i in range(number_y_points+1):
            for j in range(number_x_points+1):
                # Calculate the current position based on the grid index
                current_x = start_x + j * step_x
                current_y = start_y + i * step_y
                
                # Rotate the current position around the center point
                rotated_x = center_x + (current_x - center_x) * math.cos(angle_rad) - (current_y - center_y) * math.sin(angle_rad)
                rotated_y = center_y + (current_x - center_x) * math.sin(angle_rad) + (current_y - center_y) * math.cos(angle_rad)
                
                # Add the rotated position to the grid points
                grid_points.append((rotated_x, rotated_y))
        
        return grid_points

    def aquire_optical_spectrum_on_grid_points(self,filebase):
        print('Acquiring Optical Spectra on a grid of points...')

        ## Get the grid parameters from the Pattern Module
        number_x_points, number_y_points, center_x, center_y, width, height, angle = self.nbase.pattern.GridGet()

        ## Calculate the grid points
        grid_points = self.calculate_grid_points(number_x_points, number_y_points, center_x, center_y, width, height, angle)

        ## Loop through the grid points
        for i in range(len(grid_points)):
            point_x, point_y = grid_points[i]
            filename = filebase + '_x' + str(i)
            self.aquire_optical_spectrum_on_point(point_x, point_y, filename)
            
        print('grid loop complete...')

    def acquire_optical_spectrum_on_line_points(self,filebase):
        """
        Acquires optical spectra on a line of points.
        """
        print('Acquiring Optical Spectra on a line of points...')

        ## Get the line parameters from the Pattern Module
        number_of_points, point1_x, point1_y, point2_x, point2_y = self.nbase.pattern.LineGet()

        ## Calculate the points along the line
        points_x_coordinates = np.linspace(point1_x, point2_x, number_of_points)
        points_y_coordinates = np.linspace(point1_y, point2_y, number_of_points)

        ## Loop through the line points
        for i in range(len(points_x_coordinates)):
            point_x, point_y = points_x_coordinates[i], points_y_coordinates[i]
            filename = filebase + '_x' + str(i)
            self.aquire_optical_spectrum_on_point(point_x, point_y, filename)
            
        print('line loop complete...')

    ### STM Parameters Change Tools ###
    
    def set_new_bias_current_with_ref_location(self,loc_data,new_bias,new_current):
        """
        Set new bias and current with reference location while in feedback. Waits for 2 seconds after each command.
        
        Input:
        ----------
        loc_data= {'x': x_position(m), 'y': y_position(m)}
        new_bias= new bias voltage(V)
        new_current= new current(A)
        """
        self.nbase.zcontroller.OnOffSet(True)
        time.sleep(2)
        self.nbase.followme.XYPosSet(loc_data['x'],loc_data['y'], Wait_end_of_move=True) # set reference position
        time.sleep(2)
        self.nbase.zcontroller.SetpntSet(new_current)
        time.sleep(2)
        self.nbase.bias.Set(new_bias)
        time.sleep(2)
        self.nbase.zcontroller.OnOffSet(False)
        time.sleep(2)
        print('New Bias '+str(new_bias)+' V and Current '+str(new_current)+' A set.')

    def get_current_xy_location(self):
        """
        Returns the current XY location.
        
        Output:
        ----------
        loc_data= {'x': x_position(m), 'y': y_position(m)}
        """
        x_position, y_position = self.nbase.followme.XYPosGet(Wait_for_newest_data=True)
        loc_data = {'x': x_position, 'y': y_position}
        print('Current Location: ', loc_data)
        return loc_data
    
