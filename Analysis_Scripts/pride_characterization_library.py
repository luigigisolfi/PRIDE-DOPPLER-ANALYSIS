#!/usr/bin/env python
# coding: utf-8
from astroquery.jplhorizons import Horizons
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import os
from scipy.interpolate import CubicSpline
import regex as re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from astropy.time import Time
import allantools
from matplotlib import cm
from datetime import datetime, timedelta, timezone
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from collections import defaultdict
import pandas as pd
import re
import csv
from scipy.stats import norm
import random

########################################################################################################################################
##################################################### Main Class Definition ############################################################
########################################################################################################################################
class PrideDopplerCharacterization:

    """
    This class allows fdets (PRIDE) data characterization and pre-post processing for the following ESA missions:

    JUICE:   [experiments: EC094A, EC094B] + [data between Apr - May 2023]
    MEX:     [experiment: GR035 (Phobos Flyby)], mex_solar_conjunction_2021)
    INSIGHT: [experiments: ED045A,ED045C,ED045D,ED045E,ED045F]

    To be included:
    VEX,
    MRO,
    what else...?

    """

    def __init__(self):
        self.result = 0
        self.process_fdets = self.ProcessFdets()
        self.analysis = self.Analysis

    ########################################################################################################################################
    ##################################################### ProcessFdets class ###############################################################
    ########################################################################################################################################

    # Function to extract data from the file
    class ProcessFdets:

        def __init__(self):
            self.result = 0
            self.Utilities = PrideDopplerCharacterization.Utilities()
            self.Analysis = PrideDopplerCharacterization.Analysis(self, self.Utilities)
            self.ProcessVexFiles = PrideDopplerCharacterization.ProcessVexFiles(self, self.Utilities)

        ########################################################################################################################################
        ########################################################################################################################################

        def get_observation_date(self, filename):
            """
             Description:
                Retrieves the observation date from a given file. The function first attempts to extract the date from the header line. If unsuccessful, it tries alternative methods based on the column names in the data.

             Inputs:
                - `filename` (str, required): Path to the file containing observation data.

             Outputs:
                - Returns the observation date as a string in `YYYY.MM.DD` or `YYYY-MM-DD` format.
                - If the observation date cannot be determined, prints an error message and skips processing.
            """
            with open(filename, 'r') as file:
                lines = file.readlines()
                header_match = re.search(r'Observation conducted on',lines[0])
                if not header_match:
                    print(f'Invalid File: {filename}. Reason: Invalid Observation Header: {header_match}. Skipping...\n')

                else:

                    date_match = re.search(r'Observation conducted on (\d{4}\.\d{2}\.\d{2})', lines[0])
                    if date_match:
                        # Return the date as a string in YYYY.MM.DD format
                        self.observation_date = date_match.group(1)
                        return (self.observation_date)
                    else:
                        print(f'Invalid File: {filename}. Reason: Invalid Observation Date Header. First Header Line is:\n{lines[0]}\n Trying Another Way...\n')

                    if not self.observation_date:
                        self.first_col_name = self.get_columns_names(filename)['first_col_name']
                        if self.first_col_name.strip() == 'UTC Time':
                            lines = file.readlines()
                            parts = lines[5].strip().split()
                            utc_time = parts[0]
                            utc_datetime = datetime.strptime(utc_time, "%Y-%m-%dT%H:%M:%S.%f")
                            self.observation_date = utc_datetime[0].strftime("%Y-%m-%d")
                            return self.observation_date


                        elif self.first_col_name.strip() == 'Modified Julian Date' or self.first_col_name.strip() == 'Modified JD':
                            lines = file.readlines()
                            parts = lines[5].strip().split()
                            mjd = float(parts[0])
                            self.observation_date = self.Utilities.mjd_to_utc(mjd)
                            return(self.observation_date)

                        else:
                            print(f'Could Not Retrieve Observation Date from File: {filename}. Skipping...\n')

        ########################################################################################################################################
        ########################################################################################################################################

        def get_base_frequency(self, filename):
            """
            Description:
                Extracts the base frequency from the given file. The function first ensures that a valid observation
                date is present in the file before attempting to retrieve the base frequency.

            Inputs:
                - `filename` (str, required): Path to the file containing observation data.

            Outputs:
                - Returns the base frequency as a float in Hz.
                - If the base frequency is invalid, prints an error message with recommended corrective actions.
                - If no valid observation date is found, prints a message and skips processing.
            """

            observation_date_flag = self.get_observation_date(filename)
            if observation_date_flag:
                # Open the file and read lines
                with open(filename, 'r') as file:
                    lines = file.readlines()

                    try:
                        self.base_frequency = float(lines[1].split(' ')[3])*1e6 #the base frequency in the fdets is expressed as MHz

                        #BEWARE THIS ELIF!!!!!!!! We do not know exactly the InSight frequencies
                        #elif float(lines[1].split(' ')[3]) == 0.0:
                        #    self.base_frequency =  8416.49*1e6 ##Handle InSight data, which has bad headers,
                        ##but in the VEX file you can see the channel frequencies .
                    except:
                        print(f'Not a valid base frequency: {float(lines[1].split(" ")[3])*1e6}.\n '
                              'Recommended action: correct base frequencies of your dataset based on the vex files.\n '
                              'You can use the correct_baseband_fdets.py script. \n')

                        #if lines[1].split(' ')[3] == '2xxx.xx':
                        #    self.base_frequency = 8412*1e6 #Handle MEX data, which has some bad headers.
                        #                                    #Assuming all observations where at 8412 MHz (based on the good headers)

                return self.base_frequency
            else:
                print(f'Pointless to retrieve base frequency, as there is no valid observation date in the header.')

            ########################################################################################################################################
        ########################################################################################################################################

        def get_station_name_from_file(self, fdets_file_name):
            """
            Description:
                Extracts the station name from the given file name based on a predefined pattern.

            Inputs:
                - `fdets_file_name` (str, required): The name of the file from which to extract the station name.

            Outputs:
                - Returns the station name as a string if a match is found.
                - Returns `None` if the file name does not match the expected pattern.
            """
            fdets_filename_pattern = r"Fdets\.\w+\d{4}\.\d{2}\.\d{2}(?:-\d{4}-\d{4})?\.(\w+)(?:\.complete)?\.r2i\.txt"
            match = re.search(fdets_filename_pattern, fdets_file_name)
            if match:
                self.receiving_station_name = match.group(1)
                return self.receiving_station_name
            else:
                return None

        def get_columns_names(self, filename):
            """
            Description:
                Extracts column names from the third line of a given file and determines the number of columns present.
                The function supports two formats with either 5 or 6 columns and assigns appropriate column names.

            Inputs:
                - `filename` (str, required): Path to the file containing observation data.

            Outputs:
                - Returns a dictionary containing:
                    - `number_of_columns` (int): The total number of columns detected.
                    - `first_col_name` (str): Name of the first column.
                    - `second_col_name` (str): Name of the second column.
                    - `third_col_name` (str): Name of the third column.
                    - `fourth_col_name` (str): Name of the fourth column.
                    - `fifth_col_name` (str): Name of the fifth column.
                    - `sixth_col_name` (str, optional): Name of the sixth column if present.
                - If the number of columns is invalid, prints an error message and skips processing.
            """

            with open(filename, 'r') as file:
                lines = file.readlines()

                # Extract column names from row 3 (index 2)
                columns_header = lines[2].strip()

                # Split the header line by '|'
                columns_header_parts = columns_header.split('|')

                if columns_header_parts[0] is None:
                    self.n_columns = 0

                if columns_header_parts[-1] == '':
                    self.n_columns = len(columns_header_parts) - 1 #some column headers end with |
                else:
                    self.n_columns = len(columns_header_parts)  # others dont ...

                if self.n_columns == 5:  # Assign column names from the extracted parts

                    try: #sometimes there is Format:, others there is not...
                        self.first_col_name = columns_header_parts[0].split(':',1)[1] #Typically UTC time (in YY-MM-DD for JUICE)
                        # or Time(UTC) [s] for VEX
                    except:
                        self.first_col_name = columns_header_parts[0]   # Typically Modified Julian Date or Modified JD
                    self.second_col_name = columns_header_parts[1]  # Signal-to-Noise
                    self.third_col_name = columns_header_parts[2]  # Spectral Max
                    self.fourth_col_name = columns_header_parts[3]  # Freq. Detection
                    self.fifth_col_name = columns_header_parts[4]  # Doppler Noise

                    return {
                        'number_of_columns': self.n_columns,
                        'first_col_name': self.first_col_name,
                        'second_col_name': self.second_col_name,
                        'third_col_name': self.third_col_name,
                        'fourth_col_name': self.fourth_col_name,
                        'fifth_col_name': self.fifth_col_name,
                    }

                elif self.n_columns == 6:

                    try:  #sometimes there is Format:, others there is not...
                        self.first_col_name = columns_header_parts[0].split(':',1)[1]   # Typically Modified Julian Date or Modified JD
                    except:
                        self.first_col_name = columns_header_parts[0]   # Typically Modified Julian Date or Modified JD

                    try:
                        self.second_col_name = columns_header_parts[1].split(':',1)[1]  # Typically Time(UTC) [s]
                    except:
                        self.second_col_name = columns_header_parts[1] # Typically Time(UTC) [s]

                    self.third_col_name = columns_header_parts[2]  # Signal-to-Noise
                    self.fourth_col_name = columns_header_parts[3]  # Spectral Max
                    self.fifth_col_name = columns_header_parts[4]  # Freq. Detection
                    self.sixth_col_name = columns_header_parts[5]  # Doppler Noise

                    return {
                        'number_of_columns': self.n_columns,
                        'first_col_name': self.first_col_name,
                        'second_col_name': self.second_col_name,
                        'third_col_name': self.third_col_name,
                        'fourth_col_name': self.fourth_col_name,
                        'fifth_col_name': self.fifth_col_name,
                        'sixth_col_name': self.sixth_col_name,
                    }

                else:
                    print(f'Invalid number of columns: {self.n_columns}. Skipping File: {filename}...\n')



        def extract_parameters(self, filename):

            fdets_filename_pattern = r"Fdets\.\w+\d{4}\.\d{2}\.\d{2}(?:-\d{4}-\d{4})?\.(\w+)(?:\.complete)?\.r2i\.txt"
            match = re.search(fdets_filename_pattern, filename)

            if not match:
                print(f'No match found between pattern: {fdets_filename_pattern} and filename: {filename}')
                return None

            self.receiving_station_name = match.group(1)
            self.observation_date = self.get_observation_date(filename)
            self.base_frequency = self.get_base_frequency(filename)

            if self.observation_date is None:
                return None

            columns_dict = self.get_columns_names(filename)
            self.n_columns = columns_dict['number_of_columns']

            # Predefine empty lists
            utc_time = []
            signal_to_noise = []
            doppler_noise_hz = []
            frequency_detection = []

            # Setup column indices based on format
            if self.n_columns == 5:
                idx_time, idx_snr, idx_freq, idx_dopp = 0, 1, 3, 4
                self.first_col_name = columns_dict['first_col_name']
                self.second_col_name = columns_dict['second_col_name']
                self.fifth_col_name = columns_dict['fifth_col_name']
                is_utc_time = self.first_col_name.strip() == 'UTC Time'

            elif columns_dict['first_col_name'].strip() == 'scan':
                idx_time, idx_snr, idx_freq, idx_dopp = 1, 2, 4, 5
                self.first_col_name = columns_dict['first_col_name']
                self.second_col_name = columns_dict['second_col_name']
                self.fifth_col_name = columns_dict['fifth_col_name']
                is_utc_time = self.second_col_name.strip() == 'UTC Time'

            else:
                idx_time, idx_snr, idx_freq, idx_dopp = 1, 2, 4, 5
                mjd_day = self.Utilities.utc_to_mjd(self.observation_date)
                self.first_col_name = columns_dict['first_col_name']
                self.second_col_name = columns_dict['second_col_name']
                self.fifth_col_name = columns_dict['fifth_col_name']
                is_mjd_seconds = True

            # Read file line by line (skipping headers)
            with open(filename, 'r') as file:
                for i, line in enumerate(file):
                    if i < 4: continue  # Skip header lines
                    parts = line.strip().split()

                    utc_time.append(parts[idx_time])
                    signal_to_noise.append(float(parts[idx_snr]))
                    doppler_noise_hz.append(float(parts[idx_dopp]))
                    frequency_detection.append(float(parts[idx_freq]))

            # Convert UTC time
            if self.n_columns == 5 or columns_dict['first_col_name'].strip() == 'scan':
                try:
                    self.utc_datetime = [datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f") for t in utc_time]
                except ValueError:
                    self.utc_datetime = [self.Utilities.format_observation_time(self.observation_date, float(t)) for t in utc_time]
            else:
                utc_dates = [self.Utilities.mjd_utc_seconds_to_utc(mjd_day, float(t)) for t in utc_time]
                self.utc_datetime = [self.parse_datetime(t) for t in utc_dates]

            # Assign parsed lists to attributes
            self.signal_to_noise = signal_to_noise
            self.doppler_noise_hz = doppler_noise_hz
            self.frequency_detection = frequency_detection

            return {
                'receiving_station_name': self.receiving_station_name,
                'utc_datetime': self.utc_datetime,
                'Signal-to-Noise': self.signal_to_noise,
                'Doppler Noise [Hz]': self.doppler_noise_hz,
                'base_frequency': self.base_frequency,
                'Freq detection [Hz]': self.frequency_detection,
                'first_col_name': self.first_col_name,
                'second_col_name': self.second_col_name,
                'fifth_col_name': self.fifth_col_name,
                'utc_date': self.observation_date
            }

        ########################################################################################################################################
        ########################################################################################################################################

        def parse_datetime(self, t):
            """
            Description
            Utility function to parse different time formats, as various missions or experiments might utilize different formats. This function attempts to convert a given time string into a datetime object by trying several predefined formats until a match is found.

            Inputs
            t : str or any
                The time input to be parsed, which can be in various string formats representing date and time.

            Outputs
            datetime.datetime
                Returns a datetime object representing the parsed time. If none of the formats match, a ValueError is raised indicating that the time format is not recognized.
            """
            formats = [
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S"
            ]

            t = str(t)  # Ensure it's a string

            for fmt in formats:
                try:
                    return datetime.strptime(t, fmt)
                except ValueError:
                    continue  # Try the next format

            raise ValueError(f"Time format not recognized: {t}")

        def extract_folder_data(self,fdets_folder_path):

            """
            Description
            Extracts parameters from text files in a specified folder. The function scans the given directory for files that start with 'Fdets' and end with '.txt', and extracts parameters from each of these files using a separate method. The extracted parameters are collected into a list and returned.

            Inputs
            fdets_folder_path : str
                The path to the directory containing the text files to be processed. The function expects files that start with 'Fdets' and end with '.txt'.

            Outputs
            list
                Returns a list of extracted parameters from the processed text files. Each element in the list corresponds to the parameters extracted from a single file.
            """

            extracted_data_list =list()
            directory_path = fdets_folder_path
            for file in os.listdir(directory_path):
                if file.startswith('Fdets') and file.endswith('r2i.txt'):
                    file_path = os.path.join(directory_path, file)

                    extracted_parameters = self.extract_parameters(file_path)
                    extracted_data_list.append(extracted_parameters)


            return extracted_data_list

    class Utilities:

        def __init__(self):
            self.result = 0

            """
            Define dictionary for 
            1) a bunch of radio-tracking experiments experiments 
            2) all missions present in the Mas Said table file.
            ########################################################################################################################
            """

            self.experiments = {
                "gr035": {
                    "mission_name": "mex",
                    "vex_file_name": "gr035.vix",
                    "exper_description": "mars_express tracking test",
                    "exper_nominal_start": "2013y362d17h40m00s",
                    "exper_nominal_stop": "2013y363d18h30m00s"
                },

                "m0303": {
                    "mission_name": "mex",
                    "vex_file_name": "m0303.vix",
                    "exper_description": "mars express tracking test",
                    "exper_nominal_start": "2010y062d20h00m00s",
                    "exper_nominal_stop": "2010y062d21h59m00s"
                },

                "m0325": {
                    "mission_name": "mex",
                    "vex_file_name": "m0325.vix",
                    "exper_description": "mars express tracking test",
                    "exper_nominal_start": "2012y085d13h00m00s",
                    "exper_nominal_stop": "2012y085d13h59m00s"
                },

                "m0327": {
                    "mission_name": "mex",
                    "vex_file_name": "m0327.vix",
                    "exper_description": "mars express tracking test",
                    "exper_nominal_start": "2012y087d01h30m00s",
                    "exper_nominal_stop": "2012y087d02h49m00s"
                },

                "m0403": {
                    "mission_name": "mex",
                    "vex_file_name": "m0403.vix",
                    "exper_description": "mars express tracking test",
                    "exper_nominal_start": "2012y087d01h30m00s",
                    "exper_nominal_stop": "2012y087d02h49m00s"
                },

                "ed045a": {
                    "mission_name": "min",
                    "vex_file_name": "ed045a.vix",
                    "exper_description": "min tracking",
                    "exper_nominal_start": "2020y053d01h30m00s",
                    "exper_nominal_stop": "2020y053d03h00m00s"
                },
                "ed045c": {
                    "mission_name": "min",
                    "vex_file_name": "ed045c.vix",
                    "exper_description": "min tracking",
                    "exper_nominal_start": "2020y150d08h00m00s",
                    "exper_nominal_stop": "2020y150d09h30m00s"
                },
                "ed045d": {
                    "mission_name": "min",
                    "vex_file_name": "ed045d.vix",
                    "exper_description": "min tracking",
                    "exper_nominal_start": "2020y151d08h30m00s",
                    "exper_nominal_stop": "2020y151d10h00m00s"
                },
                "ed045e": {
                    "mission_name": "min",
                    "vex_file_name": "ed045e.vix",
                    "exper_description": "min tracking",
                    "exper_nominal_start": "2020y295d02h45m00s",
                    "exper_nominal_stop": "2020y295d04h15m00s"
                },
                "ed045f": {
                    "mission_name": "min",
                    "vex_file_name": "ed045f.vix",
                    "exper_description": "min tracking",
                    "exper_nominal_start": "2020y296d02h45m00s",
                    "exper_nominal_stop": "2020y296d04h15m00s"
                },

                "ec094a": {
                    "mission_name": "juice",
                    "vex_file_name": "ec094a.vix",
                    "exper_description": "JUICE tracking",
                    "exper_nominal_start": "2023y292d14h00m00s",
                    "exper_nominal_stop": "2023y292d16h00m00s"
                },

                "ec094b": {
                    "mission_name": "juice",
                    "vex_file_name": "ec094b.vix",
                    "exper_description": "JUICE tracking",
                    "exper_nominal_start": "2024y066d05h30m00s",
                    "exper_nominal_stop": "2024y066d07h30m00s"
                },

                "ec064": {
                    "mission_name": ["mro", 'mex', 'tgo'],
                    "vex_file_name": "ec064.vix",
                    "experiment_description": "MRO-TGO-MEX tracking",
                    "exper_nominal_start": "2018y155d04h00m00s",
                    "exper_nominal_stop": "2018y155d06h00m00s"
                },

                "v140314": {
                    "mission_name": "vex",
                    "vex_file_name": "v0314.vix",
                    "experiment_description": "VEX tracking",
                    "exper_nominal_start": "2014y073d08h30m00s",
                    "exper_nominal_stop": "2014y073d11h29m00s"
                },

            }

            self.spacecraft_data = {
                "aka": {
                    "mission_name": "akatsuki",
                    "frequency_MHz": 8410.926,
                    "antenna": "Ww",
                    "snr": 0,
                    "stochastic_noise": "0",
                    "updated": "00.00.2010"
                },
                "bco": {
                    "mission_name": "bepi colombo",
                    "frequency_MHz": 8420.293,
                    "antenna": "Cd",
                    "snr": 6317,
                    "stochastic_noise": "84 mHz",
                    "updated": "17.02.2021"
                },
                "tgo": {
                    "mission_name": "exomars - trace gas orbiter",
                    "frequency_MHz": 8410.710,
                    "antenna": "Ht",
                    "snr": 11500,
                    "stochastic_noise": "90 mHz",
                    "updated": "29.07.2017"
                },
                "exo": {
                    "mission_name": "exomars",
                    "frequency_MHz": 8424.592,
                    "antenna": "Xx",
                    "snr": 0,
                    "stochastic_noise": "0",
                    "updated": "00.00.0000"
                },
                "gai": {
                    "mission_name": "gaia",
                    "frequency_MHz": "xxxx.xxx",
                    "antenna": "Xx",
                    "snr": 0,
                    "stochastic_noise": "0",
                    "updated": "00.00.0000"
                },

                "her": {
                    "mission_name": "herschell",
                    "frequency_MHz": "xxxx.xxx",
                    "antenna": "Xx",
                    "snr": 0,
                    "stochastic_noise": "0",
                    "updated": "00.00.0000"
                },
                "huy": {
                    "mission_name": "huygens",
                    "frequency_MHz": "xxxx.xxx",
                    "antenna": "Xx",
                    "snr": 0,
                    "stochastic_noise": "0",
                    "updated": "00.00.0000"
                },
                "ika": {
                    "mission_name": "ikaros",
                    "frequency_MHz": 8431.296,
                    "antenna": "Ww",
                    "snr": 0,
                    "stochastic_noise": "0",
                    "updated": "00.00.2010"
                },
                "jno": {
                    "mission_name": "juno",
                    "frequency_MHz": 8403.523,
                    "antenna": "Ho",
                    "snr": 450,
                    "stochastic_noise": "150 mHz",
                    "updated": "07.09.2020"
                },
                "mex": {
                    "mission_name": "mars express",
                    "frequency_MHz": 8420.750,
                    "antenna": "Cd",
                    "snr": 10000,
                    "stochastic_noise": "30 mHz",
                    "updated": "22.02.2020"
                },
                "min": {
                    "mission_name": "mars insight",
                    "frequency_MHz": 8404.502,
                    "antenna": "T6",
                    "snr": 120,
                    "stochastic_noise": "300 mHz",
                    "updated": "25.07.2020"
                },
                "mod": {
                    "mission_name": "mars odyssey",
                    "frequency_MHz": 8407.250,
                    "antenna": "Cd",
                    "snr": 360,
                    "stochastic_noise": "830 mHz",
                    "updated": "22.02.2020"
                },
                "mom": {
                    "mission_name": "mars orbiter mission",
                    "frequency_MHz": "xxxx.xxx",
                    "antenna": "Xx",
                    "snr": 0,
                    "stochastic_noise": "0",
                    "updated": "00.00.0000"
                },
                "perseverance": {
                    "mission_name": "perseverance",
                    "frequency_MHz": 8435.550,
                    "antenna": "Cd",
                    "snr": 115000,
                    "stochastic_noise": "88 mHz",
                    "updated": "18.02.2021"
                },
                "m20": {
                    "mission_name": "mars2020",
                    "frequency_MHz": 8415.000,
                    "antenna": "??",
                    "snr": 0,
                    "stochastic_noise": "0",
                    "updated": "00.00.0000"
                },
                "mro": {
                    "mission_name": "mars reconnaissance orbiter",
                    "frequency_MHz": 8439.750,
                    "antenna": "Cd",
                    "snr": 30,
                    "stochastic_noise": "23.70 Hz",
                    "updated": "22.02.2020"
                },
                "mvn": {
                    "mission_name": "maven",
                    "frequency_MHz": 8446.235,
                    "antenna": "Xx",
                    "snr": 0,
                    "stochastic_noise": "0",
                    "updated": "00.00.0000"
                },
                "curiosity": {
                    "mission_name": "curiosity",
                    "frequency_MHz": 8402.777,
                    "antenna": None,
                    "snr": None,
                    "stochastic_noise": None,
                    "updated": None
                },
                "hope": {
                    "mission_name": "hope",
                    "frequency_MHz": 8401.419,
                    "antenna": None,
                    "snr": None,
                    "stochastic_noise": None,
                    "updated": None
                },
                "ras": {
                    "mission_name": "radio astron",
                    "frequency_MHz": 8399.700,
                    "antenna": "Wz",
                    "snr": 32000,
                    "stochastic_noise": "3 mHz",
                    "updated": "29.09.2012"
                },
                "ros": {
                    "mission_name": "rosetta",
                    "frequency_MHz": 8421.875,
                    "antenna": "Mh",
                    "snr": 898,
                    "stochastic_noise": "40 mHz",
                    "updated": "01.04.2016"
                },
                "sta": {
                    "mission_name": "stereo ab",
                    "frequency_MHz": 8446.230,
                    "antenna": "Wz",
                    "snr": 3000,
                    "stochastic_noise": "200 mHz",
                    "updated": "06.02.2012"
                },
                "tiw": {
                    "mission_name": "tianwen-1",
                    "frequency_MHz": 8429.938,
                    "antenna": "Cd",
                    "snr": None,
                    "stochastic_noise": "1.0",
                    "updated": "01.04.2021"
                },
                "uly": {
                    "mission_name": "ulysses",
                    "frequency_MHz": "xxxx.xxx",
                    "antenna": "Xx",
                    "snr": 0,
                    "stochastic_noise": "0",
                    "updated": "00.00.0000"
                },
                "vex": {
                    "mission_name": "venus express",
                    "frequency_MHz": 8418.100,
                    "antenna": "Ht",
                    "snr": 9500,
                    "stochastic_noise": "26 mHz",
                    "updated": "21.04.2011"
                },
                "juc": {
                    "mission_name": "juice",
                    "frequency_MHz": 8435.5, #mas said file says: 8435-8436, so i put 8435.5
                    "antenna": None,
                    "snr": None,
                    "stochastic_noise": None,
                    "updated": None
                }
            }


            self.antenna_diameters = {
                'Cd': 30,   # Ceduna
                'Hb': 12,   # Hobart 12m
                'Ho': 26,    # Hobart 26
                'Yg': 12,   # Yarragadee 12m
                'Ke': 12,   # Katherine 12m
                'Ww': 12,   # Warkworth
                'Ym': 32,   # Yamaguchi 32m
                'T6': 65,   # Tianma 65m
                'Km': 40,   # Kunming
                'Ku': 21,   # KVN Ulsan
                'Bd': 32,   # Badary
                'Ur': 25,   # Urumqi
                'Zc': 32,   # Zelenchukskaya
                'Hh': 26,   # Hartebeesthoek
                'Wz': 20,   # Wettzell
                'Sv': 32,   # Svetloe
                'Mc': 32,   # Medicina
                'Wb': 25,   # Westerbork (single dish)
                'On': 60,   # Onsala 60m
                'O6': 60,   # Onsala 60m
                'Ys': 40,   # Yebes 40m
                'Sc': 25,   # VLBA Saint Croix
                'Hn': 25,   # VLBA Hancock
                'Nl': 25,   # VLBA North Liberty
                'Fd': 25,   # VLBA Fort Davis
                'La': 25,   # VLBA Los Alamos
                'Kp': 25,   # VLBA Kitt Peak
                'Pt': 25,   # VLBA Pie Town
                'Br': 25,   # VLBA Brewster
                'Ov': 25,   # VLBA Owens Valley
                'Mk': 25,   # VLBA Mauna Kea
                'Ht': 15,   # HartRAO 15m
                'Mh': 14,   # Metsähovi
                'Ef': 100,  # Effelsberg
                'Tr': 32,   # Torun
                'Nt': 32,   # Noto
                'Ir': 32,   # Irbene
                'Ib': 32,   # Irbene
                'Mp': 15,   # Siding Spring
                'Wn': 13    # Wettzell 13m
            }

        ########################################################################################################################################
        ########################################################################################################################################

        def parse_experiment_time(self, time_str):
            """Convert time string like '2020y053d01h30m00s' to a timezone-aware UTC datetime object."""
            match = re.match(r"(\d{4})y(\d{3})d(\d{2})h(\d{2})m(\d{2})s", time_str)
            if not match:
                raise ValueError(f"Invalid time format: {time_str}. (Accepted format example: 2020y053d01h30m00s)")
            year, doy, hour, minute, second = map(int, match.groups())
            datetime_value = datetime.strptime(f"{year} {doy}", "%Y %j")

            return datetime_value.replace(tzinfo=timezone.utc)

        def find_experiment_from_yymmdd(self, yymmdd_str):
            """
            Check if a yymmdd string falls into any experiment interval.
            Return experiment name or ISO date string (YYYY-MM-DD).
            """
            try:
                # Parse 'yymmddHHMM' to datetime (assumes 2000+)
                datetime_value = datetime.strptime(yymmdd_str, "%y%m%d").replace(tzinfo=timezone.utc)
            except ValueError:
                raise ValueError(f"Invalid yymmddHHMM format: '{yymmdd_str}'. Expected format: 'yymmddHHMM' (e.g., '2407161305')")

            for exp_name, exp_data in self.experiments.items():
                start = self.parse_experiment_time(exp_data["exper_nominal_start"])
                stop = self.parse_experiment_time(exp_data["exper_nominal_stop"])
                if start <= datetime_value <= stop:
                    return exp_name

            return yymmdd_str




        def datetime_to_yymmdd(self,datetime_value):
            """Convert a datetime object to 'YYMMDD' format."""
            if datetime_value.tzinfo is None:
                datetime_value = datetime_value.replace(tzinfo=timezone.utc)
            else:
                datetime_value = datetime_value.astimezone(timezone.utc)

            return datetime_value.strftime("%y%m%d")
        def list_yymm(self, start_date, end_date):
            """Return a dictionary {yymm: [list of yymmdd]} for each day between start_date and end_date inclusive."""
            if start_date > end_date:
                raise ValueError("Start date must be before end date.")

            current = start_date
            yymm_dict = {}

            while current <= end_date:
                yymmdd = self.datetime_to_yymmdd(current)  # e.g., '210222'
                yymm = yymmdd[:4]  # extract 'yymm'
                yymm_dict.setdefault(yymm, []).append(yymmdd)
                current += timedelta(days=1)

            return yymm_dict

        def mjd_to_utc(self,mjd):
            """
            Description
            Converts a Modified Julian Date (MJD) to a UTC date. The function uses a reference date of 17 November 1858, which is the starting point for MJD calculations. It adds the number of days represented by the MJD to this reference date to compute the corresponding UTC date.

            Inputs
            mjd : float or int
                The Modified Julian Date to be converted into UTC. This represents the number of days since the MJD reference date.

            Outputs
            datetime.date
                Returns the corresponding UTC date as a datetime.date object.
            """
            # Define the starting reference date for MJD, which is 17 November 1858
            mjd_reference = datetime(1858, 11, 17, 0, 0, 0)

            # Calculate the UTC date by adding the number of days in the MJD to the reference date
            utc_date = mjd_reference + timedelta(days=mjd)
            return utc_date.date

        ########################################################################################################################################
        ########################################################################################################################################

        def utc_to_mjd(self, utc_date_str):
            """
            Description
            Converts a UTC date string in the format YYYY.MM.DD to a Modified Julian Date (MJD). The function calculates the difference in days between the given UTC date and the reference date of 17 November 1858, which marks the start of MJD.

            Inputs
            utc_date_str : str
                The UTC date as a string in the format YYYY.MM.DD to be converted into MJD.

            Outputs
            float
                Returns the corresponding Modified Julian Date as a floating-point number, which includes both the integer days and the fractional part representing the time of day.
            """
            # Convert the UTC date string (in YYYY.MM.DD format) to a datetime object
            utc_date = datetime.strptime(utc_date_str, "%Y.%m.%d")

            # MJD starts at midnight on November 17, 1858 (Julian Date 2400000.5)
            mjd_start = datetime(1858, 11, 17)

            # Calculate the difference in days between the UTC date and the MJD start
            delta = utc_date - mjd_start

            # Return the MJD date
            return delta.days + (delta.seconds / 86400.0)


        def mission_name_to_horizons_target(self, mission_name):
            """
            Description
            Maps a mission name to its corresponding Horizons target identifier. The function uses a predefined dictionary that associates mission names with their respective target values. It performs a case-insensitive lookup for the provided mission name and returns the associated target identifier.

            Inputs
            mission_name : str
                The name of the mission (e.g., "mex", "juice", "min") for which the Horizons target identifier is to be retrieved.

            Outputs
            str or None
                Returns the corresponding Horizons target identifier as a string. If the mission name is not found in the mapping, it returns None.
            """

            self.horizons_targets = {
                "mex": {
                    "target": "-41",
                },

                "jui":{
                    "target": "2023-053A",
                },

                "min":{
                    "target": "2018-042A"
                },

                "mro":{
                    "target": "2005-029A"
                },

                "vex":{
                    "target": "2005-045A"
                }

            }
            # Convert input to uppercase for case-insensitive matching
            mission_name = mission_name.lower()
            return self.horizons_targets.get(mission_name, {}).get("target", None)
        ########################################################################################################################################
        ########################################################################################################################################

        def mjd_utc_seconds_to_utc(self,mjd, utc_seconds):

            """
            Description
            Converts a Modified Julian Date (MJD) and a number of UTC seconds into a full UTC date and time. The function first converts the MJD to a Julian Date (JD) and then derives the corresponding Gregorian date. It accounts for the fractional day represented by the JD and adds any additional seconds provided to obtain the final UTC date and time.

            Inputs
            mjd : float or int
                The Modified Julian Date to be converted into UTC.

            utc_seconds : int
                The number of seconds to be added to the UTC date and time derived from the MJD.

            Outputs
            datetime.datetime
                Returns the complete UTC date and time as a datetime.datetime object.
            """

            # Convert MJD to JD
            jd = mjd + 2400000.5

            # JD to Gregorian date (subtract 2400000.5 to get the fractional day)
            jd_days = int(jd)  # integer part is the day
            jd_fraction = jd - jd_days  # fractional part is the time of the day

            # Reference date for Julian Day 0 in proleptic Gregorian calendar
            jd_start = datetime(1858, 11, 17, 0, 0, 0)

            # Convert JD days to datetime
            utc_date = jd_start + timedelta(days=jd_days - 2400000.5)

            # Handle fractional part of the day (converts to time in hours, minutes, seconds)
            utc_time = utc_date + timedelta(days=jd_fraction)

            # Add the UTC seconds from the second column
            final_utc = utc_time + timedelta(seconds=utc_seconds)

            # Return the full UTC date and time
            return final_utc

        def combine_plots(self, image_paths, output_dir, output_file_name, direction='vertical'):
            """
            Description
            Combines multiple images into a single image by stacking them either horizontally or vertically. The function resizes the images to ensure uniformity in dimensions before combining them, and it saves the resulting image to the specified directory.

            Inputs
            image_paths : list
                A list of file paths to the images that need to be combined. At least two image paths are required.

            output_dir : str
                The directory where the combined image will be saved.

            output_file_name : str
                The name of the output file for the combined image.

            direction : str
                Determines how the images are stacked. Acceptable values are 'horizontal' or 'vertical'.

            Outputs
            None
                The function saves the combined image to the specified output directory and does not return any value.
            """
            if not image_paths or len(image_paths) < 2:
                raise ValueError("At least two image paths are required.")

            os.makedirs(output_dir, exist_ok=True)

            # Open all images
            images = [Image.open(img_path) for img_path in image_paths]

            # Determine max width and height for resizing
            max_width = max(img.width for img in images)
            max_height = max(img.height for img in images)

            # Resize images to match dimensions
            if direction == 'horizontal':
                images = [img.resize((img.width, max_height)) for img in images]
                total_width = sum(img.width for img in images)
                total_height = max_height
                new_img = Image.new('RGB', (total_width, total_height))

                # Paste images side by side
                x_offset = 0
                for img in images:
                    new_img.paste(img, (x_offset, 0))
                    x_offset += img.width

            elif direction == 'vertical':
                images = [img.resize((max_width, img.height)) for img in images]
                total_width = max_width
                total_height = sum(img.height for img in images)
                new_img = Image.new('RGB', (total_width, total_height))

                # Paste images on top of each other
                y_offset = 0
                for img in images:
                    new_img.paste(img, (0, y_offset))
                    y_offset += img.height

            else:
                raise ValueError("Direction must be either 'horizontal' or 'vertical'.")

            # Save the combined image
            output_path = os.path.join(output_dir, output_file_name)
            new_img.save(output_path)
            print(f"Combined image saved to: {output_path}")

        ################################################################################################################################################################################################################################################################################

        def format_observation_time(self, observation_date, time_in_seconds):
            """
            Description
            Formats an observation date and a time in seconds into a complete datetime object. The function converts the observation date from a string format to a datetime object, splits the time in seconds into its integer and fractional components, and combines them to produce a full datetime representation.

            Inputs
            observation_date : str
                The observation date as a string in the format "%Y.%m.%d" to be converted into a datetime object.

            time_in_seconds : float
                The time of the observation in seconds, which may include fractional seconds.

            Outputs
            datetime.datetime
                Returns a datetime object representing the combined observation date and time.
            """
            # Convert self.observation_date from "%Y.%m.%d" to a datetime object
            observation_date = datetime.strptime(observation_date, "%Y.%m.%d")

            # Split seconds into integer seconds and fractional microseconds
            int_seconds, frac_seconds = divmod(time_in_seconds, 1)

            # Convert to hours, minutes, seconds
            hours, remainder = divmod(int(int_seconds), 3600)
            minutes, seconds = divmod(remainder, 60)

            # Convert fractional part to microseconds
            microseconds = round(frac_seconds * 1_000_000)

            # Create a timedelta object for the time of day
            time_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds)

            # Combine the observation date and the time of day
            full_datetime = observation_date + time_delta
            # Return the formatted date-time in the desired format
            return datetime.strptime(str(full_datetime), "%Y-%m-%d %H:%M:%S")

        def extract_ground_station_dict(self, file_path):
            """
            Description
            Extracts ground station configuration data from a specified file and parses it into a nested dictionary. The function identifies a section in the file that contains ground station information, retrieves key-value pairs associated with each station, and organizes them into a dictionary format.

            Inputs
            file_path : str
                The path to the input file containing ground station data.

            Outputs
            dict or None
                Returns a nested dictionary mapping ground station names to their corresponding key-value attributes. Returns None if no ground station section is found or if the file does not exist.
            """
            dictionary = {}
            ground_station_dict = {}
            inside_section = False
            current_station = None

            try:
                with open(file_path, 'r') as file:
                    for line in file:
                        # Start of a section
                        if line.strip().startswith('$SITE'):
                            inside_section = True
                        # Parse key-value pairs if inside the ground station section
                        if inside_section:
                            # Skip the def line itself
                            if line.strip().startswith('def'):
                                match = re.search(r'def\s+(\w+);', line.strip())
                                if match:
                                    current_station = match.group(1)
                                    ground_station_dict[current_station] = {}

                            # End of the section
                            if line.strip().startswith('$ANTENNA'):
                                current_station = None
                                break

                            # Parse key-value pairs (assumes "key = value" format)
                            if current_station and '=' in line:
                                # Use a regex to find all key-value pairs in the line
                                pairs = re.findall(r'(\w+)\s*=\s*([^\s;]+)', line)
                                for key, value in pairs:
                                    if key != 'site_name' and key != 'site_position' and key != 'horizon_map_az' and  key != 'horizon_map_el':
                                        ground_station_dict[current_station][key] = value
                # Handle line with elev, long, and lat

                # Return the dictionary or None if the section was not found
                return ground_station_dict if ground_station_dict else None

            except FileNotFoundError:
                print(f"File not found: {file_path}")
                return None

        def site_to_ID(self, site):
            """
            Maps a ground station name to its corresponding site ID.

            Args:
                site (str): The name of the ground station (case-insensitive).

            Returns:
                str: The site ID if the site name is found, or None otherwise.
            """
            site_to_id_mapping = {
                'CEDUNA': 'Cd',
                'HOBART12': 'Hb',
                'HOBART26': 'Ho',
                'YARRA12M': 'Yg',
                'KATH12M': 'Ke',
                'WARK': 'Ww',
                'YAMAGU32': 'Ym',
                'TIANMA65': 'T6',
                'KUNMING': 'Km',
                'KVNUS': 'Ku',
                'BADARY': 'Bd',
                'URUMQI': 'Ur',
                'ZELENCHK': 'Zc',
                'HARTRAO': 'Hh',
                'WETTZELL': 'Wz',
                'SVETLOE': 'Sv',
                'MEDICINA': 'Mc',
                'WSTRBORK': 'Wb',
                'ONSALA60': 'On',
                'YEBES40M': 'Ys',
                'VLBA_SC': 'Sc',
                'VLBA_HN': 'Hn',
                'VLBA_NL': 'Nl',
                'VLBA_FD': 'Fd',
                'VLBA_LA': 'La',
                'VLBA_KP': 'Kp',
                'PIETOWN': 'Pt',
                'VLBA_BR': 'Br',
                'VLBA_OV': 'Ov',
                'VLBA_MK': 'Mk',
                'HART15M': 'Ht',
                'METSAHOV': 'Mh',
                'EFLSBERG': 'Ef',
                'TORUN': 'Tr',
                'NOTO': 'Nt',
                'IRBENE': 'Ir',
                'SIDINGSPRING': 'Mp',
                'WETTZ13N': 'Wn'
            }

            # Convert input to uppercase for case-insensitive matching
            site = site.upper()
            return site_to_id_mapping.get(site, None)


        def ID_to_site(self, site_ID):

            """
            Maps a site ID to its corresponding ground station name.

            Args:
                site_ID (str): The site ID.

            Returns:
                str: The ground station name if the site ID is found, or None otherwise.
            """
            id_to_site_mapping = {
                'Cd': 'CEDUNA',
                'Hb': 'HOBART12',
                'Ho': 'HOBART26',
                'Yg': 'YARRA12M',
                'Ke': 'KATH12M',
                'Ww': 'WARK',
                'Ym': 'YAMAGU32',
                'T6': 'TIANMA65',
                'Km': 'KUNMING',
                'Ku': 'KVNUS',
                'Bd': 'BADARY',
                'Ur': 'URUMQI',
                'Zc': 'ZELENCHK',
                'Hh': 'HARTRAO',
                'Ht': 'HART15M',
                'Mh': 'METSAHOV',
                'Wz': 'WETTZELL',
                'Sv': 'SVETLOE',
                'Mc': 'MEDICINA',
                'Wb': 'WSTRBORK',
                'On': 'ONSALA60',
                'O6': 'ONSALA60',
                'Ys': 'YEBES40M',
                'Sc': 'SC-VLBA',
                'Hn': 'HN-VLBA',
                'Nl': 'NL-VLBA',
                'Fd': 'FD-VLBA',
                'La': 'LA-VLBA',
                'Kp': 'KP-VLBA',
                'Br': 'BR-VLBA',
                'Ov': 'OV-VLBA',
                'Mk': 'MK-VLBA',
                'Pt': 'PIETOWN',
                'Tr': 'TORUN',
                'Ef': 'EFLSBERG',
                'Nt': 'NOTO',
                'Ir': 'IRBENE',
                'Ib': 'IRBENE',
                'Mp': 'SIDINGSPRING',
                'Wn': 'WETTZ13N'
            }

            # Return the corresponding site name or None if the site_ID is not found
            return id_to_site_mapping.get(site_ID, None)


        def site_to_geodetic_position(self, station_name):

            """
            Description
            Maps a ground station name to its corresponding geodetic position.
            """
            # Order: Altitude (m), Latitude, Longitude (deg)
            site_to_geodetic_position_mapping = {
                "DSS-12": [
                    962.5695389993489,
                    35.29993783888286,
                    -116.80548606967588
                ],
                "DSS-13": [
                    1070.8438292415813,
                    35.24716449544569,
                    -116.79445811495766
                ],
                "DSS-14": [
                    1001.7899944689125,
                    35.42590110865716,
                    -116.88953732307806
                ],
                "DSS-15": [
                    973.6106805289164,
                    35.421853518300175,
                    -116.8871942195082
                ],
                "DSS-24": [
                    951.9105238793418,
                    35.33989303577655,
                    -116.87479350278461
                ],
                "DSS-25": [
                    960.033178656362,
                    35.337612206939944,
                    -116.87536231939559
                ],
                "DSS-26": [
                    969.0920485360548,
                    35.33568942212718,
                    -116.87301553443193
                ],
                "DSS-27": [
                    1052.8677822519094,
                    35.2382720166531,
                    -116.77664957022726
                ],
                "DSS-34": [
                    692.3553875628859,
                    -35.39847801214653,
                    148.98196417218125
                ],
                "DSS-35": [
                    695.2315960647538,
                    -35.39579468824336,
                    148.98145551930915
                ],
                "DSS-36": [
                    685.8377869874239,
                    -35.39510093643416,
                    148.97854398520033
                ],
                "DSS-42": [
                    674.9855659790337,
                    -35.40067820768434,
                    148.98126474037895
                ],
                "DSS-43": [
                    689.2020253008232,
                    -35.40242341041149,
                    148.98126706261021
                ],
                "DSS-45": [
                    674.6815513102338,
                    -35.39845685896725,
                    148.9776853860103
                ],
                "DSS-54": [
                    837.4869195409119,
                    40.42562130635732,
                    -4.254097265281252
                ],
                "DSS-55": [
                    819.4963778760284,
                    40.42429553121944,
                    -4.252633737155015
                ],
                "DSS-61": [
                    840.8882981548086,
                    40.4287370277765,
                    -4.249025124234493
                ],
                "DSS-63": [
                    865.2525680121034,
                    40.43120937830096,
                    -4.24800897985988
                ],
                "DSS-65": [
                    834.2896991213784,
                    40.42720599131806,
                    -4.250699325792482
                ],
                "AIRA": [
                    322.7899282677099,
                    31.82379448142901,
                    130.59986239311604
                ],
                "ALGOPARK": [
                    224.4077238906175,
                    45.95549914751449,
                    -78.07272499448413
                ],
                "ARECIBO": [
                    451.5799764152616,
                    18.344174013774253,
                    -66.75269824813448
                ],
                "ASKAP-29": [
                    355.8847079873085,
                    -26.690212625411007,
                    116.63715419207239
                ],
                "ATCA-104": [
                    252.4250863082707,
                    -30.312882178822672,
                    149.564760523679
                ],
                "ATCAPN5": [
                    252.4250863082707,
                    -30.312882178822672,
                    149.564760523679
                ],
                "AUSTINTX": [
                    190.03529037348926,
                    30.33932931456103,
                    -97.69574835838061
                ],
                "AZORES": [
                    58.97855842765421,
                    37.740922738175016,
                    -25.657584348940116
                ],
                "BADARY": [
                    821.986854291521,
                    51.77026195233466,
                    102.23391593941678
                ],
                "BEIJING": [
                    178.15021308884025,
                    40.557965697176776,
                    116.97595705121783
                ],
                "BERMUDA": [
                    -30.30378787498921,
                    32.361231018515824,
                    -64.66947202319365
                ],
                "BLKBUTTE": [
                    489.79232834186405,
                    33.663748341398055,
                    -115.719812571692
                ],
                "BLOOMIND": [
                    218.34525088313967,
                    39.17934714203324,
                    -86.49841588181594
                ],
                "BR-VLBA": [
                    250.912667918019,
                    48.13122493409872,
                    -119.68327744903162
                ],
                "BREST": [
                    104.89091716147959,
                    48.40786757068313,
                    -4.5038286094522055
                ],
                "CAMBRIDG": [
                    83.575824146159,
                    52.16695920302052,
                    0.03720678606462181
                ],
                "CARNUSTY": [
                    58.26747829280794,
                    56.47849811152383,
                    -2.7829911897985715
                ],
                "CARROLGA": [
                    299.9244269458577,
                    33.57255259227605,
                    -85.1095844041008
                ],
                "CEBREROS": [
                    794.501218774356,
                    40.4526893608711,
                    -4.367549975981012
                ],
                "CEDUNA": [
                    165.01153324637562,
                    -31.86769180377699,
                    133.80983107976817
                ],
                "CHICHI10": [
                    107.9205163391307,
                    27.067221973441715,
                    142.1950022747953
                ],
                "CHLBOLTN": [
                    147.2033326458186,
                    51.14500097545905,
                    -1.4384244730981337
                ],
                "CRIMEA": [
                    50.59553275257349,
                    44.397551249159505,
                    33.979571271936976
                ],
                "CTVASBAY": [
                    48.34402468241751,
                    45.40000657724998,
                    -75.91885566237158
                ],
                "CTVASTJ": [
                    155.4606042727828,
                    47.59519418390475,
                    -52.6792308300527
                ],
                "DAITO": [
                    53.125451686792076,
                    25.82893544919693,
                    131.2333932634972
                ],
                "DEADMANL": [
                    834.3085692701861,
                    34.254996622429815,
                    -116.27887791671208
                ],
                "DSS13": [
                    1070.824434939772,
                    35.24716480221695,
                    -116.79445771280155
                ],
                "DSS14": [
                    1001.7358005270362,
                    35.42590093072791,
                    -116.88953742537579
                ],
                "DSS15": [
                    973.6049485504627,
                    35.42185341680863,
                    -116.88719370905024
                ],
                "DSS34": [
                    692.4255754761398,
                    -35.39847895840477,
                    148.98196441980883
                ],
                "DSS35": [
                    695.2972646206617,
                    -35.39579692380624,
                    148.98145506571308
                ],
                "DSS43": [
                    689.2239404339343,
                    -35.40242498648992,
                    148.9812663579985
                ],
                "DSS45": [
                    674.7713237330317,
                    -35.398460012645714,
                    148.97768415794965
                ],
                "DSS63": [
                    865.2641760287806,
                    40.431208830896416,
                    -4.2480106114313685
                ],
                "DSS65": [
                    834.2452908596024,
                    40.42718412073639,
                    -4.251419725523832
                ],
                "DSS65A": [
                    834.2433073054999,
                    40.427205321115174,
                    -4.250700444732523
                ],
                "DWINGELO": [
                    43.40945273544639,
                    52.81201912499159,
                    6.396169006114649
                ],
                "EFLSBERG": [
                    417.1102016437799,
                    50.524832599574836,
                    6.883610628100581
                ],
                "ELY": [
                    1886.6062321383506,
                    39.29317840343969,
                    -114.84296319685392
                ],
                "EVPATORI": [
                    86.47950495686382,
                    45.18883801938122,
                    33.187417143878136
                ],
                "FD-VLBA": [
                    1606.8373390994966,
                    30.635029807824722,
                    -103.94482051076417
                ],
                "FLAGSTAF": [
                    2145.0759592205286,
                    35.21469651639039,
                    -111.63474483296967
                ],
                "FORTLEZA": [
                    23.47385509032756,
                    -3.8778590743323154,
                    -38.42585847478811
                ],
                "FORTORDS": [
                    250.3521402841434,
                    36.58937197560085,
                    -121.77214845546278
                ],
                "GBT-VLBA": [
                    824.0522695248947,
                    38.43312928094159,
                    -79.83983852631299
                ],
                "GBTS": [
                    812.9338396182284,
                    38.437823743364554,
                    -79.83577879948875
                ],
                "GGAO7108": [
                    14.13922628480941,
                    39.02192602303173,
                    -76.82654182447114
                ],
                "GIFU11": [
                    60.68086810875684,
                    35.46759172991373,
                    136.73706572986507
                ],
                "GIFU3": [
                    54.19150956347585,
                    35.46234558603998,
                    136.7395221461359
                ],
                "GILCREEK": [
                    332.4898388739675,
                    64.9784121995546,
                    -147.49751404321103
                ],
                "GOLDVENU": [
                    1063.3472414538264,
                    35.24769949549889,
                    -116.79488652028233
                ],
                "GORF7102": [
                    18.357117403298616,
                    39.02066499245327,
                    -76.82807267294854
                ],
                "GRASSE": [
                    1319.0469710258767,
                    43.754627452075795,
                    6.9206977655435
                ],
                "HALEAKAL": [
                    3068.2135271830484,
                    20.707612932611593,
                    -156.2560500986755
                ],
                "HART15M": [
                    1408.2375994930044,
                    -25.889735436566045,
                    27.68426900496594
                ],
                "HARTRAO": [
                    1416.1130826706067,
                    -25.889751910145527,
                    27.685392671594112
                ],
                "HATCREEK": [
                    1009.6765594538301,
                    40.817339474784376,
                    -121.47051567866028
                ],
                "HAYSTACK": [
                    117.09119327832013,
                    42.62329692317912,
                    -71.48816065015242
                ],
                "HITA32": [
                    120.36674206797034,
                    36.697457030900004,
                    140.69211160907258
                ],
                "HN-VLBA": [
                    295.97549921181053,
                    42.93360985466745,
                    -71.98657970108393
                ],
                "HOBART12": [
                    41.38966545742005,
                    -42.80558099167827,
                    147.43813722942386
                ],
                "HOBART26": [
                    65.51137382350862,
                    -42.80358617071475,
                    147.44051781909639
                ],
                "HOFN": [
                    78.67822477035224,
                    64.26774542009377,
                    -15.197442759397367
                ],
                "HOHENFRG": [
                    150.22541592642665,
                    53.050653406082986,
                    10.476451290372848
                ],
                "HOHNBERG": [
                    1006.1354262763634,
                    47.8009585558277,
                    11.017885583967486
                ],

                "IRBENE": [
                    87.3,
                    57.553333,
                    21.854722,
                ],
                "JODRELL1": [
                    179.82088891789317,
                    53.236539551974786,
                    -2.3085764986557598
                ],
                "JODRELL2": [
                    144.26745982840657,
                    53.2339636856082,
                    -2.3039041766154273
                ],
                "KAINAN": [
                    60.84208058658987,
                    34.15604020101493,
                    135.22994678674698
                ],
                "KALYAZIN": [
                    178.48927922174335,
                    57.22301642609043,
                    37.90032038049597
                ],
                "KANOZAN": [
                    373.66521003004164,
                    35.255055204491185,
                    139.95296466143782
                ],
                "KARLBURG": [
                    77.51430731080472,
                    53.98358530991791,
                    13.609252177241993
                ],
                "KASHIM11": [
                    62.84763630013913,
                    35.9555907682778,
                    140.65747229450898
                ],
                "KASHIM34": [
                    78.811497814022,
                    35.95591081024763,
                    140.66008971367023
                ],
                "KASHIMA": [
                    80.50374839734286,
                    35.954111803535056,
                    140.66273597398805
                ],
                "KATH12M": [
                    189.6656104652211,
                    -14.375462302775436,
                    132.1523738212483
                ],
                "KAUAI": [
                    1168.7225360590965,
                    22.126299802191333,
                    -159.66516442542786
                ],
                "KIRSBERG": [
                    261.63720230478793,
                    51.214244425754124,
                    14.28622716443348
                ],
                "KODIAK": [
                    31.663452265784144,
                    57.739986546593755,
                    -152.497224133247
                ],
                "KOGANEI": [
                    125.78205455373973,
                    35.71055739230251,
                    139.48808248745783
                ],
                "KOGANEI3": [
                    114.86569821089506,
                    35.70696002218587,
                    139.4875646162526
                ],
                "KOKEE": [
                    1177.000900151208,
                    22.126638118673807,
                    -159.66509770857493
                ],
                "KP-VLBA": [
                    1902.4028197508305,
                    31.956304539462703,
                    -111.612422337331
                ],
                "KUNMING": [
                    1975.1651182444766,
                    25.027332457777696,
                    102.79593230681913
                ],
                "KVNTN": [
                    441.88323371484876,
                    33.28902671599252,
                    126.4595914683317
                ],
                "KVNUS": [
                    161.56814685463905,
                    35.54560340013036,
                    129.24975995769512
                ],
                "KVNYS": [
                    122.00538948364556,
                    37.56519328935678,
                    126.94098884438489
                ],
                "KWAJAL26": [
                    57.79569187294692,
                    9.398755232714707,
                    167.48214633618267
                ],
                "LA-VLBA": [
                    1962.847928667441,
                    35.77512357204415,
                    -106.24559567720938
                ],
                "LEONRDOK": [
                    226.67807468678802,
                    35.90901651050721,
                    -95.79507175469452
                ],
                "MALARGUE": [
                    1572.1243692571297,
                    -35.77597213866333,
                    -69.39817793974382
                ],
                "MAMMOTHL": [
                    2310.8532696552575,
                    37.64164788066374,
                    -118.94517705374228
                ],
                "MARCUS": [
                    42.13955028541386,
                    24.289941143844135,
                    153.98420894697068
                ],
                "MARPOINT": [
                    -13.054128091782331,
                    38.37424751969811,
                    -77.23057657016098
                ],
                "MATERA": [
                    543.7770417444408,
                    40.649524069961224,
                    16.704015855396747
                ],
                "MEDICINA": [
                    67.59237221442163,
                    44.52049265220994,
                    11.646932866178119
                ],
                "METSAHOV": [
                    80.23954554554075,
                    60.217808701897354,
                    24.393108654495087
                ],
                "METSHOVI": [
                    60.11354560870677,
                    60.24196519343117,
                    24.384170143149845
                ],
                "MIAMI20": [
                    -11.364107962697744,
                    25.613749363848903,
                    -80.38474115990294
                ],
                "MILESMON": [
                    703.9101224672049,
                    46.396543109790414,
                    -105.8608387002156
                ],
                "MIURA": [
                    105.55774705857038,
                    35.20742201739304,
                    139.65037933070127
                ],
                "MIYAZAKI": [
                    107.53062541875988,
                    32.09077559635961,
                    131.4827926943434
                ],
                "MIYUN50": [
                    178.15021308884025,
                    40.557965697176776,
                    116.97595705121783
                ],
                "MIZNAO10": [
                    111.3652792274952,
                    39.13337270485999,
                    141.13234805321034
                ],
                "MIZUSGSI": [
                    173.5861099595204,
                    39.11041370634324,
                    141.203999276218
                ],
                "MK-VLBA": [
                    3763.451307467185,
                    19.80138093397988,
                    -155.45550252232775
                ],
                "MOJAVE12": [
                    910.6361800450832,
                    35.3316311672993,
                    -116.8876160772216
                ],
                "MOPRA": [
                    867.7282169284299,
                    -31.267809984267785,
                    149.0996439773787
                ],
                "MV2ONSLA": [
                    43.77478794567287,
                    57.39548356543683,
                    11.925395107630369
                ],
                "NL-VLBA": [
                    222.68350651860237,
                    41.77142486015605,
                    -91.57413733642368
                ],
                "NOME": [
                    332.0672449534759,
                    64.56270267056648,
                    -165.37122750622217
                ],
                "NOTO": [
                    143.6396358711645,
                    36.87604998769638,
                    14.989047577454935
                ],
                "NOTOX": [
                    143.6396358711645,
                    36.87604998769638,
                    14.989047577454935
                ],
                "NRAO20": [
                    807.0226719910279,
                    38.436851762669534,
                    -79.82551867889497
                ],
                "NWNORCIA": [
                    252.649596250616,
                    -31.048224127142696,
                    116.19150138375912
                ],
                "NYALES20": [
                    87.68489900138229,
                    78.92911031677971,
                    11.869691690599245
                ],
                "OCOTILLO": [
                    -36.34034904837608,
                    32.790101875589755,
                    -115.79618766344318
                ],
                "OHIGGINS": [
                    40.277110155671835,
                    -63.32112588649399,
                    -57.90082077214458
                ],
                "ONSALA60": [
                    59.688845879398286,
                    57.395836460308836,
                    11.926354380713045
                ],
                "ONSALA85": [
                    58.82776216883212,
                    57.39307028480311,
                    11.91776987941928
                ],
                "OV-VLBA": [
                    1196.726418708451,
                    37.23165066645818,
                    -118.27705382388386
                ],
                "PARKES": [
                    415.21882475726306,
                    -32.99840134619768,
                    148.26351410773268
                ],
                "PBLOSSOM": [
                    891.2681425027549,
                    34.51213144084759,
                    -117.9223922937184
                ],
                "PENTICTN": [
                    530.0382433906198,
                    49.32258875368787,
                    -119.61989413964633
                ],
                "PIETOWN": [
                    2365.083171754144,
                    34.30101764199717,
                    -108.11918935161412
                ],
                "PINFLATS": [
                    1235.9499097391963,
                    33.60924886467916,
                    -116.45880412804324
                ],
                "PLATTVIL": [
                    1501.7839340204373,
                    40.18279329254313,
                    -104.72634526431163
                ],
                "PRESIDIO": [
                    -29.030886554159224,
                    37.80530375516001,
                    -122.4550708207724
                ],
                "PUSHCHIN": [
                    240.54800010751933,
                    54.82060995707919,
                    37.628262209141575
                ],
                "PVERDES": [
                    69.74025499261916,
                    33.743763065652885,
                    -118.40355275005633
                ],
                "QUINCY": [
                    1106.168473736383,
                    39.97455459882284,
                    -120.94442592885365
                ],
                "RICHMOND": [
                    -13.214947178959846,
                    25.61375743352369,
                    -80.38471025743208
                ],
                "ROBLED32": [
                    840.9517508698627,
                    40.42873841300947,
                    -4.249024456368038
                ],
                "SAGARA": [
                    151.82213809899986,
                    34.67755696545865,
                    138.18287286300747
                ],
                "SANPAULA": [
                    185.29336976259947,
                    34.38786995417982,
                    -118.99879336565314
                ],
                "SANTIA12": [
                    730.6935336040333,
                    -33.15147367398958,
                    -70.66831116124747
                ],
                "SARDINIA": [
                    672.0291448831558,
                    39.49306975283907,
                    9.245147888400917
                ],
                "SC-VLBA": [
                    -14.596254682168365,
                    17.75658119675969,
                    -64.5836330239391
                ],
                "SEATTLE1": [
                    -16.012348604388535,
                    47.685686507346006,
                    -122.24912589001814
                ],
                "SESHAN25": [
                    29.833942756988108,
                    31.099162898653532,
                    121.1996588645335
                ],
                "SEST": [
                    2416.170265143737,
                    -29.2634518668311,
                    -70.73234808013247
                ],
                "SHANGHAI": [
                    18.18423420470208,
                    31.190180334092968,
                    121.4295508654589
                ],
                "SIDINGSPRING": [
                    1165,
                    -31.2733,
                    149.0644
                ],
                "SINTOTU": [
                    118.93173525296152,
                    43.52877212936207,
                    141.8445865439696
                ],
                "SINTOTU3": [
                    119.21594730298966,
                    43.528772492568,
                    141.84458642022955
                ],
                "SNDPOINT": [
                    93.15843232348561,
                    55.35232153669467,
                    -160.47548934343214
                ],
                "SOURDOGH": [
                    748.3605419220403,
                    62.66390884442377,
                    -145.48371144908754
                ],
                "SUWON": [
                    81.85622697230428,
                    37.27542633212476,
                    127.05421165147965
                ],
                "SVETLOE": [
                    86.45783763099462,
                    60.532344424062565,
                    29.781937241979143
                ],
                "SYOWA": [
                    51.37394125107676,
                    -69.00632467782064,
                    39.586286205683734
                ],
                "TATEYAMA": [
                    126.56527841836214,
                    34.937016273675845,
                    139.8486332632874
                ],
                "TIANMA65": [
                    49.60827589593828,
                    31.092103680572006,
                    121.13600362033283
                ],
                "TIDBIN64": [
                    689.1922933636233,
                    -35.40242660444019,
                    148.9812660640282
                ],
                "TIGOCONC": [
                    171.36068237852305,
                    -36.842718425920225,
                    -73.02514850304121
                ],
                "TIGOWTZL": [
                    659.2978802733123,
                    49.144495765256146,
                    12.877615573913996
                ],
                "TITIJIMA": [
                    99.39067386556417,
                    27.09799541945656,
                    142.19452035072234
                ],
                "TOMAKO11": [
                    90.26197433564812,
                    42.67377073040515,
                    141.59695038491387
                ],
                "TORUN": [
                    134.0173208033666,
                    53.095461883295435,
                    18.564057526698022
                ],
                "TOULOUSE": [
                    192.5653403615579,
                    43.559202312496375,
                    1.4833787228673134
                ],
                "TROMSONO": [
                    133.33001559134573,
                    69.6629743312902,
                    18.939440093516904
                ],
                "TRYSILNO": [
                    724.3894661320373,
                    61.4228347588988,
                    12.38163929484487
                ],
                "TSUKU3": [
                    70.01114054117352,
                    36.10563610229167,
                    140.08717774962582
                ],
                "TSUKUB32": [
                    85.0746934171766,
                    36.10314636599184,
                    140.08873671191932
                ],
                "TSUKUBA": [
                    69.57781562581658,
                    36.105637054305255,
                    140.08698310087877
                ],
                "UCHINOUR": [
                    376.05013986025006,
                    31.254311626027917,
                    131.07838708806432
                ],
                "URUMQI": [
                    2033.5917480634525,
                    43.47150957917675,
                    87.17813470712798
                ],
                "USSURISK": [
                    155.43927331548184,
                    44.01609691667893,
                    131.75727288898324
                ],
                "USUDA64": [
                    1533.2487824503332,
                    36.13240434440272,
                    138.36276633407158
                ],
                "VERAIRIK": [
                    573.9989769980311,
                    31.747899467329493,
                    130.43988636679612
                ],
                "VERAISGK": [
                    65.49898879416287,
                    24.412177764052085,
                    124.17099324451547
                ],
                "VERAMZSW": [
                    116.93410522583872,
                    39.13353531561963,
                    141.13255557664496
                ],
                "VERAOGSW": [
                    273.51179450564086,
                    27.091800081204664,
                    142.21661579087763
                ],
                "VERNAL": [
                    1590.9428569516167,
                    40.32696678320482,
                    -109.57072126583125
                ],
                "VICTORIA": [
                    25.635105778463185,
                    48.389535825440795,
                    -123.48695747882046
                ],
                "VLA": [
                    2115.2354975678027,
                    34.07881433972819,
                    -107.61833335072085
                ],
                "VLA-N8": [
                    2114.8874224517494,
                    34.082732000499895,
                    -107.61874535886832
                ],
                "VNDNBERG": [
                    -11.868766247294843,
                    34.55608168625382,
                    -120.61642122471828
                ],
                "WARK12M": [
                    128.30955241713673,
                    -36.43480911079069,
                    174.66325473173325
                ],
                "WARK30M": [
                    123.0354115087539,
                    -36.43316321340584,
                    174.66294904743398
                ],
                "WESTFORD": [
                    87.18017506226897,
                    42.61294818739873,
                    -71.49379377132308
                ],
                "WETTZ13N": [
                    672.9946459271014,
                    49.143911670044666,
                    12.877700849995955
                ],
                "WETTZ13S": [
                    672.9777042418718,
                    49.14341649832189,
                    12.87828030103346
                ],
                "WETTZELL": [
                    669.5346139473841,
                    49.145008006650144,
                    12.877450339989293
                ],
                "WHTHORSE": [
                    710.0091957319528,
                    60.71124400147673,
                    -135.07707818162015
                ],
                "WSTRB-07": [
                    71.47515780013055,
                    52.915295444039046,
                    6.606754595343532
                ],
                "WSTRBORK": [
                    71.61966542713344,
                    52.91529199748945,
                    6.633334137183639
                ],
                "YAKATAGA": [
                    21.528707158751786,
                    60.08146748040162,
                    -142.48646084508042
                ],
                "YAMAGU32": [
                    166.02296917513013,
                    34.21603628520932,
                    131.55709023938542
                ],

                "YARRA12M": [
                    250,
                    -29.0464,
                    115.3456
                ],

                "YEBES": [
                    980.2733666338027,
                    40.52415416216849,
                    -3.0894140098139173
                ],
                "YEBES40M": [
                    989.3372211074457,
                    40.52466574526898,
                    -3.08686212636507
                ],
                "YELLOWKN": [
                    177.2139637870714,
                    62.47935368942611,
                    -114.4723859957144
                ],
                "YLOW7296": [
                    179.2000213805586,
                    62.480580725828226,
                    -114.47930564717619
                ],
                "YUMA": [
                    239.00944708194584,
                    32.9391349412873,
                    -114.20313658885976
                ],
                "ZELENCHK": [
                    1175.4649443980306,
                    43.78780970706245,
                    41.56516243341224
                ]
            }

            return site_to_geodetic_position_mapping.get(station_name, None)

        def average_scan_lines(self, input_folder, fdets_file, time_interval_seconds, output_folder):
            """
            Averages values of a time-series file over a given time interval. However, this is not good whern it comes to averaging the noise,
            since the noise can have both negative and positive values, thus effectively bringing the mean value to unrealistic values.

            :param input_folder: Path to the input folder.
            :param fdets_file: Name of the input file.
            :param time_interval_seconds: Time interval for averaging in seconds.
            :param output_folder: Path to the output folder.
            :return: Path of the created output file.
            """
            os.makedirs(output_folder, exist_ok=True)
            header = []
            segment = []
            first_timestamp = None
            time_interval = timedelta(seconds=time_interval_seconds)
            header_written = False

            # Corrected regex pattern
            file_pattern = r"Fdets\.(\w+)(\d{4}\.\d{2}\.\d{2})(\.\w+(?:\.complete)?\.r2i\.txt)"
            match = re.match(file_pattern, fdets_file)
            if not match:
                raise ValueError("Error in average_values_by_time: filename does not match expected pattern")

            mission, date, the_rest = match.groups()

            input_file = os.path.join(input_folder, fdets_file)
            output_filename = f"Fdets.{mission}{date}-averaged_{time_interval_seconds}_seconds{the_rest}"
            output_filepath = os.path.join(output_folder, output_filename)

            with open(input_file, 'r') as file, open(output_filepath, 'w') as out_file:
                for line in file:
                    if line.startswith('#'):
                        header.append(line)
                        continue

                    parts = line.split()
                    timestamp = datetime.strptime(parts[0], "%Y-%m-%dT%H:%M:%S.%f")
                    values = [float(value) for value in parts[1:]]

                    if first_timestamp is None:
                        first_timestamp = timestamp
                        segment_start = first_timestamp
                        segment = [values]
                    elif timestamp < segment_start + time_interval:
                        segment.append(values)
                    else:
                        # Average the values in the segment
                        averaged_values = [sum(col) / len(col) for col in zip(*segment)]

                        # Write the averaged values to the output file
                        if not header_written:
                            out_file.writelines(header)
                            header_written = True

                        output_line = f"{segment_start.isoformat()} " + " ".join(map(str, averaged_values)) + "\n"
                        out_file.write(output_line)

                        # Start a new segment
                        segment_start = timestamp
                        segment = [values]

                # Write the last segment if not empty
                if segment:
                    averaged_values = [sum(col) / len(col) for col in zip(*segment)]
                    output_line = f"{segment_start.isoformat()} " + " ".join(map(str, averaged_values)) + "\n"
                    out_file.write(output_line)

            print(f'Created averaged file: {output_filepath}')
            return output_filepath


        def split_scan_by_time(self, input_folder, fdets_file, time_interval_minutes, output_folder):

            """
            Splits a time-series file into segments based on a given time interval.

            :param input_folder: Path to the input folder.
            :param fdets_file: Name of the input file.
            :param time_interval_minutes: Time interval for splitting in minutes.
            :return: List of created output file paths.
            """
            os.makedirs(output_folder, exist_ok=True)
            output_files = []
            header = []
            segment = []
            first_timestamp = None
            time_interval = timedelta(minutes=time_interval_minutes)

            input_file = os.path.join(input_folder, fdets_file)

            # Corrected regex pattern
            file_pattern = r"Fdets\.(\w+)(\d{4}\.\d{2}\.\d{2})(\.\w+(?:\.complete)?\.r2i\.txt)"

            match = re.match(file_pattern, fdets_file)
            if not match:
                raise ValueError("Error in split_scan_by_time: filename does not match expected pattern")

            mission, date, the_rest = match.groups()

            with open(input_file, 'r') as file:
                for line in file:
                    if line.startswith('#'):
                        header.append(line)  # Store header for each output file
                        continue

                    parts = line.split()
                    timestamp = datetime.strptime(parts[0], "%Y-%m-%dT%H:%M:%S.%f")

                    if first_timestamp is None:
                        first_timestamp = timestamp
                        segment_start = first_timestamp
                        segment = [line]
                    elif timestamp < segment_start + time_interval:
                        segment.append(line)
                    else:
                        # Create filename for the current segment
                        segment_end = segment_start + time_interval
                        output_filename = f"Fdets.{mission}{date}-{segment_start.strftime('%H%M')}-{segment_end.strftime('%H%M')}{the_rest}"
                        output_filepath = os.path.join(output_folder, output_filename)

                        # Write segment to file
                        with open(output_filepath, 'w') as out_file:
                            out_file.writelines(header)
                            out_file.writelines(segment)
                        output_files.append(output_filepath)

                        # Start a new segment
                        segment_start = timestamp
                        segment = [line]

            # Write the last segment if not empty
            if segment:
                segment_end = segment_start + time_interval
                output_filename = f"Fdets.{mission}{date}-{segment_start.strftime('%H%M')}-{segment_end.strftime('%H%M')}{the_rest}"
                output_filepath = os.path.join(output_folder, output_filename)

                with open(output_filepath, 'w') as out_file:
                    out_file.writelines(header)
                    out_file.writelines(segment)


                output_files.append(output_filepath)

            print(f'Created split scan files:\n{[output_file for output_file in output_files]}\n')

            return output_files


        def create_complete_scan_from_single_scans(self, files, output_folder):
            """
            This function concatenates single scans into a big, "complete" one, for ease of analysis.
            :param self:
            :param files: list, for instance:
            files = glob.glob('/Users/lgisolfi/Desktop/data_archiving-1.0/dataset/juice/juice_cruise/EC094A/Fdets.*.r2i.txt')
            :return: None
            """

            os.makedirs(output_folder, exist_ok=True)
            # Define the base pattern for your files (without the station part)
            file_pattern = r"Fdets\.\w+(\d{4})\.(\d{2})\.(\d{2})\.(\w{2})\.(\d{4})\.r2i\.txt"

            # Create a dictionary to store file handles for each station
            station_files = defaultdict(list)

            # Loop over each file to group them by station
            for file in files:
                # Use regex to extract station name from the filename
                match = re.search(file_pattern, file)
                if match:
                    station_name = match.group(4)  # Extract station name (2 letters, e.g., Ef)
                    scan_number = match.group(5)

                    # Add the file and its scan number to the corresponding station's list
                    station_files[station_name].append((file, str(scan_number)))  # Store as tuple (file, scan_number)

            # Now, create a complete file for each station
            for station, files in station_files.items():
                # Sort the files based on the scan number (second element of the tuple)
                files.sort(key=lambda x: x[1])  # Sorting by scan_number (second element of tuple)

                first_file = files[0][0]  # Get the first file to derive the output filename
                first_scan = files[0][1]
                base_name = os.path.basename(first_file)  # Get the base name of the input file
                # Construct the output filename by removing the scan number
                output_filename = base_name.replace(f'{first_scan}.', '')  # Remove the scan number

                # Insert 'complete' before 'r2i' in the output filename
                output_filename = output_filename.replace('.r2i.txt', '.complete.r2i.txt')
                # Create the complete output path
                output_filename = os.path.join(output_folder, output_filename)

                # Open the output file in write mode
                with open(output_filename, 'w') as output_file:
                    # Flag to check if the header has already been written
                    header_written = False

                    # Loop over each file for the current station, sorted by scan number
                    for file, _ in files:
                        # Open the file in read mode
                        with open(file, 'r') as f:
                            # Read the lines of the file
                            lines = f.readlines()

                            # Skip the header in all files except the first one
                            if not header_written:
                                # Write the first file's header
                                output_file.write(''.join(lines[:5]))  # Assuming the first 5 lines are the header
                                header_written = True

                            # Write the rest of the content, skipping the header lines
                            output_file.write(''.join(lines[5:]))  # Skip the first 5 lines (the header)

                print(f"Created {output_filename}")


        def read_allan_index(self, file_path):
            """
            Reads a file and extracts the value of 'allan_index'.

            Args:
                file_path (str): Path to the text file.

            Returns:
                float: The value of 'allan_index', or None if not found.
            """
            try:
                with open(file_path, 'r') as file:
                    for line in file:
                        if line.startswith('allan_index'):
                            # Split the line and extract the value
                            key, value = line.split('=')
                            return float(value.strip())
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except ValueError:
                print(f"Error parsing 'allan_index' value in file: {file_path}")
            return None


        def get_vex_file_path(self, experiment_name, mission_name):

            """
            Constructs the file path for a VEX file associated with a specific experiment.

            This function generates the complete path to the VEX file by using the provided
            experiment name. It looks up the experiment details from a predefined dictionary
            and combines the base folder, mission folder, and the VEX file name to create
            the full file path.

            Parameters:
            experiment_name (str): The name of the experiment for which the VEX file path is to be generated.

            Returns:
            str: The full file path to the corresponding VEX file.

            Example:
            vex_file_path = get_vex_file_path('experiment_1')
            """
            main_vex_folder = "vex_files"
            vex_file_mission_folder = mission_name
            vex_file_name = self.experiments[experiment_name]['vex_file_name']

            vex_file_path = os.path.join(main_vex_folder, vex_file_mission_folder, vex_file_name)
            return vex_file_path

        def get_mission_from_experiment(self, experiment_name):
            experiment_name = experiment_name.lower()
            for experiment, values in self.experiments.items():
                if experiment == experiment_name:
                    mission_name = values['mission_name']
                    return mission_name

        def generate_random_color(self):
            """Generates a random, well-spaced color in hexadecimal format."""
            r = random.randint(0, 220)  # Avoid extremes (too dark/light)
            g = random.randint(0, 220)
            b = random.randint(0, 220)
            return "#{:02x}{:02x}{:02x}".format(r, g, b)

    ########################################################################################################################################
    ########################################################################################################################################

    class Analysis:
        def __init__(self, process_fdets, utilities):
            self.result = 0
            self.Utilities = utilities
            self.ProcessFdets = process_fdets

        ########################################################################################################################################
        ########################################################################################################################################
        def plot_parameters_error_bounds(self, extracted_data, tau_min = None, tau_max = None, save_dir=None, suppress = False, plot_oadev_only = True, color_regions = False):

            """
            This function is supposed to be an improvement of plot_parameters, as
            it accounts for the errorbars in the slope computation. The "acceptable regions"
            are either:

            1) those regions for which, taking the error bars into account, satisfy

            -0.5 belongs to [slope_min, slope_max]

            with

            slope_min = [(slope[i+1] - error_plus[i+1]) - (slope[i] + error_minus[i])]/(tau[i+1] -tau[i])
            slope_max = [(slope[i+1] + error_plus[i+1]) - (slope[i] - error_minus[i])]/(tau[i+1] -tau[i])


            2) those regions for which the slope at the data point satisfies: abs(slope - target_slope) <= 0.1

            I think it is good to have this second condition, as, for small taus, the error bars are small.

            Input: object.ProcessFdets().extract_parameters [required]
                   save_dir (to save the plot) [optional, default is None]
                   suppress (to suppress the plot show) [optional, default is False]

            Output: oadev & SNR plots

            NOTES: Please note that, within this function, the slope is computed between data points only
                   (differently from the get_slope_at_tau function, where a cubic spline interpolation is used,
                   and hence one can retrieve slopes at any time).
                   The reason for this is that the error bars (which are used to compute
                   the minimum and maximum slope (two-sided triangle) are only available for the data points,
                   so it would be meaningless to interpolate the data,
                   as we do not actually know how to interpolate the error bars.

            """

            if extracted_data != None:
                # Extract data
                utc_datetime = extracted_data['utc_datetime']
                signal_to_noise = extracted_data['Signal-to-Noise']
                doppler_noise_hz = extracted_data['Doppler Noise [Hz]']
                first_col_name = extracted_data['first_col_name']
                second_col_name = extracted_data['second_col_name']
                fifth_col_name = extracted_data['fifth_col_name']
                utc_date = extracted_data['utc_date']
                base_frequency = extracted_data['base_frequency']
                frequency_detection = extracted_data['Freq detection [Hz]']

                # Number of x-ticks you want to display
                num_ticks = 15

                # Generate x-tick positions
                xticks = np.linspace(0, len(utc_datetime) - 1, num=num_ticks, dtype=int)

                # Generate x-tick labels based on positions
                xtick_labels = [utc_datetime[i].strftime("%H:%M:%S") for i in xticks]

                if plot_oadev_only:
                    # Set up the figure for only Overlapping Allan Deviation plot
                    fig, ax = plt.subplots(figsize=(10, 5))
                else:
                    # Set up the 4x1 subplot figure
                    fig, axs = plt.subplots(4, 1, figsize=(10, 15))
                    fig.subplots_adjust(hspace=0.5)

                    # Plot Signal-to-Noise vs UTC time
                    axs[1].scatter(range(len(utc_datetime)), signal_to_noise, marker='o', linestyle='-', color='blue', s = 5)
                    axs[1].set_xlabel(f'UTC Time (HH:MM:SS) on {utc_date}')
                    axs[1].set_ylabel('Signal-to-Noise Ratio')
                    axs[1].set_title(f'SNR vs UTC Time')
                    axs[1].set_xticks(xticks)
                    axs[1].set_xticklabels(xtick_labels, rotation=45)
                    axs[1].legend()
                    axs[1].grid(True)

                    # Plot Doppler noise [Hz] vs UTC time
                    axs[2].plot(range(len(utc_datetime)), doppler_noise_hz, marker='o', linestyle='-', color='orange', markersize = 3, linewidth=0.5)
                    axs[2].set_xlabel(f'UTC Time (HH:MM:SS) on {utc_date}')
                    axs[2].set_ylabel('Doppler Noise')
                    axs[2].set_title(f'Doppler Noise vs UTC Time')
                    axs[2].set_xticks(xticks)
                    axs[2].set_xticklabels(xtick_labels, rotation=45)
                    axs[2].legend()
                    axs[2].grid(True)


                    axs[3].plot(range(len(utc_datetime)), frequency_detection,  marker='o', color='black', markersize = 2, linewidth=0.5)

                    # Set labels and grid
                    axs[3].set_xlabel('UTC Time')
                    axs[3].set_ylabel('Freq Detections')
                    axs[3].set_title('Frequency Detections')
                    axs[3].tick_params(axis='x', which='major', labelsize=8, rotation = 45)
                    axs[3].set_xticks(xticks)
                    axs[3].set_xticklabels(xtick_labels, rotation=45)
                    axs[3].legend()
                    axs[3].grid(True)

                # Calculate sampling rate in Hz
                t_jd = [Time(time).jd for time in utc_datetime]
                # Calculate the differences
                diffs = np.diff(t_jd)

                # Get the most common difference
                most_common_diff = Counter(diffs).most_common(1)
                if most_common_diff and most_common_diff[0][0] != 0:
                    rate_fdets = 1 / (most_common_diff[0][0] * 86400)
                else:
                    print("Most common difference is zero or not found; cannot calculate rate_fdets. Hence, no plot is available.")
                    return(None)

                # Proceed to calculate Overlapping Allan Deviation, ensuring no invalid values
                try:
                    taus_doppler, oadev_doppler, original_errors, ns = allantools.oadev(
                        np.array(doppler_noise_hz) / (np.array(frequency_detection) + base_frequency),
                        rate=rate_fdets,
                        data_type='freq',
                        taus='decade'
                    )
                except Exception as e:
                    print(f"An error occurred: {e}")

                #filter by tau doppler times

                if tau_min is not None:
                    tau_min = float(tau_min)
                if tau_max is not None:
                    tau_max = float(tau_max)

                if tau_min:
                    taus_doppler = taus_doppler[taus_doppler >= tau_min]
                if tau_max:
                    taus_doppler = taus_doppler[taus_doppler <= tau_max]

                # Defining Weights
                oadev_doppler = oadev_doppler[:taus_doppler.shape[0]]
                errors = original_errors[:taus_doppler.shape[0]]
                rms_values = original_errors
                weights = 1 / (np.array(rms_values) ** 2)

                # Normalize the weights to a range between 0 and 1
                norm_weights = (weights - np.max(weights)) / (np.min(weights) - np.max(weights))
                cmap = cm.get_cmap('plasma')  # You can choose any colormap

                # Generate the white noise reference line
                oadev_white = [oadev_doppler[0]]  # Initialize with the first value
                for i in range(1, len(taus_doppler)):
                    oadev_white.append(oadev_doppler[0] * (taus_doppler[i] / taus_doppler[0])**(-1/2))

                log_ticks = np.logspace(np.log10(taus_doppler[0]), np.log10(taus_doppler[-1]), num=15)
                valid_xticks = np.searchsorted(taus_doppler, log_ticks)
                valid_xticks = valid_xticks[valid_xticks < len(taus_doppler)]

                if plot_oadev_only:
                    ax.legend()
                    ax.errorbar(taus_doppler, oadev_doppler, yerr=errors)
                    oadev_white = [oadev_doppler[0]]
                    for i in range(1, len(taus_doppler)):
                        oadev_white.append(oadev_doppler[0] * (taus_doppler[i] / taus_doppler[0])**(-1/2))
                    ax.loglog(taus_doppler, oadev_white, linestyle='--', color='black', label=r'White Freq. Noise $\propto{\tau^{-0.5}}$')
                    ax.loglog(taus_doppler, oadev_doppler, marker='o', markersize = 3, linestyle='dashed', color='b', label='Doppler Noise')
                    #ax.axhline(oadev_doppler[0], color='fuchsia', linestyle='dashdot', label=r'Pink Noise $\propto{\tau^0}$')
                    ax.set_xlabel('Averaging Time (s)')
                    ax.set_ylabel('Overlapping Allan Deviation')
                    ax.set_title('Overlapping Allan Deviation Plot')
                    ax.grid(True, which="both", ls="--")
                    #ax.tick_params(axis='x', which='major', labelsize=8, rotation=45)
                else:
                    axs[0].errorbar(taus_doppler, oadev_doppler, yerr=errors)
                    oadev_white = [oadev_doppler[0]]
                    for i in range(1, len(taus_doppler)):
                        oadev_white.append(oadev_doppler[0] * (taus_doppler[i] / taus_doppler[0])**(-1/2))
                    axs[0].loglog(taus_doppler, oadev_white, linestyle='--', color='black', label=r'White Freq. Noise $\propto{\tau^{-0.5}}$')
                    axs[0].loglog(taus_doppler, oadev_doppler, marker='o', linestyle='dashed', color='b', label='Doppler Noise')
                    #axs[0].axhline(oadev_doppler[0], color='fuchsia', linestyle='dashdot', label=r'Pink Noise $\propto{\tau^0}$')
                    axs[0].set_xlabel('Averaging Time (s)')
                    axs[0].set_ylabel('Overlapping Allan Deviation')
                    axs[0].set_title('Overlapping Allan Deviation Plot')
                    axs[0].grid(True, which="both", ls="--")
                    #axs[0].tick_params(axis='x', which='major', labelsize=8, rotation=45)

                if color_regions:
                    # Calculate slopes and determine is_close
                    slopes = []
                    is_close = []
                    mean_weights = []
                    target_slope = -0.5
                    threshold = 0.1  # Define a threshold for closeness

                    for i in range(len(taus_doppler) - 1):
                        slope = (np.log10(oadev_doppler[i + 1]) - np.log10(oadev_doppler[i])) / (np.log10(taus_doppler[i + 1]) - np.log10(taus_doppler[i]))
                        slopes.append(slope)

                        # here, for slope_error_minus, we are creating a "double triangle" connecting
                        # the lowest value at i with the highest at i+1 and checking the slope.
                        # same for slope_error_plus, but with interchanged endpoints

                        slope_error_plus = ((np.log10(oadev_doppler[i + 1] - errors[i + 1]) - np.log10(oadev_doppler[i] + errors[i])) /
                                            (np.log10(taus_doppler[i + 1]) - np.log10(taus_doppler[i])))
                        slope_error_minus = ((np.log10(oadev_doppler[i + 1] + errors[i + 1]) - np.log10(oadev_doppler[i] - errors[i])) /
                                             (np.log10(taus_doppler[i + 1]) - np.log10(taus_doppler[i])))

                        # Check if the slope is close to the target slope
                        is_close.append((slope_error_plus <= target_slope and target_slope <= slope_error_minus) or (slope_error_minus <= target_slope and target_slope <= slope_error_plus) or abs(slope - target_slope) <= 0.1)
                        mean_weights.append((norm_weights[i] + norm_weights[i+1])/2) #the mean of the two weights at endpoints
                        # is considered for the interval.

                    # Plot the regions where is_close is True
                    start_idx = None
                    for i in range(len(is_close)):
                        if is_close[i]:
                            if start_idx is None:
                                start_idx = i  # Start of a region


                            # For each interval in the region, use the specific weight of the interval
                            weight = mean_weights[i]  # Use weight for the current interval
                            color = cmap(weight)  # Get color based on current interval's weight

                            # Plot the interval using axvspan
                            if plot_oadev_only:
                                ax.axvspan(taus_doppler[i], taus_doppler[i + 1], color=color, alpha=0.1)
                            else:

                                axs[0].axvspan(taus_doppler[i], taus_doppler[i + 1], color=color, alpha=0.1)

                        else:
                            if start_idx is not None:
                                start_idx = None  # Reset start_idx when region ends

                    # Check for any remaining open region at the end
                    if start_idx is not None:
                        for i in range(start_idx, len(is_close) - 1):
                            if is_close[i]:
                                weight = mean_weights[i]
                                color = cmap(weight)
                                if plot_oadev_only:
                                    ax.axvspan(taus_doppler[i], taus_doppler[i + 1], color=color, alpha=0.1)
                                    hatch_patch = mpatches.Patch(
                                        facecolor='white', edgecolor='black', hatch='//', label="Acceptable Regions"
                                    )
                                    handles, labels = ax.get_legend_handles_labels()
                                    handles.append(hatch_patch)
                                    labels.append("Acceptable Regions")
                                    ax.legend(handles=handles, labels=labels)

                                else:
                                    axs[0].axvspan(taus_doppler[i], taus_doppler[i + 1], color=color, alpha=0.1)
                                    hatch_patch = mpatches.Patch(
                                        facecolor='white', edgecolor='black', hatch='//', label="Acceptable Regions"
                                    )
                                    handles, labels = ax.get_legend_handles_labels()
                                    handles.append(hatch_patch)
                                    labels.append("Acceptable Regions")
                                    ax.legend(handles=handles, labels=labels)
                                    axs[0].legend(handles=[hatch_patch])

                                    # Set x-ticks on log scale
                if plot_oadev_only:
                    ax.set_xticks(taus_doppler[valid_xticks])
                    ax.set_xticklabels([f"{round(tick)}" for tick in taus_doppler[valid_xticks]])
                    ax.legend()

                else:
                    axs[0].set_xticks(taus_doppler[valid_xticks])
                    axs[0].set_xticklabels([f"{round(tick)}" for tick in taus_doppler[valid_xticks]])
                    axs[0].legend()

                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(norm_weights), vmax=np.max(norm_weights)))
                sm.set_array([])  # Required for ScalarMappable

                if plot_oadev_only:
                    cbar = fig.colorbar(sm, ax = ax, alpha = 0.3)
                else:
                    cbar = fig.colorbar(sm, ax=axs[0], alpha = 0.3)
                cbar.set_label('Weight')

                # Save the figure if a directory is specified
                if save_dir:
                    if os.path.exists(os.path.join(save_dir, 'parameters_error_bounds_plot.png')):
                        print(f'Removing {os.path.join(save_dir, "parameters_error_bounds_plot.png")}')
                        os.remove(os.path.join(save_dir, 'parameters_error_bounds_plot.png'))
                    plt.savefig(os.path.join(save_dir, 'parameters_error_bounds_plot.png'))
                    print(f'Saving {os.path.join(save_dir, "parameters_error_bounds_plot.png")}')
                    plt.close(fig)

                if suppress == True:
                    plt.close(fig)  # Close the figure

                else:
                    plt.show()
                    plt.close(fig)

            else:
                print(f'Cannot plot the data for File due to the reasons explained above. Skipping...\n')

        ########################################################################################################################################
        ########################################################################################################################################

        def two_step_filter(self, extracted_parameters_list, keys=('Signal-to-Noise', 'Doppler Noise [Hz]'), threshold=3.5):
            if len(extracted_parameters_list) == 0:
                return extracted_parameters_list

            filtered_list = []

            for entry in extracted_parameters_list:
                keep_mask = None

                for key in keys:
                    values = np.array(entry.get(key, []))

                    if len(values) == 0:
                        continue

                    median = np.median(values)
                    mad = np.median(np.abs(values - median))

                    if mad == 0:
                        mask = np.ones_like(values, dtype=bool)
                    else:
                        modified_z = 0.6745 * (values - median) / mad
                        mask = np.abs(modified_z) < threshold

                    if keep_mask is None:
                        keep_mask = mask
                    else:
                        keep_mask &= mask  # logical AND for all keys

                # Apply filtering
                for key in entry:
                    values = entry[key]
                    if isinstance(values, list) and len(values) == len(keep_mask):
                        entry[key] = [v for v, k in zip(values, keep_mask) if k]

                # Step 2: Reject station if mean doppler noise > 0.005 Hz after filtering
                doppler_noise = entry.get("Doppler Noise [Hz]", [])
                if doppler_noise:
                    if  np.abs(np.mean(doppler_noise)) < 0.005:
                        filtered_list.append(entry)
                    else:
                        station_name = entry.get("receiving_station_name", [])
                        print(f'Station {station_name}: Bad observation detected.')
                        filtered_list.append(entry)
                else:
                    print('No doppler noise entry found. Maybe check the corresponding dictionary key name.')
            return filtered_list


        def plot_oadev_stations(self, extracted_data_list, mission_name, experiment_name = None, tau_min=None, tau_max=None, save_dir=None, suppress=False, color_regions=False):
            """
            Plots Overlapping Allan Deviation (oadev) and saves one plot per unique date and corresponding data to CSV files.

            Args:
                extracted_data_list (list): List of extracted_data dicts, each containing:
                                            - utc_datetime, signal_to_noise, doppler_noise_hz, frequency_detection, base_frequency, etc.
                tau_min (float, optional): Minimum tau for filtering.
                tau_max (float, optional): Maximum tau for filtering.
                save_dir (str, optional): Directory to save the plots and CSVs.
                suppress (bool, optional): If True, does not show the plot.
                color_regions (bool, optional): Enables color-coding regions based on error bounds.
            """

            if not isinstance(extracted_data_list, list):
                extracted_data_list = [extracted_data_list]

            # Extract all unique dates
            unique_dates_set = set()
            for extracted_data in extracted_data_list:
                utc_datetime = extracted_data['utc_datetime']
                unique_dates_set.update(day.strftime('%Y-%m-%d') for day in utc_datetime)

            unique_dates_list = sorted(list(unique_dates_set))  # Chronological order
            colors = cm.get_cmap('tab10', len(extracted_data_list))  # Unique colors per station

            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for date in unique_dates_list:
                fig, ax = plt.subplots(figsize=(10, 5))
                output_rows = []  # For CSV
                for i, extracted_data in enumerate(extracted_data_list):
                    color = colors(i)
                    receiving_station_name = extracted_data['receiving_station_name']
                    if receiving_station_name == 'O6':
                        receiving_station_name == 'On'
                    utc_datetime = extracted_data['utc_datetime']
                    doppler_noise_hz = extracted_data['Doppler Noise [Hz]']
                    frequency_detection = extracted_data['Freq detection [Hz]']
                    base_frequency = extracted_data['base_frequency']

                    # Determine sampling rate
                    t_jd = np.array([Time(time).jd for time in utc_datetime])
                    diffs = np.diff(t_jd)
                    most_common_diff = Counter(diffs).most_common(1)
                    if not most_common_diff or most_common_diff[0][0] == 0:
                        print(f"Skipping dataset {i+1}: Cannot determine sampling rate.")
                        continue

                    rate_fdets = 1 / (most_common_diff[0][0] * 86400)

                    # Compute oadev
                    try:
                        taus, oadev, errors, _ = allantools.oadev(
                            np.array(doppler_noise_hz) / (np.array(frequency_detection) + base_frequency),
                            rate=rate_fdets,
                            data_type='freq',
                            taus='decade'
                        )
                    except Exception as e:
                        print(f"Skipping dataset {i+1} due to error in Allan deviation computation: {e}")
                        continue

                    # Filter tau range
                    tau_mask = np.ones_like(taus, dtype=bool)
                    if tau_min is not None:
                        tau_mask &= taus >= tau_min
                    if tau_max is not None:
                        tau_mask &= taus <= tau_max

                    taus = taus[tau_mask]
                    oadev = np.array(oadev)[tau_mask]
                    errors = np.array(errors)[tau_mask]

                    # Only use data from this date
                    #date_data_mask = [d.strftime('%Y-%m-%d') == date for d in utc_datetime]
                    #if not any(date_data_mask):
                    #    continue

                    # Plot
                    ax.errorbar(taus, oadev, yerr=errors, fmt='o', markersize=3, linestyle='dashed',
                                color=color, label=receiving_station_name)

                    # Save rows for CSV
                    for t, m, e in zip(taus, oadev, errors):
                        output_rows.append([t, m, e, receiving_station_name])

                # Finalize plot
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel('Averaging Time (s)')
                ax.set_ylabel('Overlapping Allan Deviation')
                if experiment_name in self.Utilities.experiments:
                    ax.set_title(f'Mission {mission_name} on {date}, experiment {experiment_name}')
                else:
                    ax.set_title(f'Mission {mission_name} on {date}')
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                ax.grid(True, which="both", ls="--", alpha=0.3)

                log_ticks_x = np.logspace(np.log10(taus[0]), np.log10(taus[-1]), num=10)
                valid_xticks = np.searchsorted(taus, log_ticks_x)
                valid_xticks = valid_xticks[valid_xticks < len(taus)]
                ax.set_xticks(taus[valid_xticks])
                ax.set_xlim(9, 110)
                ax.set_xticklabels([f"{round(tick)}" for tick in taus[valid_xticks]])

                plt.tight_layout()

                # Save plot
                if save_dir:
                    fig_path = os.path.join(save_dir, f"{mission_name}_{date}_mad.png")
                    plt.savefig(fig_path)

                if not suppress:
                    plt.show()

                plt.close()

                # Save CSV
                if save_dir and output_rows:
                    csv_path = os.path.join(save_dir, f"{mission_name}_{date}_mad.csv")
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["Tau (s)", "Overlapping Allan Deviation", "Error", "Station"])
                        writer.writerows(output_rows)


        def get_all_stations_oadev_plot(self, fdets_folder_path, mission_name, experiment_name = None, extracted_parameters_list = None, tau_min = None, tau_max = None, two_step_filter = True, save_dir = None):

            if not extracted_parameters_list:
                extracted_parameters_list =list()
                directory_path = fdets_folder_path
                for file in os.listdir(directory_path):
                    if file.startswith('Fdets') and file.endswith('.txt'):
                        file_path = os.path.join(directory_path, file)
                        extracted_parameters = self.ProcessFdets.extract_parameters(file_path)
                        extracted_parameters_list.append(extracted_parameters)

            if two_step_filter:
                extracted_parameters_list = self.two_step_filter(extracted_parameters_list)

            if experiment_name:
                self.plot_oadev_stations(
                    extracted_parameters_list,
                    mission_name = mission_name,
                    experiment_name=experiment_name,
                    tau_min = tau_min,
                    tau_max = tau_max,
                    suppress = False,
                    save_dir = save_dir)
            else:
                self.plot_oadev_stations(
                    extracted_parameters_list,
                    mission_name = mission_name,
                    tau_min = tau_min,
                    tau_max = tau_max,
                    suppress = False,
                    save_dir = save_dir)

        def get_all_outputs(self, root_folder, save_index = False, save_plots = False): # Function to process and save plots for each TXT file

            """

            This function iterates over all folders for a given mission dataset and retrieves the
            plot_parameters and plot_parameters_error_bounds plots for (almost all) fdets.

            Some exceptions might be:
            - fdets with a bad formatting
            - fdets containing multiple scans
            - fdets with a bad header

            Inputs: root_folder (namely, dataset) [required]
                    JUICE [optional] (if JUICE == True, JUICE format is assumed. if JUICE == False, other missions format is assumed)

            Please note: For mex Phobos flyby, only the file complete.r*i.txt are processed

            """

            print(f'Getting Outputs From Folder: {root_folder} ...\n')
            # Compile the regex pattern to match filenames
            pattern = re.compile('r2i.txt$')
            pattern_mex = re.compile(r'complete.+r2i.txt$')

            # Iterate through all directories and subdirectories
            for dirpath, _, filenames in os.walk(root_folder):
                for filename in filenames:
                    # Skip files based on certain conditions
                    if 'Phases' in filename or 'sum' in filename:
                        continue
                    # Check if the filename matches the pattern and ends with .txt
                    if pattern.search(filename):
                        print(f'Processing Directory:{dirpath}')
                        if filename.endswith('.txt'):
                            # Full path to the TXT file
                            txt_file_path = os.path.join(dirpath, filename)


                            try:
                                # Extract data from the text file
                                extracted_data = self.ProcessFdets.extract_parameters(txt_file_path)

                            except Exception as e:
                                print(f"Error processing file {filename}: {e}")
                                continue
                            # Create a directory for saving plots
                            plot_dir = os.path.join(dirpath, os.path.splitext(filename)[0])  # Use the TXT file name without extension
                            os.makedirs(plot_dir, exist_ok=True)

                            if save_plots == True:
                                # Generate and save plots, suppresses output so as not to show all the plots
                                print(f'Generating Plot File...\n')
                                self.plot_user_defined_parameters(
                                    extracted_data,
                                    save_dir = plot_dir,
                                    suppress = True,
                                    plot_snr = True,
                                    plot_doppler_noise = True,
                                    plot_fdets = True)

                                self.plot_parameters_error_bounds(
                                    extracted_data,
                                    save_dir = plot_dir,
                                    suppress = True,
                                    plot_oadev_only = True,
                                    tau_min = 0,
                                    tau_max = 100)

                            if save_index == True:
                                # Generate and save allan index report
                                print(f'Generating Allan Index File...')
                                self.get_allan_index(extracted_data, save_dir = plot_dir, suppress = True)

                    if pattern_mex.search(filename): #mex phobos flyby has slightly different names and problematic files
                        print(f'Processing Directory:{dirpath}')
                        if filename.endswith('.txt'):
                            # Full path to the TXT file
                            txt_file_path = os.path.join(dirpath, filename)

                            try:
                                # Extract data from the text file
                                extracted_data = self.ProcessFdets.extract_parameters(txt_file_path)

                            except Exception as e:
                                print(f"Error processing file {filename}: {e}")
                                continue
                            # Create a directory for saving plots
                            plot_dir = os.path.join(dirpath, os.path.splitext(filename)[0])  # Use the TXT file name without extension
                            os.makedirs(plot_dir, exist_ok=True)

                            if save_plots == True:
                                # Generate and save plots, suppresses output so as not to show all the plots
                                print(f'Generating Plot File...\n')
                                self.plot_user_defined_parameters(
                                    extracted_data,
                                    save_dir = plot_dir,
                                    suppress = True,
                                    plot_snr = True,
                                    plot_doppler_noise = True,
                                    plot_fdets = True)

                                self.plot_parameters_error_bounds(
                                    extracted_data,
                                    save_dir = plot_dir,
                                    suppress = True,
                                    plot_oadev_only = True,
                                    tau_min = 0,
                                    tau_max = 100)

                            if save_index == True:
                                # Generate and save allan index report
                                print(f'Generating Allan Index File...')
                                self.get_allan_index(extracted_data, save_dir = plot_dir, suppress = True)


            print(f'...Done.\n')

        ########################################################################################################################################
        ########################################################################################################################################

        ########################################################################################################################################
        ########################################################################################################################################

        def get_allan_index(self, extracted_data, tau_min = None, tau_max = None, save_dir=None, suppress=False):

            """
            This function computes the Allan Index, based on the two arrays:

            1) is_close = array made of as many boolean values (true or false) as the number of the fdets data points
            2) average_weights = average weight between two consecutive data points

            Input: object.ProcessFdets().extract_parameters [required]
                   save_dir (to create and save a .txt file report) [optional, default is None]
                   suppress (to suppress the function output, useful when used within Get_All_Plots) [optional, default is False]

            Output: allan_index.txt and/or allan_index value

            NOTES: Please note that, within this function, the slope is computed between data points only
                   (differently from the get_slope_at_tau function, where a cubic spline interpolation is used,
                   and hence one can retrieve slopes at any time).
                   The reason for this is that the error bars (which are used to compute
                   the minimum and maximum slope (two-sided triangle) are only available for the data points,
                   so it would be meaningless to interpolate the data,
                   as we do not actually know how to interpolate the error bars.

            """

            if extracted_data != None:

                # Extract data
                utc_datetime = extracted_data['utc_datetime']
                signal_to_noise = extracted_data['Signal-to-Noise']
                doppler_noise_hz = extracted_data['Doppler Noise [Hz]']
                first_col_name = extracted_data['first_col_name']
                second_col_name = extracted_data['second_col_name']
                fifth_col_name = extracted_data['fifth_col_name']
                utc_date = extracted_data['utc_date']
                base_frequency = extracted_data['base_frequency']
                frequency_detection = extracted_data['Freq detection [Hz]']

                # Calculate sampling rate in Hz
                t_jd = [Time(time).jd for time in utc_datetime]


                # Calculate the differences
                diffs = np.diff(t_jd)

                # Get the most common difference
                most_common_diff = Counter(diffs).most_common(1)
                if most_common_diff and most_common_diff[0][0] != 0:
                    rate_fdets = 1 / (most_common_diff[0][0] * 86400)
                else:
                    print("Most common difference is zero or not found; cannot calculate rate_fdets. Hence, no plot is available.")
                    return(None)
                # Proceed to calculate Overlapping Allan Deviation, ensuring no invalid values
                try:
                    taus_doppler, oadev_doppler, errors, ns = allantools.oadev(
                        np.array(doppler_noise_hz) / (np.array(frequency_detection) + base_frequency),
                        rate=rate_fdets,
                        data_type='freq',
                        taus='all'
                    )
                except Exception as e:
                    print(f"An error occurred: {e}")

                # Defining Weights
                rms_values = errors
                weights = 1 / (np.array(rms_values) ** 2)

                # Normalize the weights to a range between 0 and 1
                norm_weights = (weights - np.max(weights)) / (np.min(weights) - np.max(weights))
                cmap = cm.get_cmap('plasma')  # You can choose any colormap

                # Generate the white noise reference line
                oadev_white = [oadev_doppler[0]]  # Initialize with the first value
                for i in range(1, len(taus_doppler)):
                    oadev_white.append(oadev_doppler[0] * (taus_doppler[i] / taus_doppler[0])**(-1/2))

                # Calculate slopes and determine is_close
                slopes = []
                is_close = []
                mean_weights = []
                target_slope = -0.5
                threshold = 0.01  # Define a threshold for closeness

                if tau_min is not None:
                    tau_min = float(tau_min)
                if tau_max is not None:
                    tau_max = float(tau_max)

                if tau_min:
                    taus_doppler = np.array(taus_doppler)[taus_doppler >= tau_min]
                if tau_max:
                    taus_doppler = np.array(taus_doppler)[taus_doppler <= tau_max]

                taus_doppler = list(taus_doppler)
                num_points = len(taus_doppler) - 1

                for i in range(num_points):
                    slope = (np.log10(oadev_doppler[i + 1]) - np.log10(oadev_doppler[i])) / (np.log10(taus_doppler[i + 1]) - np.log10(taus_doppler[i]))
                    slopes.append(slope)

                    # here, for slope_error_minus, we are creating a "double triangle" connecting
                    # the lowest value at i with the highest at i+1 and checking the slope.
                    # same for slope_error_plus, but with interchanged endpoints

                    slope_error_plus = ((np.log10(oadev_doppler[i + 1] - errors[i + 1]) - np.log10(oadev_doppler[i] + errors[i])) /
                                        (np.log10(taus_doppler[i + 1]) - np.log10(taus_doppler[i])))
                    slope_error_minus = ((np.log10(oadev_doppler[i + 1] + errors[i + 1]) - np.log10(oadev_doppler[i] - errors[i])) /
                                         (np.log10(taus_doppler[i + 1]) - np.log10(taus_doppler[i])))

                    # Check if the slope is close to the target slope
                    is_close.append((slope_error_plus <= target_slope and target_slope <= slope_error_minus) or (slope_error_minus <= target_slope and target_slope <= slope_error_plus) or abs(slope - target_slope) <= 0.1)

                    mean_weights.append((norm_weights[i] + norm_weights[i+1])/2) #the mean of the two weights at endpoints
                    # is considered for the interval.

                    # Convert lists to NumPy arrays
                    is_close_array = np.array(is_close)
                    mean_weights_array = np.array(mean_weights)


                # Convert is_close to float (1 for True, 0 for False) for multiplication
                is_close_float = is_close_array.astype(float)

                # Element-wise multiplication
                product_array = is_close_float * mean_weights_array
                self.allan_index = np.sum(product_array)/(num_points -1)


                # Prepare data to save in the required format
                #is_close_str = f"is_close: [{', '.join(map(str, is_close_array))}]"
                #mean_weights_str = f"average_weights:  [{', '.join(map(str, mean_weights_array))}]"
                allan_index_str = f"allan_index = {self.allan_index}"


                # Save to file if save_dir is specified
                if save_dir:
                    # Ensure the directory exists
                    os.makedirs(save_dir, exist_ok=True)
                    file_path = os.path.join(save_dir, 'allan_index.txt')
                    if os.path.exists(file_path):
                        print(f'Removing {file_path}')

                    # Save data to file
                    with open(file_path, 'w') as f:
                        #f.write(f"{is_close_str}\n")
                        #f.write(f"{mean_weights_str}\n")
                        f.write(f"{allan_index_str}\n")

                    print(f'Saved Allan index data to: {file_path}\n')

                if suppress == False:
                    return (self.allan_index)
            else:
                print(f'Cannot compute Allan Index due to the reasons explained above. Skipping ...')

        ########################################################################################################################################
        ########################################################################################################################################
        #Function to compute the averaged Overlapping Allan Deviation Slope for a given tau (in seconds)
        def get_slope_at_tau(self, extracted_data, tau, delta_tau):

            """

            This function computes the oadev slope at a given tau, using numerical derivatives of the type:

            slope = [oadev(tau) + oadev(tau+dt)]/dt

            The choice of tau and dt is somehow arbitrary and is given as input

            Inputs: object.ProcessFdets().extract_parameters [required]
                    tau [required] (in seconds)
                    dt [required] (in seconds)

            """

            utc_datetime = extracted_data['utc_datetime']
            base_frequency = extracted_data['base_frequency']
            frequency_detection = extracted_data['Freq detection [Hz]']
            doppler_noise_hz = extracted_data['Doppler Noise [Hz]']

            # Convert UTC datetime to Julian Date
            t_jd = [Time(time).jd for time in utc_datetime]
            # Calculate the differences
            diffs = np.diff(t_jd)

            # Get the most common difference
            most_common_diff = Counter(diffs).most_common(1)
            if most_common_diff and most_common_diff[0][0] != 0:
                rate_fdets = 1 / (most_common_diff[0][0] * 86400)
            else:
                print("Most common difference is zero or not found; cannot calculate rate_fdets. Hence, no plot is available.")
                return(None)


            # Proceed to calculate Overlapping Allan Deviation, ensuring no invalid values
            try:
                taus_doppler, oadev_doppler, errors, ns = allantools.oadev(
                    np.array(doppler_noise_hz) / (np.array(frequency_detection) + base_frequency),
                    rate=rate_fdets,
                    data_type='freq',
                    taus='all'
                )
            except Exception as e:
                print(f"An error occurred: {e}")

            # Calculate Overlapping Allan Deviation for Doppler noise
            taus_doppler, oadev_doppler, errors, ns = allantools.oadev(
                np.array(doppler_noise_hz) / (np.array(frequency_detection) + base_frequency),
                rate=rate_fdets,
                data_type='freq',
                taus='all'
            )
            # Ensure tau and interpol_time are within the range of taus_doppler
            if tau >= max(taus_doppler) or (tau + delta_tau) >= max(taus_doppler):
                print("Most common difference is zero or not found; cannot calculate rate_fdets. Hence, no plot is available.")
                return(None)


            # Interpolation
            interpolated_data = CubicSpline(taus_doppler, oadev_doppler, extrapolate=True)

            # Compute values at tau and tau + delta_tau
            oadev_tau = interpolated_data(tau)
            oadev_tau_delta = interpolated_data(tau + delta_tau)

            # Compute the slope in the log-log plot
            if oadev_tau == 0 or oadev_tau_delta == 0:
                print("oadev values at tau or tau + delta_tau cannot be zero.")
                return(None)

            self.slope = (np.log10(oadev_tau_delta) - np.log10(oadev_tau)) / (np.log10(tau + delta_tau) - np.log10(tau))

            return(self.slope)

        ########################################################################################################################################
        ########################################################################################################################################

        def plot_user_defined_parameters(self, extracted_data, save_dir=None, suppress=False,
                                         plot_snr=False, plot_doppler_noise=False, plot_fdets=False, plot_mad=False):
            """
            Description:
            Plots user-defined parameters (such as Signal-to-Noise Ratio, Doppler Noise, Frequency Detections, and MAD)
            extracted from observational data over a specified time range (UTC). The function generates subplots for the
            selected parameters and saves the plot as an image file in the specified directory. It also displays the plot
            on the screen unless suppression is enabled.

            Input:
            extracted_data : dict
                A dictionary containing the following keys:
                - 'utc_datetime' (list of datetime objects): UTC times of the observations.
                - 'Signal-to-Noise' (list of floats): Signal-to-Noise Ratio values.
                - 'Doppler Noise [Hz]' (list of floats): Doppler noise values in Hz.
                - 'Freq detection [Hz]' (list of ints): Frequency detection counts.
                - 'utc_date' (str): Date of the observations (e.g., 'YYYY-MM-DD').
                - 'receiving_station_name' (str): Name of the receiving station.

            save_dir : str, optional
                Directory where the plots will be saved. If not specified, the plot is not saved.

            suppress : bool, optional
                If True, suppresses displaying the plot. Default is False.

            plot_snr : bool, optional
                If True, plots the Signal-to-Noise Ratio. Default is False.

            plot_doppler_noise : bool, optional
                If True, plots the Doppler Noise. Default is False.

            plot_fdets : bool, optional
                If True, plots the Frequency Detections. Default is False.

            plot_mad : bool, optional
                If True, plots the Mean Absolute Deviation (MAD). Default is False.

            Output:
            None
                The function displays the plot and saves the figure to the specified directory if a valid save_dir is given.
                If save_dir is provided, a plot image file (PNG) is saved under a folder named after the active flags, together with the correspoding .txt file
                If suppress=False, the plot is also shown on the screen.
            """
            if extracted_data is not None:
                utc_datetime = extracted_data['utc_datetime']
                signal_to_noise = extracted_data['Signal-to-Noise']
                doppler_noise_hz = extracted_data['Doppler Noise [Hz]']
                frequency_detection = extracted_data['Freq detection [Hz]']
                utc_date = extracted_data['utc_date']
                receiving_station = extracted_data["receiving_station_name"]

                num_ticks = 15
                xticks = np.linspace(0, len(utc_datetime) - 1, num=num_ticks, dtype=int)
                xtick_labels = [utc_datetime[i].strftime("%H:%M:%S") for i in xticks]

                plot_flags = {'snr': plot_snr, 'noise': plot_doppler_noise, 'fdets': plot_fdets, 'mad': plot_mad}
                active_flags = [key for key, value in plot_flags.items() if value]

                fig, axs = plt.subplots(len(active_flags), 1, figsize=(12, 10))
                fig.subplots_adjust(hspace=0.5)
                if len(active_flags) == 1:
                    axs = [axs]  # Ensure axs is always iterable

                plot_index = 0
                output_data = []

                if plot_snr:
                    filtered_utc_datetime_snr = utc_datetime
                    filtered_signal_to_noise = signal_to_noise
                    axs[plot_index].plot(range(len(utc_datetime)), signal_to_noise, marker='+', linestyle='-',
                                         color='blue', markersize=5, linewidth=0.5)

                    for t, snr in zip(filtered_utc_datetime_snr, filtered_signal_to_noise):
                        output_data.append([t.strftime('%Y-%m-%d %H:%M:%S'), 'SNR', snr])

                    axs[plot_index].set_xlabel(f'UTC Time (HH:MM:SS) on {utc_date}')
                    axs[plot_index].set_ylabel('Signal-to-Noise Ratio')
                    axs[plot_index].set_title('SNR vs UTC Time')
                    axs[plot_index].set_xticks(xticks)
                    axs[plot_index].set_xticklabels(xtick_labels, rotation=45, fontsize=8)
                    axs[plot_index].grid(True)
                    plot_index += 1

                if plot_doppler_noise:
                    filtered_utc_datetime_doppler = utc_datetime
                    filtered_doppler_noise_hz = doppler_noise_hz
                    axs[plot_index].plot(range(len(utc_datetime)), doppler_noise_hz, marker='+', linestyle='-',
                                         color='orange', markersize=5, linewidth=0.5)

                    for t, doppler in zip(filtered_utc_datetime_doppler, filtered_doppler_noise_hz):
                        output_data.append([t.strftime('%Y-%m-%d %H:%M:%S'), 'Doppler_noise', doppler])

                    axs[plot_index].set_xlabel(f'UTC Time (HH:MM:SS) on {utc_date}')
                    axs[plot_index].set_ylabel('Doppler Noise')
                    axs[plot_index].set_title('Doppler Noise vs UTC Time')
                    axs[plot_index].set_xticks(xticks)
                    axs[plot_index].set_xticklabels(xtick_labels, rotation=45, fontsize=8)
                    axs[plot_index].grid(True)
                    plot_index += 1

                if plot_fdets:
                    axs[plot_index].scatter(range(len(utc_datetime)), frequency_detection, marker='o',
                                            color='black', s=5)
                    for t, fd in zip(utc_datetime, frequency_detection):
                        output_data.append([t.strftime('%Y-%m-%d %H:%M:%S'), 'Freq_Detection', fd])

                    axs[plot_index].set_xlabel(f'UTC Time (HH:MM:SS) on {utc_date}')
                    axs[plot_index].set_ylabel('Freq Detections')
                    axs[plot_index].set_title('Frequency Detections')
                    axs[plot_index].set_xticks(xticks)
                    axs[plot_index].set_xticklabels(xtick_labels, rotation=45, fontsize=8)
                    axs[plot_index].grid(True)
                    plot_index += 1

                if save_dir:
                    os.makedirs(f"{save_dir}/{'_'.join(active_flags)}", exist_ok=True)
                    filename_base = f"{receiving_station}_{utc_date}-{xtick_labels[0]}-{xtick_labels[-1]}_{'_'.join(active_flags)}"
                    plot_path = os.path.join(f"{save_dir}/{'_'.join(active_flags)}", f"{filename_base}.png")
                    data_path = os.path.join(f"{save_dir}/{'_'.join(active_flags)}", f"{filename_base}.txt")

                    plt.tight_layout()
                    plt.savefig(plot_path)
                    print(f"Figure saved to: {plot_path}")

                    with open(data_path, 'w') as f:
                        f.write("UTC Time, Parameter, Value\n")
                        for row in output_data:
                            f.write(f"{row[0]}, {row[1]}, {row[2]}\n")
                    print(f"Data saved to: {data_path}")

                if not suppress:
                    plt.show()

                plt.close()

        ########################################################################################################################################
        ########################################################################################################################################

        def get_doppler_noise_statistics(self, extracted_data_list, mission_name, save_dir=None, suppress=False):
            """
            Plots histograms of Doppler noise for each station, grouped by date.

            Args:
                extracted_data_list (list): List of extracted_data dicts, each containing:
                                            - utc_datetime, signal_to_noise, doppler_noise_hz, frequency_detection, base_frequency, etc.
                mission_name (str): Name of the mission.
                save_dir (str, optional): Directory to save the plot.
                suppress (bool, optional): If True, does not show the plot.
            """
            if not isinstance(extracted_data_list, list):
                extracted_data_list = [extracted_data_list]  # Ensure list format

            # Extract unique stations
            unique_stations = {extracted_data['receiving_station_name'] for extracted_data in extracted_data_list}

            for station in unique_stations:
                # Filter data for this station
                station_data_list = [data for data in extracted_data_list if data['receiving_station_name'] == station]

                # Extract unique dates
                unique_dates_set = set()
                for extracted_data in station_data_list:
                    utc_datetime = extracted_data['utc_datetime']
                    unique_dates_set.update(d.strftime('%Y-%m-%d') for d in utc_datetime)

                unique_dates_list = sorted(unique_dates_set)
                num_days = len(unique_dates_list)

                # Create a separate figure for each station
                fig, axs = plt.subplots(num_days, 1, figsize=(10, 5 * num_days), sharex=True)
                if num_days == 1:
                    axs = [axs]  # Ensure axs is always a list

                colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(station_data_list)))  # Unique colors per station data

                for i, extracted_data in enumerate(station_data_list):
                    utc_datetime = np.array(extracted_data['utc_datetime'])
                    doppler_noise_hz = np.array(extracted_data['Doppler Noise [Hz]'])

                    for date in unique_dates_list:
                        date_data_mask = np.array([d.strftime('%Y-%m-%d') == date for d in utc_datetime])

                        if np.any(date_data_mask):  # Only plot if there is data for this date
                            doppler_noise_filtered = doppler_noise_hz[date_data_mask]  # Apply mask

                            mean_doppler_noise = np.mean(doppler_noise_filtered)
                            rms_doppler_noise = np.sqrt(np.mean(doppler_noise_filtered**2))
                            ax = axs[unique_dates_list.index(date)]
                            ax.hist(doppler_noise_filtered, bins=30, alpha=0.6, color=colors[i], label = f'mean: {round(mean_doppler_noise, 3)}, rms = {round(rms_doppler_noise, 3)}')

                            # Fit Gaussian
                            mu, std = norm.fit(doppler_noise_filtered)

                            # Plot Gaussian
                            fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=True)
                            ax.hist(doppler_noise_filtered, bins=30, density=True, alpha=0.6)
                            xmin, xmax = ax.get_xlim()  # Get the range from the histogram plot
                            x = np.linspace(xmin, xmax, 100)  # Create 100 evenly spaced points between
                            p = norm.pdf(x, mu, std)
                            ax.plot(x, p, 'k', linewidth=2, linestyle = '--', label=f'Gaussian fit: μ={mu:.3e}, σ={std:.3e}', alpha = 0.4)
                            ax.set_xlabel('Doppler Noise (Hz)')
                            ax.set_ylabel('Counts')
                            ax.set_title(f'{station} Station - Mission {mission_name} on {date}')
                            ax.legend()

                plt.tight_layout()

                if save_dir:
                    os.makedirs(f'{save_dir}/doppler_noise_histograms/', exist_ok=True)
                    plt.savefig(f"{save_dir}/doppler_noise_histograms/{station}_doppler_noise_hist.png")

                if not suppress:
                    plt.show()
                plt.close()
        def get_snr_statistics(self, extracted_data_list, mission_name, save_dir=None, suppress=False):
            """
            Plots histograms of SNR for each station, grouped by date.

            Args:
                extracted_data_list (list): List of extracted_data dicts, each containing:
                                            - utc_datetime, signal_to_noise, doppler_noise_hz, frequency_detection, base_frequency, etc.
                mission_name (str): Name of the mission.
                save_dir (str, optional): Directory to save the plot.
                suppress (bool, optional): If True, does not show the plot.
            """
            if not isinstance(extracted_data_list, list):
                extracted_data_list = [extracted_data_list]  # Ensure list format

            # Extract unique stations
            unique_stations = {extracted_data['receiving_station_name'] for extracted_data in extracted_data_list}

            for station in unique_stations:
                # Filter data for this station
                station_data_list = [data for data in extracted_data_list if data['receiving_station_name'] == station]

                # Extract unique dates
                unique_dates_set = set()
                for extracted_data in station_data_list:
                    utc_datetime = extracted_data['utc_datetime']
                    unique_dates_set.update(d.strftime('%Y-%m-%d') for d in utc_datetime)

                unique_dates_list = sorted(unique_dates_set)
                num_days = len(unique_dates_list)

                # Create a separate figure for each station
                fig, axs = plt.subplots(num_days, 1, figsize=(10, 5 * num_days), sharex=True)
                if num_days == 1:
                    axs = [axs]  # Ensure axs is always a list

                colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(station_data_list)))  # Unique colors per station data

                for i, extracted_data in enumerate(station_data_list):
                    utc_datetime = extracted_data['utc_datetime']
                    snr = np.array(extracted_data['Signal-to-Noise'])

                    for date in unique_dates_list:
                        date_data_mask = np.array([d.strftime('%Y-%m-%d') == date for d in utc_datetime])

                    if np.any(date_data_mask):  # Only plot if there is data for this date
                        snr_filtered = snr[date_data_mask]  # Apply mask
                        mean_snr = np.mean(snr_filtered)
                        rms_snr = np.sqrt(np.mean(snr_filtered**2))

                        ax = axs[unique_dates_list.index(date)]
                        ax.hist(snr_filtered, bins=30, alpha=0.3, color=colors[i], label = f'mean: {round(mean_snr, 4)}, rms = {round(rms_snr, 4)}')
                        ax.set_xlabel('Signal to Noise Ratio (SNR)')
                        ax.set_ylabel('Counts')
                        ax.set_title(f'{station} Station - Mission {mission_name} on {date}')
                        ax.legend()

                plt.tight_layout()

                if save_dir:
                    os.makedirs(f'{save_dir}/snr_histograms', exist_ok=True)
                    plt.savefig(f"{save_dir}/snr_histograms/{station}_snr_hist.png")

                if not suppress:
                    plt.show()

                plt.close()


        def plot_doppler_noise_distribution(self, extracted_data_list, mission_name, save_dir=None, suppress=True):
            """
            Plots the Doppler noise distribution for all stations in a single histogram using sns.displot.
            Adjusts for stations with very bad noise.

            Args:
                extracted_data_list (list): List of extracted_data dicts, each containing:
                                            - 'utc_datetime': List of timestamps
                                            - 'Doppler Noise [Hz]': List of Doppler noise values
                                            - 'receiving_station_name': Name of the station
                mission_name (str): Name of the mission.
                save_dir (str, optional): Directory to save the plot.
                suppress (bool, optional): If True, does not show the plot.
            """

            # Convert extracted data into a DataFrame
            data_list = []
            for extracted_data in extracted_data_list:
                station_name = extracted_data['receiving_station_name']
                #if station_name == 'Wz':
                #    continue
                doppler_noise_hz = extracted_data['Doppler Noise [Hz]']
                # Store each value with its corresponding station
                for value in doppler_noise_hz:
                    data_list.append({'Doppler Noise (Hz)': value, 'Station': station_name})

            df = pd.DataFrame(data_list)  # Convert to DataFrame

            # Create Seaborn distribution plot (histogram)
            sns.set(style="whitegrid")
            plt.figure(figsize=(15, 10))



            g = sns.displot(df, x="Doppler Noise (Hz)", hue="Station", kind="kde", palette="tab10", fill = True)


            plt.title(f"Doppler Noise Distribution - {mission_name}")
            plt.xlabel("Doppler Noise (Hz)")
            plt.ylabel("Counts")
            plt.tight_layout()

            if save_dir:
                os.makedirs(f"{save_dir}/doppler_noise_histograms/", exist_ok = True)
                plt.savefig(f"{save_dir}/doppler_noise_histograms/all_stations_doppler_noise_distribution.png",  dpi='figure')

            if not suppress:
                plt.show()

            plt.close()

        def plot_snr_distribution(self, extracted_data_list, mission_name, save_dir=None, suppress=True):
            """
            Plots the Doppler noise distribution for all stations in a single histogram using sns.displot.
            Adjusts for stations with very bad noise.

            Args:
                extracted_data_list (list): List of extracted_data dicts, each containing:
                                            - 'utc_datetime': List of timestamps
                                            - 'snr': List of snr values
                                            - 'receiving_station_name': Name of the station
                mission_name (str): Name of the mission.
                save_dir (str, optional): Directory to save the plot.
                suppress (bool, optional): If True, does not show the plot.
            """

            # Convert extracted data into a DataFrame
            data_list = []
            for extracted_data in extracted_data_list:
                station_name = extracted_data['receiving_station_name']
                #if station_name == 'Wz':
                #    continue
                snr = extracted_data['Signal-to-Noise']

                # Store each value with its corresponding station
                for value in snr:
                    data_list.append({'SNR': value, 'Station': station_name})

            df = pd.DataFrame(data_list)  # Convert to DataFrame

            # Create Seaborn distribution plot (histogram)
            sns.set(style="whitegrid")
            plt.figure(figsize=(15, 10))

            g = sns.displot(df, x="SNR", hue="Station", kind="kde", palette="tab10", fill = True)


            plt.title(f"SNR Distribution - {mission_name}")
            plt.xlabel("SNR")
            plt.ylabel("Counts")
            plt.xscale('log')
            plt.tight_layout()

            if save_dir:
                os.makedirs(f"{save_dir}/snr_histograms/", exist_ok = True)
                plt.savefig(f"{save_dir}/snr_histograms/all_stations_snr_distribution.png",  dpi='figure')

            if not suppress:
                plt.show()

            plt.close()

        def plot_snr_and_doppler_noise_statistics(self, extracted_data_list, mission_name, save_dir=None, suppress=False):
            """
            Computes the median and standard deviation of SNR and Doppler noise for each station on each day,
            and creates two subplots with error bars representing 1 standard deviation.

            Args:
                extracted_data_list (list): List of extracted_data dicts, each containing:
                                            - 'utc_datetime': List of timestamps
                                            - 'Signal-to-Noise': List of SNR values
                                            - 'Doppler Noise [Hz]': List of Doppler noise values
                                            - 'receiving_station_name': Name of the station
                mission_name (str): Name of the mission.
                save_dir (str, optional): Directory to save the plot.
                suppress (bool, optional): If True, does not show the plot.
            """
            station_stats = {}

            for extracted_data in extracted_data_list:
                station_name = extracted_data['receiving_station_name']
                #if station_name == 'Wz':
                #    continue
                utc_datetime = extracted_data['utc_datetime']
                snr = np.array(extracted_data['Signal-to-Noise'])
                doppler_noise = np.array(extracted_data['Doppler Noise [Hz]'])

                unique_dates = sorted(set(d.strftime('%Y-%m-%d') for d in utc_datetime))

                for date in unique_dates:
                    date_mask = np.array([d.strftime('%Y-%m-%d') == date for d in utc_datetime])

                    if np.any(date_mask):
                        median_snr = np.median(snr[date_mask])
                        std_snr = np.std(snr[date_mask])
                        median_doppler = np.median(doppler_noise[date_mask])
                        std_doppler = np.std(doppler_noise[date_mask])

                        if station_name not in station_stats:
                            station_stats[station_name] = {'dates': [], 'median_snr': [], 'std_snr': [], 'median_doppler': [], 'std_doppler': []}

                        station_stats[station_name]['dates'].append(date)
                        station_stats[station_name]['median_snr'].append(median_snr)
                        station_stats[station_name]['std_snr'].append(std_snr)
                        station_stats[station_name]['median_doppler'].append(median_doppler)
                        station_stats[station_name]['std_doppler'].append(std_doppler)

            stations = list(station_stats.keys())
            median_snr_values = [np.mean(stats['median_snr']) for stats in station_stats.values()]
            std_snr_values = [np.mean(stats['std_snr']) for stats in station_stats.values()]
            median_doppler_values = [np.mean(stats['median_doppler']) for stats in station_stats.values()]
            std_doppler_values = [np.mean(stats['std_doppler']) for stats in station_stats.values()]

            fig, axs = plt.subplots(2, 1, figsize=(12, 8))
            axs[0].errorbar(stations, median_snr_values, yerr=std_snr_values, fmt='o', markersize = 2, capsize=5, label='SNR')
            axs[0].set_xlabel("Station")
            axs[0].set_ylabel("SNR")
            axs[0].set_title("Median SNR with 1σ Standard Deviation")
            axs[0].grid(True)


            axs[1].errorbar(stations, median_doppler_values, yerr=std_doppler_values, markersize = 2, fmt='o', capsize=5, label='Doppler Noise', color='r')
            axs[1].set_xlabel("Station")
            axs[1].set_ylabel("Doppler Noise (Hz)")
            axs[1].set_title("Median Doppler Noise with 1σ Standard Deviation")
            axs[1].grid(True)

            plt.tight_layout()

            if save_dir:
                os.makedirs(f'{save_dir}', exist_ok=True)
                plt.savefig(f"{save_dir}/snr_and_doppler_noise_statistics_{mission_name}.png")

            if not suppress:
                plt.show()

            plt.close()

        def get_all_stations_statistics(self, fdets_folder_path,  mission_name, extracted_parameters_list = None, doppler_noise_statistics = False, snr_statistics = False, suppress = True, save_dir = None):


            """
            Description:
            Extracts and processes Doppler noise and Signal-to-Noise Ratio (SNR) statistics for multiple receiving stations.
            The function scans a specified directory for files that match a certain naming convention and extracts parameters
            from these files. It then generates various plots and statistics for Doppler noise and SNR based on the extracted
            data. The resulting figures can be saved to a specified directory, and the display of the plots can be suppressed.

            Input:
            fdets_folder_path : str
                The path to the directory containing the Fdets text files from which parameters will be extracted.

            mission_name : str
                Name of the experiment for labeling the plots.

            extracted_parameters_list : list, optional
                A pre-existing list of extracted parameters. If not provided, the function will extract parameters from the files.

            doppler_noise_statistics : bool, optional
                If True, generates Doppler noise statistics and related plots. Default is False.

            snr_statistics : bool, optional
                If True, generates SNR statistics and related plots. Default is False.

            suppress : bool, optional
                If True, suppresses displaying the plots. Default is True.

            save_dir : str, optional
                Directory where the generated plots will be saved. If not specified, the plots are not saved.

            Output:
            None
                The function processes the data, generates statistics and plots, and saves them to the specified directory if
                a valid `save_dir` is provided. It also prints messages indicating the progress of data extraction.
            """

            if not extracted_parameters_list:
                extracted_parameters_list =list()
                directory_path = fdets_folder_path

                for file in os.listdir(directory_path):
                    if file.startswith('Fdets') and file.endswith('.txt'):
                        file_path = os.path.join(directory_path, file)

                        print(f'Extracting data from {file}')

                        extracted_parameters = self.ProcessFdets.extract_parameters(file_path)

                        extracted_parameters_list.append(extracted_parameters)

            if doppler_noise_statistics:
                self.plot_doppler_noise_distribution(
                    extracted_parameters_list,
                    mission_name = mission_name,
                    save_dir = save_dir,
                    suppress = suppress)

                self.get_doppler_noise_statistics(
                    extracted_parameters_list,
                    mission_name = mission_name,
                    save_dir = save_dir,
                    suppress = suppress)

            if snr_statistics:

                self.plot_snr_distribution(
                    extracted_parameters_list,
                    mission_name = mission_name,
                    save_dir = save_dir,
                    suppress = suppress)

                self.plot_snr_and_doppler_noise_statistics(
                    extracted_parameters_list,
                    mission_name = mission_name,
                    save_dir = save_dir,
                    suppress = suppress)

                self.get_snr_statistics(
                    extracted_parameters_list,
                    mission_name = mission_name,
                    save_dir = save_dir,
                    suppress = suppress)

        def get_elevation_plot(self, files_list, target, station_ids, mission_name, suppress=False, save_dir=None):

           """Reads a list of observation files, extracts time bounds,
           queries JPL Horizons for elevation data, and plots results.

           Parameters:
               files_list (list): List of file paths to process.
               target (str): Target body name for Horizons query (e.g., 'JUICE').
                   station_ids (list): List of station IDs for Horizons query.
                   mission_name (str): Name of the mission.
               suppress (bool): Flag to suppress plot display.
               save_dir (str): Directory to save the plot.
            """
           station_names = [self.Utilities.ID_to_site(site_ID) for site_ID in station_ids if site_ID is not None]
           geodetic_states = [self.Utilities.site_to_geodetic_position(station_name) for station_name in station_names]
           fdets_filename_pattern = r"Fdets\.\w+\d{4}\.\d{2}\.\d{2}(?:-\d{4}-\d{4})?\.(\w+)(?:\.complete)?\.r2i\.txt"
           plt.figure(figsize=(13, 10))
           station_plots = []

           # Store data for saving later
           all_station_data = []

           for i, file_path in enumerate(files_list):
               match = re.search(fdets_filename_pattern, file_path)

               if not match:
                   print(f"Skipping file: {file_path} (No valid station name found)")
                   continue

               receiving_station_name = match.group(1)

               times = []
               times_strings = []
               with open(file_path, 'r') as file:
                   for line in file:
                       if line.startswith("#"):
                           continue

                       parts = line.split()
                       if parts and len(parts) == 5:
                           time_str = parts[0]
                           try:
                               time = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%f")
                               times_strings.append(time_str)
                               times.append(time)
                           except ValueError:
                               print(f"Skipping invalid datetime: {time_str}")

                       elif parts and len(parts) == 6:
                           time_str = parts[1]
                           try:
                               time = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%f")
                               times_strings.append(time_str)
                               times.append(time)
                           except ValueError:
                               print(f"Skipping invalid datetime: {time_str}")

               if not times:
                   print(f"No valid timestamps found in {file_path}")
                   continue

               times = np.array(times)
               min_times_strings = times_strings[np.argmin(times)]
               max_times_strings = times_strings[np.argmax(times)]

               geodetic_state = geodetic_states[i]
               horizons_coord = {'lon': geodetic_state[2],
                                 'lat': geodetic_state[1],
                                 'elevation': geodetic_state[0]/1000}

               obj = Horizons(id=target,
                              location=horizons_coord,
                              epochs={'start':min_times_strings, 'stop':max_times_strings,
                                      'step':'1m'})

               horizons_table = obj.ephemerides()

               # Create the time list with 1-minute intervals
               time_list = []
               current_time = np.min(times)
               while current_time <= np.max(times):
                   time_list.append(current_time)
                   current_time += timedelta(minutes=1)

               # Plot elevation for this station
               plt.plot(time_list, horizons_table['EL'], label=receiving_station_name)
               station_plots.append(receiving_station_name)

               # Store data for this station
               all_station_data.append({
                   'station': receiving_station_name,
                   'time_list': time_list,
                   'elevations': horizons_table['EL'],
                   'times': times
               })

           # Final plot settings
           if all_station_data:
               utc_date = all_station_data[0]['times'][0].date()
               plt.xlabel(f"UTC Time (HH:MM:SS) on {utc_date}")
               plt.ylabel("Elevation (degrees)")
               plt.title(f"Elevation Plot for {target} - Mission {mission_name}")
               plt.legend()
               plt.grid(True)
               plt.xticks(rotation=45)

               # Set x-ticks format to HH:MM:SS and limit number of ticks
               plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
               plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(15))

               # SAVE BEFORE SHOW
               if save_dir:
                   os.makedirs(save_dir, exist_ok=True)
                   try:
                       # Get the x-tick values
                       x_ticks = plt.gca().get_xticks()
                       first_tick = mdates.num2date(x_ticks[0]).strftime('%H%M')
                       last_tick = mdates.num2date(x_ticks[-1]).strftime('%H%M')

                       if len(files_list) == 1:
                           station_data = all_station_data[0]
                           save_path = f"{save_dir}/{station_data['station']}_elevation_plot_{utc_date}-{first_tick}-{last_tick}.png"
                           txt_filename = os.path.join(save_dir, f"{station_data['station']}_elevation_data_{utc_date}-{first_tick}-{last_tick}.txt")

                           with open(txt_filename, "w") as txt_file:
                               txt_file.write("# Time (UTC) | Elevation (degrees)\n")
                               for t, el in zip(station_data['time_list'], station_data['elevations']):
                                   txt_file.write(f"{t.strftime('%Y-%m-%dT%H:%M:%S')} | {el:.2f}\n")

                           plt.savefig(save_path, bbox_inches='tight', dpi=150)
                           print(f"Plot saved to: {save_path}")
                       else:
                           save_path = f"{save_dir}/elevation_plot_{mission_name}_{utc_date}.png"
                           plt.savefig(save_path, bbox_inches='tight', dpi=150)
                           print(f"Plot saved to: {save_path}")
                   except Exception as e:
                       print(f'Error saving plot: {e}')

               # SHOW AFTER SAVE
               if not suppress:
                   plt.show()

               plt.close()

        def read_user_defined_parameters_file(self, filename):
            data = defaultdict(list)  # Dictionary to store parameter data

            with open(filename, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header

                for row in reader:
                    date = datetime.strptime(row[0].strip(), "%Y-%m-%d %H:%M:%S")
                    parameter = row[1].strip()
                    value = float(row[2])
                    data[parameter].append((date, value))  # Store (datetime, value) tuple

            return data
        def get_mean_elevation_from_file(self, filename):
            elevations = []  # List to store elevation values

            with open(filename, 'r') as file:
                reader = csv.reader(file, delimiter='|')
                next(reader)  # Skip header

                for row in reader:
                    if len(row) < 2:  # Ensure there's an elevation value
                        continue
                    elevation = float(row[1].strip())  # Convert to float
                    elevations.append(elevation)

            return np.mean(elevations) if elevations else None  # Avoid error on empty file

        def get_elevations_from_file(self, filename):
            elevations = []  # List to store elevation values

            with open(filename, 'r') as file:
                reader = csv.reader(file, delimiter='|')
                next(reader)  # Skip header

                for row in reader:
                    if len(row) < 2:  # Ensure there's an elevation value
                        continue
                    elevation = float(row[1].strip())  # Convert to float
                    elevations.append(elevation)

            return elevations if elevations else None  # Avoid error on empty file

    class ProcessVexFiles:
        def __init__(self, process_fdets, utilities):
            self.result = 0
            self.Utilities = utilities
            self.ProcessFdets = process_fdets

        def extract_vex_block(self, vex_file_path, block_name):
            """
            Reads the VEX file, extracts the IF block and returns it as a string.
            """
            with open(vex_file_path, 'r') as file:
                vex_content = file.read()

            # Match the IF block (assuming it starts with "$IF" and ends before the next block)
            if_block_match = re.search(f"\${block_name}\s*;.*?(\$|$)", vex_content, re.DOTALL)

            if if_block_match:
                return if_block_match.group(0)
            else:
                return None

        def extract_station_mapping(self, block, block_type):
            """
            Extracts a dictionary mapping station names to corresponding names from the given block.
            Supports both 'IF' and 'FREQ' blocks.
            """
            # Initialize dictionary to hold station to mapping
            station_mapping = defaultdict(list)

            if block_type == 'IF':
                stations_dict = dict()

                # Flag to track the IF block
                in_if_block = False
                current_stations = []

                # Regex pattern to match the chan_def lines
                if_def_pattern = re.compile(r"if_def\s*=\s*(&IF_\w+)\s*:\s*\w+\s*:\s*[A-Za-z]+\s*:\s*[\d\.]+ MHz\s*:\s*[A-Za-z]+\s*;\s*\* PCall off!.*\s*[\d\.]+\s*[\d\.]+\s*\d+cm\s*\d+\s*")
                for line in block.splitlines():
                    line = line.strip()

                    # Check for the start of the $FREQ block
                    if line.startswith("$IF;"):
                        in_if_block = True
                        continue

                    # Check for the end of a definition
                    if line.startswith("enddef;") and in_if_block:
                        current_stations = []
                        continue

                    # Skip lines outside the $FREQ block
                    if not in_if_block:
                        continue

                    # Extract stations from the "stations = ..." line
                    if "stations =" in line:
                        stations_part = line.split("stations =")[1].strip()
                        current_stations = [station.strip() for station in stations_part.split(":")]
                        current_stations[-1] = current_stations[-1].rstrip('\\') # to remove the slash for last station in the vex file
                        continue
                    else:
                        # Handle a new frequency definition
                        if line.startswith("def"):
                            current_def = {"name": line.split()[1], "stations": [], "chan_defs": [], "sample_rate": None}
                            continue

                        # Extract station information from the line
                        if "evn+global" in line:
                            if ":" in line:
                                stations_part = line.split(":")[1].strip()
                                current_stations = [station.strip() for station in stations_part.split(",")]
                                continue

                    # Parse chan_def lines
                    try:
                        match = if_def_pattern.match(line)
                    except:
                        try:
                            match = if_def_pattern.match(line)
                        except:
                            print('It was not possible to retrieve any baseband frequency.')

                    if match:
                        IF_name = match.groups()[0]

                        # Add the channel details to each current station
                        for station in current_stations:
                            stations_dict[station] = {
                                "IF_name": IF_name,
                            }

                return stations_dict


            elif block_type == 'FREQ':
                # Dictionary to store the parsed data
                stations_dict = dict()

                # Flag to track the $FREQ block
                in_freq_block = False
                current_stations = []

                # Regex pattern to match the chan_def lines
                chan_def_pattern = re.compile(r"chan_def\s*=\s*:\s*(\d+(?:\.\d+)? MHz)\s*:\s*(\w+)\s*:\s*(\d+\.\d+ MHz)\s*:\s*(&CH\d+)\s*:\s*(&BBC\d+)\s*:\s*(&\w+);")
                chan_def_pattern_new = re.compile(r"chan_def\s*=\s*:\s*(\d+(?:\.\d+)? MHz)\s*:\s*(\w+)\s*:\s*(\d+\.\d+ MHz)\s*:\s*(&CH\d+)\s*:\s*(&BBC\d+)\s*:\s*(&\w+);\s*\*\s*(\w+)")
                for line in block.splitlines():
                    line = line.strip()

                    # Check for the start of the $FREQ block
                    if line.startswith("$FREQ;"):
                        in_freq_block = True
                        continue

                    # Check for the end of a definition
                    if line.startswith("enddef;") and in_freq_block:
                        current_stations = []
                        continue

                    # Skip lines outside the $FREQ block
                    if not in_freq_block:
                        continue

                    # Extract stations from the "stations = ..." line
                    if "stations =" in line:
                        stations_part = line.split("stations =")[1].strip()
                        current_stations = [station.strip() for station in stations_part.split(":")]
                        current_stations[-1] = current_stations[-1].rstrip('\\') # to remove the slash for last station in the vex file
                        continue

                    else:
                        # Handle a new frequency definition
                        if line.startswith("def"):
                            current_def = {"name": line.split()[1], "stations": [], "chan_defs": [], "sample_rate": None}
                            continue

                        # Extract station information from the line
                        if "evn+global" in line:
                            if ":" in line:
                                stations_part = line.split(":")[1].strip()
                                current_stations = [station.strip() for station in stations_part.split(",")]
                                continue

                    # Parse chan_def lines
                    try:
                        match = chan_def_pattern.match(line)
                    except:
                        try:
                            match = chan_def_pattern_new.match(line)
                        except:
                            print('It was not possible to retrieve any baseband frequency.')

                    if match:
                        frequency, polarization, bandwidth, channel, bbc, cal = match.groups()

                        # Add the channel details to each current station
                        for station in current_stations:
                            stations_dict.setdefault(station, {})[channel] = {
                                "frequency": frequency,
                                "polarization": polarization,
                                "bandwidth": bandwidth,
                                "bbc": bbc,
                                "cal": cal,
                            }

                return stations_dict

            elif block_type == 'BBC':
                # Dictionary to store the parsed data
                stations_dict = dict()

                # Flag to track the $FREQ block
                in_freq_block = False
                current_stations = []

                # Regex pattern to match the chan_def lines
                bbc_assign_pattern = re.compile(r"\*?\s*BBC_assign\s*=\s*(&BBC\d{2})\s*:\s*\d+\s*:\s*(&IF_\w+);")
                #bbc_assing_pattern_new = re.compile(r"chan_def\s*=\s*:\s*(\d+(?:\.\d+)? MHz)\s*:\s*(\w+)\s*:\s*(\d+\.\d+ MHz)\s*:\s*(&CH\d+)\s*:\s*(&BBC\d+)\s*:\s*(&\w+);\s*\*\s*(\w+)")
                for line in block.splitlines():
                    line = line.strip()

                    # Check for the start of the $FREQ block
                    if line.startswith("$BBC;"):
                        in_bbc_block = True
                        continue

                    # Check for the end of a definition
                    if line.startswith("enddef;") and in_freq_block:
                        current_stations = []
                        continue

                    # Skip lines outside the $FREQ block
                    if not in_bbc_block:
                        continue

                    # Extract stations from the "stations = ..." line
                    if "stations =" in line:
                        stations_part = line.split("stations =")[1].strip()
                        current_stations = [station.strip() for station in stations_part.split(":")]
                        current_stations[-1] = current_stations[-1].rstrip('\\') # to remove the slash for last station in the vex file
                        continue

                    else:
                        # Handle a new frequency definition
                        if line.startswith("def"):
                            current_def = {"name": line.split()[1], "stations": [], "chan_defs": [], "sample_rate": None}
                            continue

                        # Extract station information from the line
                        if "evn+global" in line:
                            if ":" in line:
                                stations_part = line.split(":")[1].strip()
                                current_stations = [station.strip() for station in stations_part.split(",")]
                                continue

                    # Parse chan_def lines
                    try:
                        match = bbc_assign_pattern.match(line)
                    except:
                        print('It was not possible to retrieve any baseband frequency.')

                    if match:
                        bbc_name, if_name = match.groups()[0], match.groups()[1]
                        # Add the channel details to each current station
                        for station in current_stations:
                            stations_dict[station] = {
                                "BBC_name": bbc_name,
                                "IF_name": if_name,
                            }

                return stations_dict


        def parse_vex_freq_block(self, file_path):
            """
            Parses the $FREQ block from a vex file and returns a dictionary of stations with their channels.

            Parameters:
                file_path (str): Path to the vex file.

            Returns:
                dict: A dictionary where each station has its sub-dictionary of channels.
            """
            # Dictionary to store the parsed data
            stations_dict = defaultdict(dict)

            # Read the file
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Flag to track the $FREQ block
            in_freq_block = False
            current_stations = []

            # Pattern to match chan_def lines. Two patterns are defined, since the vex file format slightly changed over the years.

            chan_def_pattern = re.compile(r"chan_def\s*=\s*:\s*(\d+(?:\.\d+)? MHz)\s*:\s*(\w+)\s*:\s*(\d+\.\d+ MHz)\s*:\s*(&CH\d+)\s*:\s*(&BBC\d+)\s*:\s*(&\w+);")
            chan_def_pattern_new = re.compile(r"chan_def\s*=\s*:\s*(\d+(?:\.\d+)? MHz)\s*:\s*(\w+)\s*:\s*(\d+\.\d+ MHz)\s*:\s*(&CH\d+)\s*:\s*(&BBC\d+)\s*:\s*(&\w+);\s*\*\s*(\w+)")
            for line in lines:
                line = line.strip()

                # Check for the start of the $FREQ block
                if line.startswith("$FREQ;"):
                    in_freq_block = True
                    continue

                # Check for the end of a definition
                if line.startswith("enddef;") and in_freq_block:
                    current_stations = []
                    continue

                # Skip lines outside the $FREQ block
                if not in_freq_block:
                    continue

                # Extract stations from the "stations = ..." line
                if "stations =" in line:
                    stations_part = line.split("stations =")[1].strip()
                    current_stations = [station.strip() for station in stations_part.split(":")]
                    current_stations[-1] = current_stations[-1].rstrip('\\') # to remove the slash for last station in the vex file
                    continue

                else:
                    # Handle a new frequency definition
                    if line.startswith("def"):
                        current_def = {"name": line.split()[1], "stations": [], "chan_defs": [], "sample_rate": None}
                        continue

                    # Extract station information from the line
                    if "evn+global" in line:
                        if ":" in line:
                            stations_part = line.split(":")[1].strip()
                            current_stations = [station.strip() for station in stations_part.split(",")]
                            continue

                # Parse chan_def lines
                try:
                    match = chan_def_pattern.match(line)
                except:
                    try:
                        match = chan_def_pattern_new.match(line)
                    except:
                        print('It was not possible to retrieve any baseband frequency.')

                if match:
                    frequency, polarization, bandwidth, channel, bbc, cal = match.groups()

                    # Add the channel details to each current station
                    for station in current_stations:
                        stations_dict[station][channel] = {
                            "frequency": frequency,
                            "polarization": polarization,
                            "bandwidth": bandwidth,
                            "bbc": bbc,
                            "cal": cal,
                        }

            return dict(stations_dict)


        def get_baseband_frequency(self, mission_name, experiment_name, fdets_file):

            """
            Extracts the baseband frequency for a given station from a VEX file.

            This function retrieves the frequency and bandwidth information for each
            station from the VEX file, parses the data, and checks if the frequency
            of a given station falls within the specified range. If the condition is met,
            the baseband frequency is returned for that station.

            Parameters:
            mission (str): The name of the mission (e.g., 'JUICE', 'MRO') used to extract spacecraft data.
            experiment_name (str): The name of the experiment to fetch the corresponding VEX file.
            fdets_file (str): The file path of the FDets file, which contains the station information.

            Returns:
            float: The baseband frequency in MHz for the given station if found and within the correct range.
                   If not found, the function will print a message and return None.

            Example:
            baseband_frequency = get_baseband_frequency('JUICE', 'experiment_1', 'fdets_data_file.txt')
            """
            station_fdets = fdets_file.split('/')[-1].split('.')[4]
            file_path = self.Utilities.get_vex_file_path(experiment_name, mission_name)
            result = self.parse_vex_freq_block(file_path)
            mas_x_band = self.Utilities.spacecraft_data[mission_name]['frequency_MHz']

            if station_fdets not in result.keys():
                print(f'Fdets station: {station_fdets} not found in the vex file $FREQ block stations:\n'
                      f'{result.keys()}.\n')
                frequencies_file_name = f'baseband_frequencies/{mission_name}/{experiment_name}_baseband_frequencies.txt'
                print(f'Trying retrieval from baseband frequencies file: {frequencies_file_name}...')
                if os.path.exists(frequencies_file_name):
                    with open(frequencies_file_name, 'r') as f:
                        lines = f.readlines()
                        for line in lines:

                            if station_fdets in line:
                                baseband_frequency = float(line.split()[1])

                else:
                    print(f'Error in get_baseband_frequency: frequencies_file_name: {frequencies_file_name} does not exist.')

            else:
                for station, channels in result.items():
                    for channel in channels:
                        freq_string = channels[channel]['frequency']
                        bandwidth_string = channels[channel]['bandwidth']

                        # Remove " MHz" and convert to float
                        float_frequency = float(freq_string.replace(" MHz", ""))
                        float_bandwidth = float(bandwidth_string.replace(" MHz", ""))
                        if  float_frequency < mas_x_band < float_frequency + float_bandwidth :
                            if station == station_fdets:
                                # set baseband_frequency for station
                                baseband_frequency = float_frequency

            return(baseband_frequency)

        def get_baseband_frequencies_file(self, mission_name, experiment_name):

            """
            Retrieves the baseband frequencies for all stations from a VEX file and writes them into a handy file.

            This function parses the frequency and bandwidth information from a VEX file
            associated with a given experiment and extracts the baseband frequency for
            each station that falls within the specified frequency range based on the
            spacecraft's X-band frequency.

            Parameters:
            mission (str): The name of the mission (e.g., 'JUICE', 'MRO') used to extract spacecraft data.
            experiment_name (str): The name of the experiment to fetch the corresponding VEX file.

            Returns:
            tuple: A tuple containing:
                - baseband_frequency (dict): A dictionary mapping each station to its baseband frequency
                  (in MHz), rounded to 1 decimal place.
                - mas_x_band (float): The X-band frequency (in MHz) for the spacecraft.

            Example:
            baseband_frequencies, x_band = get_baseband_frequencies_file('JUICE', 'experiment_1')
            """

            file_path = self.Utilities.get_vex_file_path(experiment_name, mission_name)
            result = self.parse_vex_freq_block(file_path)
            mas_x_band = self.Utilities.spacecraft_data[mission_name]['frequency_MHz']

            baseband_frequency = {mission_name: {}}
            # Print parsed dictionary
            for station, channels in result.items():
                for channel in channels:
                    freq_string = channels[channel]['frequency']
                    bandwidth_string = channels[channel]['bandwidth']

                    # Remove " MHz" and convert to float
                    float_frequency = float(freq_string.replace(" MHz", ""))
                    float_bandwidth = float(bandwidth_string.replace(" MHz", ""))
                    if  float_frequency < mas_x_band < float_frequency + float_bandwidth :
                        # create baseband_frequency mission dictionary with baseband frequency for each station
                        baseband_frequency[mission_name][station] = np.round(float_frequency,1)

            # Write to file
            os.makedirs(f'baseband_frequencies/{mission_name}', exist_ok = True)
            output_file = f"baseband_frequencies/{mission_name}/{experiment_name}_baseband_frequencies.txt"
            with open(output_file, 'w') as f:
                # Write the header
                f.write(f"# Mission: {mission_name}\n")
                f.write(f"# X-band Observable: {mas_x_band}\n")
                f.write(f"# VEX file name: {file_path}\n")
                f.write("# Station | Baseband Frequency (MHz)\n\n")

                # Write the station and baseband frequencies
                for station, freq in baseband_frequency[mission_name].items():
                    f.write(f"{station}: {freq}\n")

    class FormatFdets:
        def __init__(self, process_fdets, utilities, process_vex_files):
            self.result = 0
            self.Utilities = utilities
            self.ProcessFdets = process_fdets
            self.ProcessVexFiles = process_vex_files

        def assign_missing_baseband_frequencies(self, mission_name, experiment_name):

            mission_name = mission_name.lower()
            experiment_name = experiment_name.lower()

            BBC_block = self.ProcessVexFiles.extract_vex_block(f'./vex_files/{mission_name}/{experiment_name}.vix', 'BBC')
            IF_block = self.ProcessVexFiles.extract_vex_block(f'./vex_files/{mission_name}/{experiment_name}.vix', 'IF')
            FREQ_block = self.ProcessVexFiles.extract_vex_block(f'./vex_files/{mission_name}/{experiment_name}.vix', 'FREQ')
            BBC_dict = self.ProcessVexFiles.extract_station_mapping(BBC_block, 'BBC')
            IF_dict =  self.ProcessVexFiles.extract_station_mapping(IF_block, 'IF')
            FREQ_dict =  self.ProcessVexFiles.extract_station_mapping(FREQ_block, 'FREQ')

            IF_keys, IF_values = IF_dict.keys(), IF_dict.values()
            FREQ_keys, FREQ_values = FREQ_dict.keys(), FREQ_dict.values()

            IF_keys_not_in_FREQ_keys = [key_IF for key_IF in IF_keys if key_IF not in FREQ_keys]

            if len(IF_keys_not_in_FREQ_keys) >= 1:
                for IF_station, IF_inner_dict in IF_dict.items():  # Iterate over IF_dict to get station
                    if IF_station in IF_keys_not_in_FREQ_keys:
                        for IF_name in IF_inner_dict.values():
                            for BBC_station, BBC_inner_dict in BBC_dict.items():
                                if BBC_station in FREQ_keys:
                                    if IF_name not in BBC_inner_dict.values():
                                        missing_station_frequency_flag = False
                                        continue
                                    else:
                                        # Printing the station name by referencing IF_station
                                        print(f'IF_name: {IF_name} for station: {IF_station} was found in BBC dictionary.')
                                        missing_station_frequency_flag = True
                                        self.write_missing_baseband_frequency(mission_name, experiment_name, BBC_station, IF_station)
                                        break

                    else:
                        continue

                    if not missing_station_frequency_flag:
                        print(f'Could not assign a baseband frequency to station: {IF_station} ')

        def write_missing_baseband_frequency(self, mission, experiment, station, missing_station):

            """
            Description:
            Assigns a baseband frequency to a missing station by searching for the frequency of a specified station
            in a baseband frequency file. If the station is found, its frequency is used to append an entry for
            the missing station to the same file. If the specified station is not found, a message is printed indicating
            the failure to assign a frequency.

            Input:
            mission : str
            The name of the mission in lowercase.

            experiment : str
            The name of the experiment in lowercase.

            station : str
            The name of the station whose frequency will be used for assignment.

            missing_station : str
            The name of the missing station that requires a baseband frequency assignment.

            Output:
            None
            The function modifies the baseband frequencies file by appending a line for the missing station
            with the assigned frequency if the specified station is found. If the frequencies file does not exist
            or if the station is not found, it prints appropriate messages to the console.
            """
            find_station_flag = False
            mission = mission.lower()
            experiment = experiment.lower()
            frequencies_file_name = f'baseband_frequencies/{mission}/{experiment}_baseband_frequencies.txt'
            if os.path.exists(frequencies_file_name):
                with open(frequencies_file_name, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if station in line:
                            find_station_flag = True
                            missing_station_frequency = line.split()[1]
                            with open(frequencies_file_name, 'a') as a:
                                a.write(f'{missing_station}: {missing_station_frequency} # assigned, missing from $FREQ block\n')
                                f.close()
                                break
                        else:
                            continue

                    if not find_station_flag:
                        print(f'Could not assign any baseband frequency to missing station: {missing_station}')
                f.close()


            else:
                print(f'The frequencies file: {frequencies_file_name} does not exist.')
