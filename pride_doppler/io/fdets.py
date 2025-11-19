"""
pride_doppler/io/fdets.py

Parses Fdets (Frequency Detection) ASCII files.
Connects raw file I/O with the FdetsData dataclass.
"""

import re
from datetime import datetime
import numpy as np

from pride_doppler.core.types import FdetsData
from pride_doppler.utils import time as time_utils


def get_station_name_from_file(filename: str) -> str:
    """Extracts station code from filename using Regex."""
    pattern = r"Fdets\.\w+\d{4}\.\d{2}\.\d{2}(?:-\d{4}-\d{4})?\.(\w+)(?:\.complete)?\.r2i\.txt"
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None

def get_columns_names(filename: str):
    """
    Analyzes line 3 of the file to determine column structure.
    Returns a dict describing the columns.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        if len(lines) < 3:
            return {'number_of_columns': 0}

        # Line 3 (index 2) contains headers
        columns_header = lines[2].strip()
        parts = columns_header.split('|')

        # Handle trailing pipes
        n_columns = len(parts) - 1 if parts[-1] == '' else len(parts)

        col_map = {'number_of_columns': n_columns}

        # Helper to safely strip 'Format:' prefixes
        def clean_col(s):
            return s.split(':', 1)[1] if ':' in s else s

        if n_columns >= 5:
            col_map['first_col_name'] = clean_col(parts[0])
            col_map['second_col_name'] = clean_col(parts[1]) if n_columns >= 2 else ""
            col_map['third_col_name'] = parts[2]
            col_map['fourth_col_name'] = parts[3]
            col_map['fifth_col_name'] = parts[4]

        if n_columns == 6:
            col_map['sixth_col_name'] = parts[5]

        return col_map

def get_observation_date(filename: str, first_col_name: str) -> str:
    """
    Determines the observation date string (YYYY.MM.DD or YYYY-MM-DD).
    Tries header regex first, then data inspection.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

        # Method 1: Header Regex
        header_line = lines[0]
        match = re.search(r'Observation conducted on (\d{4}\.\d{2}\.\d{2})', header_line)
        if match:
            return match.group(1)

        # Method 2: Data Inspection (Line 6, index 5)
        if len(lines) > 5:
            data_line = lines[5].strip().split()

            if first_col_name.strip() == 'UTC Time':
                # format: 2023-04-05T12:00:00...
                utc_part = data_line[0]
                dt = datetime.strptime(utc_part, "%Y-%m-%dT%H:%M:%S.%f")
                return dt.strftime("%Y-%m-%d")

            elif 'Modified Julian Date' in first_col_name or 'Modified JD' in first_col_name:
                mjd_val = float(data_line[0])
                dt_date = time_utils.mjd_to_utc(mjd_val)
                return dt_date.strftime("%Y.%m.%d")

    print(f"Warning: Could not retrieve observation date from {filename}")
    return None

def extract_parameters(filename: str) -> FdetsData:
    """
    Main parsing function. Reads file, parses data, returns FdetsData object.
    """
    station_name = get_station_name_from_file(filename)
    if not station_name:
        print(f"Could not extract station name from {filename}")
        return None

    # 1. Analyze Columns
    col_info = get_columns_names(filename)
    n_cols = col_info.get('number_of_columns', 0)

    # 2. Get Date
    first_col = col_info.get('first_col_name', '')
    obs_date = get_observation_date(filename, first_col)
    if not obs_date:
        return None

    # 3. Get Base Frequency (Line 2)
    base_freq = 0.0
    with open(filename, 'r') as f:
        header_lines = [next(f) for _ in range(4)] # Read first 4 lines
        try:
            # Line 2 (index 1) typically: "Base Frequency: 8400.00 MHz"
            parts = header_lines[1].split(' ')
            base_freq = float(parts[3]) * 1e6 # Convert MHz to Hz
        except:
            print(f"Invalid base frequency in header of {filename}.")
            # Original code had logic to fallback or prompt user.
            # We leave as 0.0 or raise error depending on preference.
            pass

    # 4. Parse Data Rows
    utc_time_list = []
    snr_list = []
    doppler_list = []
    freq_det_list = []

    # Determine logic based on column structure
    idx_time = 0
    idx_snr = 1
    idx_freq = 3
    idx_dopp = 4
    is_mjd = False

    if n_cols == 5:
        if first_col.strip() == 'scan':
            idx_time, idx_snr, idx_freq, idx_dopp = 1, 2, 4, 5
        # Standard 5 col: Time, SNR, SpecMax, Freq, Dopp
    else:
        # 6 columns or MJD start
        idx_time, idx_snr, idx_freq, idx_dopp = 1, 2, 4, 5
        is_mjd = True

    # Pre-calculate MJD epoch day if needed
    mjd_ref_day = 0
    if is_mjd:
        mjd_ref_day = time_utils.utc_to_mjd(obs_date)

    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            if i < 4: continue # Skip header

            parts = line.strip().split()
            if not parts: continue

            try:
                # Extract Raw Values
                t_raw = parts[idx_time]
                snr = float(parts[idx_snr])
                dopp = float(parts[idx_dopp])
                freq = float(parts[idx_freq])

                # Parse Time
                if not is_mjd:
                    # Try ISO format first
                    try:
                        dt = datetime.strptime(t_raw, "%Y-%m-%dT%H:%M:%S.%f")
                    except ValueError:
                        # Fallback to seconds from start of day
                        dt = time_utils.format_observation_time(obs_date, float(t_raw))
                else:
                    # MJD Logic
                    dt = time_utils.mjd_utc_seconds_to_utc(mjd_ref_day, float(t_raw))
                    # Original code ran parse_datetime on the result of this?
                    # No, mjd_utc_seconds_to_utc returns a datetime object.

                utc_time_list.append(dt)
                snr_list.append(snr)
                doppler_list.append(dopp)
                freq_det_list.append(freq)

            except (ValueError, IndexError) as e:
                continue

    # 5. Create Data Object
    return FdetsData(
        receiving_station_name=station_name,
        utc_datetime=utc_time_list,
        utc_date=obs_date,
        base_frequency=base_freq,
        signal_to_noise=np.array(snr_list),
        doppler_noise_hz=np.array(doppler_list),
        frequency_detection=np.array(freq_det_list),
        first_col_name=col_info.get('first_col_name', ''),
        second_col_name=col_info.get('second_col_name', ''),
        fifth_col_name=col_info.get('fifth_col_name', '')
    )