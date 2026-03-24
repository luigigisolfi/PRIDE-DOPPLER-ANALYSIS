"""
pride_doppler/io/vex.py

Parses VEX (.vix) files to extract station and frequency configurations.
Reproduces logic from original ProcessVexFiles class.
"""

import re
import os
from collections import defaultdict


def extract_vex_block(vex_content: str, block_name: str) -> str:
    """
    Finds a specific block (e.g. $FREQ) in the VEX content string.

    Args:
        vex_content (str): The full content of the VEX file.
        block_name (str): The name of the block to extract (without the $).

    Returns:
        str: The content of the block including the header, or None if not found.
    """
    pattern = f"\\${block_name}\\s*;.*?(\\$|$)"
    match = re.search(pattern, vex_content, re.DOTALL)
    return match.group(0) if match else None


def parse_freq_block(block_content: str) -> dict[str, dict[str, dict[str, str]]]:
    """
    Parses the content of a $FREQ block.

    Args:
        block_content (str): The string content of the $FREQ block.

    Returns:
        dict: A nested dictionary mapping station codes to channel definitions.
    """
    stations_dict = defaultdict(dict)
    in_freq_block = False
    current_stations = []

    # Legacy regex patterns from original code
    chan_def_pattern = re.compile(
        r"chan_def\s*=\s*:\s*(\d+(?:\.\d+)? MHz)\s*:\s*(\w+)\s*:\s*(\d+\.\d+ MHz)\s*:\s*(&CH\d+)\s*:\s*(&BBC\d+)\s*:\s*(&\w+);"
    )
    chan_def_pattern_new = re.compile(
        r"chan_def\s*=\s*:\s*(\d+(?:\.\d+)? MHz)\s*:\s*(\w+)\s*:\s*(\d+\.\d+ MHz)\s*:\s*(&CH\d+)\s*:\s*(&BBC\d+)\s*:\s*(&\w+);\s*\*\s*(\w+)"
    )

    lines = block_content.splitlines()

    for line in lines:
        line = line.strip()

        if line.startswith("$FREQ;"):
            in_freq_block = True
            continue

        if line.startswith("enddef;") and in_freq_block:
            current_stations = []
            continue

        if not in_freq_block:
            continue

        # Station definition line
        if "stations =" in line:
            stations_part = line.split("stations =")[1].strip()
            current_stations = [s.strip() for s in stations_part.split(":")]
            # Remove trailing slash from last station (VEX quirk)
            current_stations[-1] = current_stations[-1].rstrip("\\")
            continue

        # Global definition fallback
        if "evn+global" in line and ":" in line:
            stations_part = line.split(":")[1].strip()
            current_stations = [s.strip() for s in stations_part.split(",")]
            continue

        # Channel Definitions
        match = None
        try:
            match = chan_def_pattern.match(line)
        except:
            pass

        if not match:
            try:
                match = chan_def_pattern_new.match(line)
            except:
                pass

        if match:
            # Extract groups (handling the optional 7th group in 'new' pattern gracefully)
            groups = match.groups()
            frequency = groups[0]
            polarization = groups[1]
            bandwidth = groups[2]
            channel = groups[3]
            bbc = groups[4]
            cal = groups[5]

            for station in current_stations:
                stations_dict[station][channel] = {
                    "frequency": frequency,
                    "polarization": polarization,
                    "bandwidth": bandwidth,
                    "bbc": bbc,
                    "cal": cal,
                }

    return dict(stations_dict)


def get_baseband_frequency_from_file(
    vex_file_path: str, station_code: str, target_freq_mhz: float
) -> float:
    """
    Reads a VEX file and finds the baseband frequency for a station
    that covers the target X-band frequency.

    Args:
        vex_file_path (str): Path to the .vix file.
        station_code (str): The station identifier (e.g., 'Mc').
        target_freq_mhz (float): The target frequency in MHz to match.

    Returns:
        float: The baseband frequency (f_start) if found, else None.
    """
    if not os.path.exists(vex_file_path):
        print(f"VEX file not found: {vex_file_path}")
        return None

    with open(vex_file_path, "r") as f:
        content = f.read()

    freq_block = extract_vex_block(content, "FREQ")
    if not freq_block:
        return None

    parsed_data = parse_freq_block(freq_block)

    if station_code not in parsed_data:
        return None

    # Iterate channels to find the one covering the target freq
    channels = parsed_data[station_code]
    for chan_id, info in channels.items():
        freq_str = info["frequency"].replace(" MHz", "")
        bw_str = info["bandwidth"].replace(" MHz", "")

        f_start = float(freq_str)
        bw = float(bw_str)

        if f_start < target_freq_mhz < (f_start + bw):
            return f_start

    return None
