import os
from datetime import datetime, timedelta
from collections import defaultdict
import re
import glob


def split_scan_by_time(
    input_folder: str, fdets_file: str, time_interval_minutes: float, output_folder: str
) -> list[str]:
    """
    Splits a time-series file into segments based on a given time interval.

    Parameters
    ----------
    input_folder : str
        Path to the input folder.
    fdets_file : str
        Name of the input file.
    time_interval_minutes : float
        Time interval for splitting in minutes.
    output_folder : str
        Path to the output folder.

    Returns
    -------
    output_files : list[str]
        List of created output file paths.
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
        raise ValueError(
            "Error in split_scan_by_time: filename does not match expected pattern"
        )

    mission, date, the_rest = match.groups()

    with open(input_file, "r") as file:
        for line in file:
            if line.startswith("#"):
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
                with open(output_filepath, "w") as out_file:
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

        with open(output_filepath, "w") as out_file:
            out_file.writelines(header)
            out_file.writelines(segment)

        output_files.append(output_filepath)

    print(
        f"Created split scan files:\n{[output_file for output_file in output_files]}\n"
    )

    return output_files


def create_complete_scan_from_single_scans(
    files: list[str], output_folder: str
) -> None:
    """
    This function concatenates single scans into a big, "complete" one, for ease of analysis.

    Parameters
    ----------
    files : list[str]
        For instance,
        files = glob.glob('/Users/lgisolfi/Desktop/data_archiving-1.0/dataset/juice/juice_cruise/EC094A/Fdets.*.r2i.txt')
    output_folder : str
        Path to the folder where the concatenated files will be saved.
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
            station_files[station_name].append(
                (file, str(scan_number))
            )  # Store as tuple (file, scan_number)

    # Now, create a complete file for each station
    for station, files in station_files.items():
        # Sort the files based on the scan number (second element of the tuple)
        files.sort(
            key=lambda x: x[1]
        )  # Sorting by scan_number (second element of tuple)

        first_file = files[0][0]  # Get the first file to derive the output filename
        first_scan = files[0][1]
        base_name = os.path.basename(first_file)  # Get the base name of the input file
        # Construct the output filename by removing the scan number
        output_filename = base_name.replace(
            f"{first_scan}.", ""
        )  # Remove the scan number

        # Insert 'complete' before 'r2i' in the output filename
        output_filename = output_filename.replace(".r2i.txt", ".complete.r2i.txt")
        # Create the complete output path
        output_filename = os.path.join(output_folder, output_filename)

        # Open the output file in write mode
        with open(output_filename, "w") as output_file:
            # Flag to check if the header has already been written
            header_written = False

            # Loop over each file for the current station, sorted by scan number
            for file, _ in files:
                # Open the file in read mode
                with open(file, "r") as f:
                    # Read the lines of the file
                    lines = f.readlines()

                    # Skip the header in all files except the first one
                    if not header_written:
                        # Write the first file's header
                        output_file.write(
                            "".join(lines[:5])
                        )  # Assuming the first 5 lines are the header
                        header_written = True

                    # Write the rest of the content, skipping the header lines
                    output_file.write(
                        "".join(lines[5:])
                    )  # Skip the first 5 lines (the header)

        print(f"Created {output_filename}")
