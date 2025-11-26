"""
PRIDE Doppler Data Characterization Script (Refactored)
-----------------------------------------------------
"""

import os
import shutil
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import timezone
from pride_doppler.core.constants import EXPERIMENTS, HORIZONS_TARGETS
from pride_doppler.io.fdets import extract_parameters
from pride_doppler.utils.time import list_yymm
from pride_doppler.utils.images import combine_plots
from pride_doppler.analysis.processing import filter_data_zscore
from pride_doppler.visualization.plots import (
    plot_user_parameters,
    plot_histograms,
    plot_allan_deviation,
    plot_filter_comparison
)
from pride_doppler.analysis.geometry import compute_elevation_data
from pride_doppler.visualization.plots import plot_elevation_profile, get_plot_color
import random
from pride_doppler.core.constants import ANTENNA_DIAMETERS

# --- CONFIGURATION ---
RUN_EXPERIMENTS_STATISTICS_FLAG = True
ALLAN_DEVIATIONS_FLAG = True
BAD_OBSERVATIONS_FLAG = True
ZSCORE_FILTERING_FLAG = True
COMPARE_FILTERS_FLAG = True

# Dates & Paths
start_date = datetime.datetime(2000, 1, 1, tzinfo=timezone.utc)
end_date = datetime.datetime(2024, 12, 31, tzinfo=timezone.utc)
missions_to_analyse = ['min']
root_dir = '/Users/lgisolfi/Desktop/PRIDE_DATA_NEW/'
# --- HELPER: Experiment Lookup ---
from pride_doppler.core.constants import EXPERIMENTS
from datetime import datetime, timezone

def find_experiment(yymmdd_str):
    """
    Check if a yymmdd string falls into any experiment interval defined in constants.py.
    """
    try:
        # Parse 'yymmdd' to datetime (Assumes 2000+)
        # Note: Your original code handled 'yymmddHHMM', but the folder names seem to be 'yymmdd'
        # Adjust format if your input string includes time.
        current_dt = datetime.strptime(yymmdd_str, "%y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return f"Unknown_{yymmdd_str}"

    for exp_name, exp_data in EXPERIMENTS.items():
        # Parse the start/stop strings from constants (e.g., "2023y292d14h00m00s")
        try:
            fmt = "%Yy%jd%Hh%Mm%Ss"
            start = datetime.strptime(exp_data["exper_nominal_start"], fmt).replace(tzinfo=timezone.utc)
            stop = datetime.strptime(exp_data["exper_nominal_stop"], fmt).replace(tzinfo=timezone.utc)

            # Check if current date falls within this experiment
            # Note: We compare the whole day. You might want to refine logic if boundaries are tight.
            if start.date() <= current_dt.date() <= stop.date():
                return exp_name
        except ValueError:
            continue

    return yymmdd_str # Fallback to date if no experiment found

# --- EXECUTION ---

# 1. Scan Folders
print("Scanning for folders...")
yymm_dict = list_yymm(start_date, end_date)
days_list = [d for sublist in yymm_dict.values() for d in sublist]
yymmdd_folders_per_mission = defaultdict(list)

for mission in missions_to_analyse:
    mission_root = os.path.join(root_dir, mission)
    if not os.path.exists(mission_root):
        continue

    for day in days_list:
        # Mission specific logic from original script
        if mission == 'vex' and not day.startswith('1401'): continue

        # Check if day folder exists
        day_folder = f"{mission}_{day}"
        if os.path.exists(os.path.join(mission_root, day_folder)):
            yymmdd_folders_per_mission[mission].append(day)

# 2. Processing Phase
mean_rms_stats = defaultdict(list)


for mission, days in yymmdd_folders_per_mission.items():

    run_bad_obs_check = BAD_OBSERVATIONS_FLAG
    if mission == 'mro': run_bad_obs_check = False

    # Get Horizons ID
    horizons_id = HORIZONS_TARGETS.get(mission, {}).get('target', '-999')

    for day in days:
        folder_name = f"{mission}_{day}"
        base_path = os.path.join(root_dir, mission, folder_name)
        input_dir = os.path.join(base_path, 'input')
        output_dir = os.path.join(base_path, 'output')

        print(f"Processing: {folder_name}")

        if RUN_EXPERIMENTS_STATISTICS_FLAG:
            if os.path.exists(output_dir): shutil.rmtree(output_dir)

            # --- A. Extract Data ---
            raw_data_list = [] # <-- Store raw data here
            if os.path.exists(input_dir):
                for f in sorted(os.listdir(input_dir)):
                    if f.startswith('Fdets') and f.endswith('r2i.txt'):
                        data_obj = extract_parameters(os.path.join(input_dir, f))
                        if data_obj:
                            raw_data_list.append(data_obj)

            if not raw_data_list:
                print(f"No valid fdets found in {input_dir}")
                continue

            filtered_data_list = raw_data_list # Default to raw if not filtering
            if ZSCORE_FILTERING_FLAG:
                print("Applying Z-score filtering")
                # This creates a new list of filtered data objects
                filtered_data_list = filter_data_zscore(raw_data_list)

        print(f"  > Found {len(raw_data_list)} stations to process for this day.")
        for original_data, processed_data in zip(raw_data_list, filtered_data_list):
            station_name = original_data.receiving_station_name
            print(f"    - Processing station: {station_name}")

            # 1. Filter Comparison Plot (if enabled)
            if ZSCORE_FILTERING_FLAG and COMPARE_FILTERS_FLAG:
                plot_filter_comparison(
                    original_data=original_data,
                    filtered_data=processed_data,
                    save_path=os.path.join(output_dir, 'filter_comparison', f"{station_name}_filter_comp.png"),
                    suppress=True
                )

            # 2. Time Series Plot (SNR, Doppler, Fdets)
            # This plot needs to be generated so combine_plots can find it later.
            plot_user_parameters(
                processed_data,
                save_dir=os.path.join(output_dir, 'user_defined_parameters'),
                suppress=True
            )

            # 3. Elevation Logic
            # A. CALCULATION (Analysis Layer)
            times, elevations, mean_el = compute_elevation_data(
                processed_data,
                target_name=horizons_id
            )

            # B. VISUALIZATION (Vis Layer) - Returns nothing
            plot_elevation_profile(
                times,
                elevations,
                station_name=station_name,
                mission_name=mission,
                save_dir=os.path.join(output_dir, 'elevation'),
                suppress=True
            )

            # 4. Collect Statistics for this station
            mean_snr = np.mean(processed_data.signal_to_noise)
            rms_snr = np.std(processed_data.signal_to_noise)
            mean_dopp = np.mean(processed_data.doppler_noise_hz)
            rms_dopp = np.std(processed_data.doppler_noise_hz)

            exp_name = find_experiment(day)
            mean_rms_stats[exp_name].append({
                station_name: {
                    'mean_snr': mean_snr,
                    'rms_snr': rms_snr,
                    'mean_doppler_noise': mean_dopp,
                    'rms_doppler_noise': rms_dopp,
                    'mean_elevation': mean_el
                }
            })

            # 5. Combine Plots (TimeSeries + Elevation) for this station
            ts_name = f"{station_name}_{processed_data.utc_date}_params.png"
            el_name = f"{station_name}_{processed_data.utc_date}_elevation.png"
            ts_path = os.path.join(output_dir, 'user_defined_parameters', ts_name)
            el_path = os.path.join(output_dir, 'elevation', el_name)

            combine_plots(
                [ts_path, el_path],
                output_dir=os.path.join(output_dir, 'combined'),
                output_file_name=f"{station_name}_combined.png"
            )

        # =====================================================================
        # --- D. Aggregate Plots for the Day (After processing all stations) ---
        # =====================================================================
        print("  > Generating aggregate plots for the day...")

        # 1. Aggregate Histograms
        plot_histograms(
            filtered_data_list, # Use the list of all (filtered) stations for this day
            param='doppler',
            save_dir=os.path.join(output_dir, 'statistics'),
            suppress=True
        )

        # 2. Aggregate Allan Deviation Plot
        if ALLAN_DEVIATIONS_FLAG:
            plot_allan_deviation(
                data_list=filtered_data_list, # Use the list of all (filtered) stations
                title=f"Allan Deviation - {folder_name}",
                save_dir=os.path.join(output_dir, 'allan_deviations'),
                suppress=True
            )

        plt.close('all')

# 3. Final Summary
print("Run Complete.")

# =============================================================================
# 4. SUMMARY PLOTTING & STATISTICS PHASE
# =============================================================================
print("\nGenerating Summary Statistics and Plots...")

# Initialize Summary Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
ax1, ax2 = axes

#########################################################################################################
# OPTIONALLY ADD MEAN ELEVATION PLOT (NEEDS IMPROVEMENT BECAUSE "MEAN ELEVATION IS NOT A GOOD Figure of Merit")
#fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
#ax1, ax2, ax3 = axes
########################################################################################################
labels_snr = set()
count = 0
count_bad = 0
scans_to_remove = defaultdict(list)

# Helper for colors (Ported from original logic)
# Configuration for Bad Data Filter
BAD_OBSERVATIONS_MEAN_DOPPLER_FILTER = 0.005 # 5 mHz
skip_elevation = True
# --- A. Main Plotting Loop ---
for experiment_name, station_list in mean_rms_stats.items():

    curr_mission = EXPERIMENTS.get(experiment_name, {}).get('mission_name', 'VEX (JAN 2014)')

    color = get_plot_color(curr_mission, experiment_name)
    label = experiment_name if experiment_name not in labels_snr else None
    labels_snr.add(experiment_name)

    for entry in station_list:
        for station, metrics in entry.items():
            # Retrieve Metrics
            # Note: Convert to dB and mHz to match original units
            mean_snr = 10 * np.log10(metrics['mean_snr']) if metrics['mean_snr'] > 0 else 0
            rms_snr = 10 * np.log10(metrics['rms_snr']) if metrics['rms_snr'] > 0 else 0

            # Doppler is stored in Hz in the new class, convert to mHz here
            mean_doppler_mhz = metrics['mean_doppler_noise'] * 1000
            rms_doppler_mhz = metrics['rms_doppler_noise'] * 1000

            # Elevation (Handle case where it might be missing)
            mean_elevation = metrics.get('mean_elevation', 0)

            # Get Antenna Diameter for marker size
            diam = ANTENNA_DIAMETERS.get(station, 30) # Default 30m if unknown
            count += 1

            # --- Bad Observation Filtering Logic ---
            is_bad = False
            if BAD_OBSERVATIONS_FLAG and curr_mission != 'mro':
                if np.abs(mean_doppler_mhz) > (BAD_OBSERVATIONS_MEAN_DOPPLER_FILTER * 1000):
                    is_bad = True
                    count_bad += 1
                    scans_to_remove[experiment_name].append(station)
                    print(f"  [Bad Scan Flagged] {station} in {experiment_name}: Mean Dopp={mean_doppler_mhz:.3f} mHz")
                    continue
            # --- Plotting ---
            marker_style = {'fmt': 'o', 'markersize': 6, 'alpha': 0.6, 'color': color}

            # Standard Points
            # 1. Station Code vs SNR
            ax1.errorbar(station, mean_snr, label=label, **marker_style)

            # 2. SNR vs RMS Doppler Noise
            ax2.errorbar(mean_snr, rms_doppler_mhz, label=label, **marker_style)
            if is_bad:
                ax2.annotate(station, (mean_snr, rms_doppler_mhz), fontsize=7, alpha=0.7)

            ########################################################################################################
            # OPTIONALLY ADD MEAN ELEVATION PLOT (NEEDS IMPROVEMENT BECAUSE "MEAN ELEVATION IS NOT A GOOD Figure of Merit")
            # 3. Elevation vs SNR
            #ax3.errorbar(mean_elevation, mean_snr, label=label, markersize=3 * diam / 10, fmt='o', alpha=0.6, color=color)
            #ax3.annotate(station, (mean_elevation, mean_snr), fontsize=7, alpha=0.7)
            ########################################################################################################

            # Only label the legend once per experiment
            if label: label = None


if BAD_OBSERVATIONS_FLAG and count > 0:
    print(f"Percentage of Bad Scans: {(count_bad/count)*100:.2f} %")

# Subplot 1
ax1.set_ylabel('SNR [dB]')
ax1.set_xlabel('Station Code')
ax1.grid(True)
# Move legend outside
ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# Subplot 2
ax2.set_xlabel('SNR [dB]')
ax2.set_ylabel('RMS Doppler Noise [mHz]')
ax2.grid(True)
ax2.set_yscale('log')

########################################################################################################
# OPTIONALLY ADD MEAN ELEVATION PLOT (NEEDS IMPROVEMENT BECAUSE "MEAN ELEVATION IS NOT A GOOD Figure of Merit")
# Subplot 3
#ax3.set_xlabel('Elevation [deg]')
#ax3.set_ylabel('SNR [dB]')
#ax3.grid(True)
########################################################################################################

plt.tight_layout(pad=2)
plt.savefig(os.path.join(root_dir, 'final_mission_summary.png'))
print(f"Summary plot saved to {root_dir}")
# plt.show() # Uncomment to view interactive


# =============================================================================
# 5. WEIGHTED AGGREGATES (Table 5 Logic)
# =============================================================================

print("\nComputing Weighted Mission Aggregates...")

mission_aggregates = defaultdict(lambda: {
    'mean_snr': [],
    'mean_doppler_noise': [],
    'rms_doppler_noise': []
})

for experiment_name, station_list in mean_rms_stats.items():
    # Identify Removed Stations
    removed = scans_to_remove.get(experiment_name, [])

    mission_val = EXPERIMENTS.get(experiment_name, {}).get('mission_name', 'VEX (JAN 2014)')

    if isinstance(mission_val, list):
        # If it's a list (like for 'ec064'), default to the first mission.
        curr_mission = mission_val[0]
    else:
        # Otherwise, use the string value as is.
        curr_mission = mission_val
    # Aggregate Containers
    agg_weighted = defaultdict(lambda: {'sum': 0.0, 'weight': 0.0})
    agg_snr_raw = []

    for entry in station_list:
        for station, vals in entry.items():
            # Skip bad stations if Flag is ON
            if station in removed: continue

            # Extract Values
            snr = vals['mean_snr']
            mean_dopp = vals['mean_doppler_noise']
            rms_dopp = vals['rms_doppler_noise']

            # Store raw SNR for unweighted mean
            agg_snr_raw.append(snr)

            # Weighted Means (Weight by SNR)
            if snr > 0:
                agg_weighted['mean_doppler_noise']['sum'] += mean_dopp * snr
                agg_weighted['mean_doppler_noise']['weight'] += snr

                agg_weighted['rms_doppler_noise']['sum'] += rms_dopp * snr
                agg_weighted['rms_doppler_noise']['weight'] += snr

    # Calculate Final Values for this Experiment
    if agg_snr_raw:
        mission_aggregates[curr_mission]['mean_snr'].append(np.mean(agg_snr_raw))

    for k in ['mean_doppler_noise', 'rms_doppler_noise']:
        if agg_weighted[k]['weight'] > 0:
            # Weighted Average -> Convert to mHz
            val = (agg_weighted[k]['sum'] / agg_weighted[k]['weight']) * 1000
            mission_aggregates[curr_mission][k].append(val)

# --- Print Final Table ---
print("-" * 60)
print(f"{'MISSION':<10} | {'Mean SNR (dB)':<15} | {'Mean Dopp (mHz)':<15} | {'RMS Dopp (mHz)':<15}")
print("-" * 60)

for mission, foms in mission_aggregates.items():
    m_snr = 10 * np.log10(np.mean(foms['mean_snr'])) if foms['mean_snr'] else 0
    m_dopp = np.mean(foms['mean_doppler_noise']) if foms['mean_doppler_noise'] else 0
    r_dopp = np.mean(foms['rms_doppler_noise']) if foms['rms_doppler_noise'] else 0

    print(f"{mission.upper():<10} | {m_snr:<15.2f} | {m_dopp:<15.4f} | {r_dopp:<15.4f}")
print("-" * 60)