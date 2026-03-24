"""
Visualization module for PRIDE Doppler analysis.

This module provides functions for plotting time-series data (SNR, Doppler noise),
elevation profiles via JPL Horizons, Allan deviation analysis, and statistical
distributions of frequency detection results.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
from astroquery.jplhorizons import Horizons
from datetime import datetime
from ..core.constants import ID_TO_SITE, STATION_GEODETIC_POSITIONS
from ..analysis.allan import compute_oadev
from matplotlib.ticker import MaxNLocator
from ..core.types import FdetsData
from matplotlib.ticker import ScalarFormatter
import random
import pandas as pd
import numpy as np
from scipy.stats import norm


def get_plot_color(mission: str, exp_name: str):
    """
    Returns a consistent color mapping for specific missions or experiments.
    
    Args:
        mission: The mission identifier (e.g., 'vex', 'mro').
        exp_name: The experiment identifier (e.g., 'ed045a').
        
    Returns:
        A string representing a matplotlib-compatible color.
    """
    if mission == "vex":
        return "red"
    if mission == "mro":
        return "black"
    if mission == "mex":
        return "magenta"
    if exp_name == "ed045a":
        return "blue"
    if exp_name == "ed045c":
        return "cyan"
    if exp_name == "ed045e":
        return "orangered"
    # Random color generator
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


def plot_user_parameters(
    data: FdetsData, save_dir: str | None = None, suppress: bool = False
) -> None:
    """
    Standard 3-panel time series plot showing SNR, Doppler Noise, and Frequency Detections.

    Args:
        data: FdetsData object containing the time series.
        save_dir: Directory to save the resulting PNG and CSV files.
        suppress: If True, prevents plt.show() from being called.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.subplots_adjust(hspace=0.3)

    # SNR
    axs[0].plot(
        data.utc_datetime,
        data.signal_to_noise,
        "+-",
        color="blue",
        lw=0.5,
        markersize=5,
    )
    axs[0].set_ylabel("SNR")
    axs[0].grid(True)
    axs[0].set_title(f"Station: {data.receiving_station_name} | Date: {data.utc_date}")

    # Doppler
    axs[1].plot(
        data.utc_datetime,
        data.doppler_noise_hz * 1000,
        "+-",
        color="orange",
        lw=0.5,
        markersize=5,
    )
    axs[1].set_ylabel("Doppler Noise [mHz]")
    axs[1].grid(True)

    # Fdets
    axs[2].plot(
        data.utc_datetime, data.frequency_detection, "o", color="black", markersize=2
    )
    axs[2].set_ylabel("Freq Det [Hz]")
    axs[2].set_xlabel("UTC Time")
    axs[2].grid(True)

    # Format Date Axis
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.setp(axs[2].xaxis.get_majorticklabels(), rotation=45)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"{data.receiving_station_name}_{data.utc_date}_params.png"
        plt.savefig(os.path.join(save_dir, fname))

        txt_name = fname.replace(".png", ".csv")
        df_out = pd.DataFrame(
            {
                "UTC_Time": data.utc_datetime,
                "SNR": data.signal_to_noise,
                "Doppler_Noise_mHz": data.doppler_noise_hz,
                "Frequency_Detections_Hz": data.frequency_detection,
            }
        )
        df_out.to_csv(os.path.join(save_dir, txt_name), sep=",", index=False)

    if not suppress:
        plt.show()
    plt.close(fig)


def get_elevation_plot(
    data_list: list[FdetsData],
    target_name: str,
    mission_name: str,
    save_dir: str | None = None,
    suppress: bool = True,
) -> None:
    """
    Queries JPL Horizons and plots elevation for the given stations.

    Args:
        data_list: List of FdetsData objects to determine time ranges and stations.
        target_name: SPICE target ID or name for JPL Horizons.
        mission_name: Name of the mission for the plot title.
        save_dir: Directory to save the plot and text data.
        suppress: If True, prevents plt.show() from being called.
    """
    if not data_list:
        return

    plt.figure(figsize=(12, 7))

    # To store text data for saving
    txt_output_lines = ["# Time (UTC) | Elevation (deg)"]

    for data in data_list:
        station_id = data.receiving_station_name
        site_name = ID_TO_SITE.get(station_id)

        if not site_name or site_name not in STATION_GEODETIC_POSITIONS:
            print(f"Skipping elevation for unknown station: {station_id}")
            continue

        # Get Coordinates [Alt, Lat, Lon]
        geo = STATION_GEODETIC_POSITIONS[site_name]
        # Horizons expects: lon, lat, elevation(km)
        location = {"lon": geo[2], "lat": geo[1], "elevation": geo[0] / 1000.0}

        # Define time range
        start = data.utc_datetime[0].strftime("%Y-%m-%d %H:%M")
        stop = data.utc_datetime[-1].strftime("%Y-%m-%d %H:%M")

        try:
            obj = Horizons(
                id=target_name,
                location=location,
                epochs={"start": start, "stop": stop, "step": "1m"},
            )
            eph = obj.ephemerides()

            # Convert Horizons time to datetime
            times = [
                datetime.strptime(str(t), "%Y-%b-%d %H:%M") for t in eph["datetime_str"]
            ]
            elevations = eph["EL"]

            plt.plot(times, elevations, label=f"{station_id} ({site_name})")

            print(times)
            for t, el in zip(times, elevations):
                txt_output_lines.append(f"{t} | {el:.2f}")

        except Exception as e:
            print(f"Horizons Query Failed for {station_id}: {e}")

    plt.ylabel("Elevation [deg]")
    plt.xlabel("UTC Time")
    plt.title(f"Elevation: {target_name} ({mission_name})")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # Determine filename based on first dataset
        ref = data_list[0]
        fname = f"{ref.receiving_station_name}_{ref.utc_date}_elevation.png"
        plt.savefig(os.path.join(save_dir, fname))

        # Save txt
        txt_name = fname.replace(".png", ".txt")
        with open(os.path.join(save_dir, txt_name), "w") as f:
            f.write("\n".join(txt_output_lines))

    if not suppress:
        plt.show()
    plt.close()


def plot_histograms(
    data_list: list[FdetsData],
    param: str | None = "snr",
    save_dir: str | None = None,
    suppress: bool = True,
):
    """
    Plots distributions for SNR or Doppler Noise using Seaborn.

    Args:
        data_list: List of FdetsData objects.
        param: Either 'snr' or 'doppler'.
        save_dir: Directory to save the plot and CSV data.
        suppress: If True, prevents plt.show() from being called.
    """
    import pandas as pd

    plot_data = []
    for data in data_list:
        arr = data.signal_to_noise if param == "snr" else data.doppler_noise_hz
        for val in arr:
            plot_data.append({"Value": val, "Station": data.receiving_station_name})

    if not plot_data:
        return

    df = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    sns.displot(df, x="Value", hue="Station", kind="kde", fill=True)

    plt.title(f"{param.upper()} Distribution")
    if param == "snr":
        plt.xscale("log")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"all_stations_{param}_dist.png"
        plt.savefig(os.path.join(save_dir, fname))

        txt_name = fname.replace(".png", ".csv")
        df.to_csv(os.path.join(save_dir, txt_name), sep=",", index=False)

    if not suppress:
        plt.show()
    plt.close()


def plot_allan_deviation(
    data_list: list[FdetsData],
    title: str,
    save_dir: str | None = None,
    suppress: bool | None = False,
) -> None:
    """
    Computes and plots the Allan Deviation for a list of FdetsData objects.
    X-axis is formatted as integers (1, 10, 100) instead of scientific notation.

    Args:
        data_list: List of FdetsData objects to analyze.
        title: Title for the plot.
        save_dir: Directory to save the plot and CSV results.
        suppress: If True, prevents plt.show() from being called.
    """
    plt.figure(figsize=(10, 6))

    for data in data_list:
        # Use the dedicated analysis function
        # Ensure compute_oadev is imported in your script
        taus, oadev, err = compute_oadev(data, tau_min=1, tau_max=1000)
        if taus is not None and len(taus) > 0:
            plt.loglog(taus, oadev, ".-", label=data.receiving_station_name)

    # Plot -1/2 slope reference white noise segments
    anchor_taus = [0, 10, 20, 50, 100]  # Anchor points

    # Create evenly-spaced reference levels in log space
    oadev_levels = np.logspace(np.log10(plt.ylim()[0]), np.log10(plt.ylim()[1]), 50)

    for i, tau_start in enumerate(anchor_taus):
        # Determine end tau (next anchor point, or plot limit)
        if i < len(anchor_taus) - 1:
            tau_end = anchor_taus[i + 1]
        else:
            tau_end = plt.xlim()[1]

        if tau_start == 0:
            tau_start = plt.xlim()[0]

        tau_segment = np.array([tau_start, tau_end])

        for oadev_ref in oadev_levels:
            # Each segment: OADEV(tau) = OADEV_ref * (tau/tau_start)^(-0.5)
            ref_segment = oadev_ref * (tau_segment / tau_start) ** -0.5
            plt.loglog(
                tau_segment,
                ref_segment,
                color="gray",
                linestyle="--",
                alpha=0.3,
                linewidth=0.8,
            )

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                # Construct a directory based on the save_path
                txt_name = f"allan_deviation_{data.receiving_station_name}.csv"
                df_out = pd.DataFrame(
                    {"Tau": taus, "Overlapping Allan Deviation": oadev, "Error": err}
                )
                df_out.to_csv(os.path.join(save_dir, txt_name), sep=",", index=False)

    plt.grid(True, which="both", ls="--", alpha=0.2)
    plt.xlabel("Averaging Time (τ) [s]")
    plt.ylabel("Overlapping Allan Deviation")
    plt.legend()
    plt.title(title)

    # --- FORMATTING CHANGE ---
    ax = plt.gca()
    # Force the x-axis to use a standard Scalar Formatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    # Disable scientific notation (e.g., 1e2)
    ax.ticklabel_format(style="plain", axis="x")
    # -------------------------

    if save_dir:
        save_path = os.path.join(save_dir, f"allan_deviation.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  > Allan Deviation plot saved to: {os.path.basename(save_path)}")

    if not suppress:
        plt.show()

    plt.close()


def plot_filter_comparison(
    original_data: FdetsData,
    filtered_data: FdetsData,
    save_path: str = None,
    suppress: bool = False,
) -> None:
    """
    Creates a 2-panel plot comparing original vs. Z-score filtered data for SNR and Doppler noise.
    This version now includes the percentage of data retained in the legend.

    Args:
        original_data (FdetsData): The FdetsData object containing the raw, unfiltered data.
        filtered_data (FdetsData): The FdetsData object after filtering has been applied.
        save_path (str, optional): Path to save the output plot.
        suppress (bool, optional): If True, does not display the plot.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.subplots_adjust(hspace=0.3)
    # =====================================================================
    # --- NEW: Calculate the percentage of retained data ---
    # =====================================================================
    num_original = len(original_data.utc_datetime)
    num_filtered = len(filtered_data.utc_datetime)

    # Avoid division by zero if the original data is empty
    retained_percentage = (
        (num_filtered / num_original) * 100 if num_original > 0 else 0.0
    )
    # =====================================================================

    # --- 1. SNR Comparison Plot ---
    ax1.plot(
        original_data.utc_datetime,
        original_data.signal_to_noise,
        label="Original SNR",
        marker="o",
        linestyle="-",
        color="blue",
        markersize=3,
        linewidth=0.5,
        alpha=0.7,
    )

    # --- Add the retained percentage to the label ---
    ax1.plot(
        filtered_data.utc_datetime,
        filtered_data.signal_to_noise,
        label=f"Filtered SNR (Retained: {retained_percentage:.2f}%)",
        marker="x",
        linestyle="None",
        color="orange",
        markersize=5,
        alpha=0.9,
    )

    ax1.set_title(
        f"Filter Comparison | Station: {original_data.receiving_station_name} | Date: {original_data.utc_date}"
    )
    ax1.set_ylabel("Signal-to-Noise Ratio (SNR)")
    ax1.grid(True)
    ax1.legend()

    # --- 2. Doppler Noise Comparison Plot ---
    # Convert to mHz for plotting
    ax2.plot(
        original_data.utc_datetime,
        original_data.doppler_noise_hz * 1000,
        label="Original Doppler Noise",
        marker="o",
        linestyle="-",
        color="blue",
        markersize=3,
        linewidth=0.5,
        alpha=0.7,
    )

    # --- FIX: Add the percentage to the label ---
    ax2.plot(
        filtered_data.utc_datetime,
        filtered_data.doppler_noise_hz * 1000,
        label=f"Filtered Doppler Noise (Retained: {retained_percentage:.2f}%)",
        marker="x",
        linestyle="None",
        color="orange",
        markersize=5,
        alpha=0.9,
    )

    ax2.set_xlabel(f"UTC Time")
    ax2.set_ylabel("Doppler Noise [mHz]")
    ax2.grid(True)
    ax2.legend()

    # --- Formatting ---
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    locator = MaxNLocator(prune="both", nbins=20)
    ax2.xaxis.set_major_locator(locator)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"  > Filter comparison plot saved to: {os.path.basename(save_path)}")

        base_dir = os.path.dirname(save_path)
        base_name = os.path.basename(save_path).replace(".png", "")

        # Save Original
        df_orig = pd.DataFrame(
            {
                "UTC": original_data.utc_datetime,
                "SNR_Original": original_data.signal_to_noise,
                "Doppler_Original": original_data.doppler_noise_hz,
            }
        )
        df_orig.to_csv(
            os.path.join(base_dir, f"{base_name}_original_data.csv"),
            sep=",",
            index=False,
        )

        # Save Filtered
        df_filt = pd.DataFrame(
            {
                "UTC": filtered_data.utc_datetime,
                "SNR_Filtered": filtered_data.signal_to_noise,
                "Doppler_Filtered": filtered_data.doppler_noise_hz,
            }
        )

        df_filt.to_csv(
            os.path.join(base_dir, f"{base_name}_filtered_data.csv"),
            sep=",",
            index=False,
        )

    if not suppress:
        plt.show()

    plt.close(fig)


def plot_elevation_profile(
    times: list[datetime],
    elevations: list[float],
    station_name: str,
    mission_name: str,
    save_dir: str | None = None,
    suppress: bool | None = True,
) -> None:
    """
    Pure plotting function. No queries, no calculations.

    Args:
        times: List of datetime objects.
        elevations: List of elevation values in degrees.
        station_name: Name of the receiving station.
        mission_name: Name of the mission.
        save_dir: Directory to save the plot and CSV data.
        suppress: If True, prevents plt.show() from being called.
    """
    if times is None or elevations is None:
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(times, elevations, label=station_name)

    ax.set_ylabel("Elevation [deg]")
    ax.set_xlabel("UTC Time")
    ax.set_title(f"Elevation Profile: {mission_name} - {station_name}")
    ax.grid(True)
    ax.legend()

    # Format Time Axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # Assuming times[0] is a datetime object for the filename
        date_str = times[0].strftime("%Y-%m-%d")
        fname = f"{station_name}_{date_str}_elevation.png"
        plt.savefig(os.path.join(save_dir, fname))

        csv_name = fname.replace(".png", ".csv")

        df_out = pd.DataFrame({"UTC_Time": times, "Elevation_deg": elevations})

        df_out.to_csv(os.path.join(save_dir, csv_name), sep=",", index=False)

    if not suppress:
        plt.show()

    plt.close(fig)


def plot_gaussian(filtered_doppler_noise, station_code, mission_name, save_dir=None):
    """
    Fits a Gaussian to the Doppler noise and saves/shows the plot.

    Args:
        filtered_doppler_noise: Numpy array of Doppler noise values in Hz.
        station_code: Identifier for the station.
        mission_name: Name of the mission.
        save_dir: Directory to save the plot.
    """
    import numpy as np
    from scipy.stats import norm

    # Fit data
    mu, std = norm.fit(filtered_doppler_noise * 1000)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(
        filtered_doppler_noise * 1000, bins=60, density=True, alpha=0.6, color="skyblue"
    )

    # Generate PDF curve
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    ax.plot(
        x,
        p,
        "r",
        linewidth=2,
        label=f"Gaussian fit: μ={mu:.3f}, σ={std:.3f}",
        linestyle="--",
    )

    ax.set_title(f"{mission_name.upper()} - Gaussian Fit | Station: {station_code}")
    ax.set_xlabel("Doppler Noise [mHz]")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{station_code}_gaussian_fit.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    > Gaussian fit saved to: {os.path.basename(save_path)}")

    # Close to prevent memory accumulation in loops
    plt.close(fig)
