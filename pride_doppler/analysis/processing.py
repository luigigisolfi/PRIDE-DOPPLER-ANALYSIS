"""
This module provides filtering utilities for PRIDE Doppler data, including
Z-score based outlier detection and noise-level validation for FdetsData
objects and parameter dictionaries.
"""

import numpy as np
from ..core.types import FdetsData
from typing import List, Dict, Union, Tuple


def filter_data_zscore(
    data_list: list[FdetsData], threshold: float | None = 3.5
) -> list[FdetsData]:
    """
    Applies Z-score filtering to SNR and Doppler Noise.

    Parameters:
    -----------
    data_list : list[FdetsData]
        A list of FdetsData objects to be filtered.
    threshold : float | None, optional
        The modified Z-score threshold for outlier detection (default is 3.5).

    Returns:
    --------
    list[FdetsData]
        A new list of FdetsData objects with outliers removed.
    """
    filtered_list = []

    for entry in data_list:
        # Start with a mask that keeps everything
        combined_mask = np.ones(len(entry.utc_datetime), dtype=bool)

        # Sequentially apply filters for SNR and Doppler Noise
        for array in [entry.signal_to_noise, entry.doppler_noise_hz]:
            if len(array) == 0:
                continue

            median = np.median(array)
            mad = np.median(np.abs(array - median))

            if mad == 0:
                # If there is no deviation, there are no outliers. Keep all points.
                continue
            else:
                modified_z = 0.6745 * (array - median) / mad
                combined_mask &= np.abs(modified_z) < threshold
            # =====================================================================

        # Create a new FdetsData object using the final combined mask
        new_entry = FdetsData(
            receiving_station_name=entry.receiving_station_name,
            utc_datetime=[
                t for i, t in enumerate(entry.utc_datetime) if combined_mask[i]
            ],
            utc_date=entry.utc_date,
            base_frequency=entry.base_frequency,
            fdets_sampling_in_seconds=entry.fdets_sampling_in_seconds,
            signal_to_noise=entry.signal_to_noise[combined_mask],
            doppler_noise_hz=entry.doppler_noise_hz[combined_mask],
            frequency_detection=entry.frequency_detection[combined_mask],
            first_col_name=entry.first_col_name,
            second_col_name=entry.second_col_name,
            fifth_col_name=entry.fifth_col_name,
        )

        if len(new_entry.doppler_noise_hz) > 0:
            if np.abs(np.mean(new_entry.doppler_noise_hz)) < 0.005:
                filtered_list.append(new_entry)
            else:
                print(
                    f"  [Filter Warning] Station {entry.receiving_station_name}: high mean Doppler noise."
                )
                filtered_list.append(new_entry)
        else:
            filtered_list.append(new_entry)

    return filtered_list
