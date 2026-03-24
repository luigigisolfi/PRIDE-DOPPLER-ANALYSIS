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


def two_step_filter(
    extracted_parameters_list: List[Dict[str, Union[float, str]]],
    keys: Tuple[str, ...] = ("Signal-to-Noise", "Doppler Noise [Hz]"),
    threshold: float | None = 3.5,
) -> List[Dict[str, Union[float, str]]]:
    """
    Applies a two-step filtering process to a list of parameter dictionaries.

    Parameters:
    -----------
    extracted_parameters_list : list[dict[str, float | str]]
        A list of dictionaries containing extracted parameters.
    keys : tuple, optional
        The dictionary keys to apply Z-score filtering on (default is SNR and Doppler Noise).
    threshold : float | None, optional
        The modified Z-score threshold for outlier detection (default is 3.5).

    Returns:
    --------
    list[dict[str, float | str]]
        The filtered list of parameter dictionaries.
    """
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
            if np.abs(np.mean(doppler_noise)) < 0.005:
                filtered_list.append(entry)
            else:
                station_name = entry.get("receiving_station_name", [])
                print(f"Station {station_name}: Bad observation detected.")
                filtered_list.append(entry)
        else:
            print(
                "No doppler noise entry found. Maybe check the corresponding dictionary key name."
            )
    return filtered_list
