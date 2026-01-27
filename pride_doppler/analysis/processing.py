import numpy as np
from ..core.types import FdetsData

# In pride_doppler/analysis/processing.py

import numpy as np
from ..core.types import FdetsData

def filter_data_zscore(data_list: list[FdetsData], threshold: float | None = 3.5) -> list[FdetsData]:
    """
    Applies Z-score filtering to SNR and Doppler Noise.
    Returns a NEW list of FdetsData objects with outliers removed.
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
                combined_mask &= (np.abs(modified_z) < threshold)
            # =====================================================================

        # Create a new FdetsData object using the final combined mask
        new_entry = FdetsData(
            receiving_station_name=entry.receiving_station_name,
            utc_datetime=[t for i, t in enumerate(entry.utc_datetime) if combined_mask[i]],
            utc_date=entry.utc_date,
            base_frequency=entry.base_frequency,
            signal_to_noise=entry.signal_to_noise[combined_mask],
            doppler_noise_hz=entry.doppler_noise_hz[combined_mask],
            frequency_detection=entry.frequency_detection[combined_mask],
            first_col_name=entry.first_col_name,
            second_col_name=entry.second_col_name,
            fifth_col_name=entry.fifth_col_name
        )

        # Secondary Check (Mean Doppler) - This logic was correct, but let's make it more explicit
        if len(new_entry.doppler_noise_hz) > 0:
            if np.abs(np.mean(new_entry.doppler_noise_hz)) < 0.005:
                filtered_list.append(new_entry)
            else:
                # This part matches the original script's behavior of flagging but still including the data.
                # For a stricter filter, you could 'continue' here instead of appending.
                print(f"  [Filter Warning] Station {entry.receiving_station_name}: Kept despite high mean Doppler noise.")
                filtered_list.append(new_entry)
        else:
            # Append even if empty, to maintain list parallelism with original data
            filtered_list.append(new_entry)

    return filtered_list


def two_step_filter(extracted_parameters_list: list[dict[str, float | str]],
                    keys=('Signal-to-Noise', 'Doppler Noise [Hz]'),
                    threshold: float | None =3.5
    ):
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
