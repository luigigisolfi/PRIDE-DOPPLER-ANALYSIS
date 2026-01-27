"""
pride_doppler/core/types.py

This module defines the data contracts (Dataclasses) used to pass data
between the IO, Analysis, and Visualization layers.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

@dataclass
class FdetsData:
    """
    Represents the extracted parameters from a single Fdets file.

    Replaces the dictionary returned by `extract_parameters` in the original code.
    Includes a `.to_dict()` method for backward compatibility with legacy plotting functions.
    """
    receiving_station_name: str
    utc_datetime: List[datetime]
    utc_date: str  # YYYY-MM-DD string
    base_frequency: float

    # Data arrays (Stored as numpy arrays for efficient analysis)
    signal_to_noise: np.ndarray
    doppler_noise_hz: np.ndarray
    frequency_detection: np.ndarray

    # Metadata regarding file structure (preserved for logic compatibility)
    first_col_name: str
    second_col_name: str
    fifth_col_name: str

    def __post_init__(self) -> None:
        """
        Validates that data arrays match the length of timestamps.
        Converts lists to numpy arrays if they were passed as lists.
        """
        # Auto-convert lists to numpy arrays if needed
        if isinstance(self.signal_to_noise, list):
            self.signal_to_noise = np.array(self.signal_to_noise)
        if isinstance(self.doppler_noise_hz, list):
            self.doppler_noise_hz = np.array(self.doppler_noise_hz)
        if isinstance(self.frequency_detection, list):
            self.frequency_detection = np.array(self.frequency_detection)

        # Validation
        n_times = len(self.utc_datetime)
        if len(self.signal_to_noise) != n_times:
            raise ValueError(f"Length mismatch: Time ({n_times}) vs SNR ({len(self.signal_to_noise)})")
        if len(self.doppler_noise_hz) != n_times:
            raise ValueError(f"Length mismatch: Time ({n_times}) vs Doppler Noise ({len(self.doppler_noise_hz)})")
        if len(self.frequency_detection) != n_times:
            raise ValueError(f"Length mismatch: Time ({n_times}) vs Freq Detection ({len(self.frequency_detection)})")

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary strictly adhering to the legacy format extracted
        by the original ProcessFdets.extract_parameters method.

        This ensures compatibility with legacy analysis and plotting functions.
        """
        return {
            'receiving_station_name': self.receiving_station_name,
            'utc_datetime': self.utc_datetime,
            'Signal-to-Noise': self.signal_to_noise.tolist(),  # Legacy expects lists
            'Doppler Noise [Hz]': self.doppler_noise_hz.tolist(),
            'base_frequency': self.base_frequency,
            'Freq detection [Hz]': self.frequency_detection.tolist(),
            'first_col_name': self.first_col_name,
            'second_col_name': self.second_col_name,
            'fifth_col_name': self.fifth_col_name,
            'utc_date': self.utc_date
        }