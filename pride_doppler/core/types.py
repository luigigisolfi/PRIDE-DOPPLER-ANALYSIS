import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel


# --- 1. The Pydantic Model (For API & JSON) ---
class FdetsDataModel(BaseModel):
    """
    Pydantic schema for data validation and API serialization.

    Attributes:
        receiving_station_name (str): Name of the station that received the signal.
        utc_datetime (List[datetime]): List of timestamps for each data point.
        utc_date (str): The date of the observation in string format.
        base_frequency (float): The reference base frequency in Hz.
        signal_to_noise (List[float]): Signal-to-noise ratio values.
        doppler_noise_hz (List[float]): Doppler noise measurements in Hz.
        frequency_detection (List[float]): Detected frequency values in Hz.
        first_col_name (Optional[str]): Original name of the first column in the source file.
        second_col_name (Optional[str]): Original name of the second column in the source file.
        fifth_col_name (Optional[str]): Original name of the fifth column in the source file.
    """

    receiving_station_name: str
    utc_datetime: List[datetime]
    utc_date: str
    base_frequency: float
    fdets_sampling_in_seconds: int
    signal_to_noise: List[float]
    doppler_noise_hz: List[float]
    frequency_detection: List[float]
    first_col_name: Optional[str] = ""
    second_col_name: Optional[str] = ""
    fifth_col_name: Optional[str] = ""

    def to_domain(self) -> "FdetsData":
        """Converts this Pydantic model into the NumPy-optimized Domain object."""
        return FdetsData(
            receiving_station_name=self.receiving_station_name,
            utc_datetime=self.utc_datetime,
            utc_date=self.utc_date,
            base_frequency=self.base_frequency,
            fdets_sampling_in_seconds=self.fdets_sampling_in_seconds,
            signal_to_noise=np.array(self.signal_to_noise),
            doppler_noise_hz=np.array(self.doppler_noise_hz),
            frequency_detection=np.array(self.frequency_detection),
            first_col_name=self.first_col_name,
            second_col_name=self.second_col_name,
            fifth_col_name=self.fifth_col_name,
        )


@dataclass
class FdetsData:
    """
    Internal domain object optimized for scientific calculation.
    Attributes like signal_to_noise are kept as np.ndarrays.

    Attributes:
        receiving_station_name (str): Name of the station that received the signal.
        utc_datetime (List[datetime]): List of timestamps for each data point.
        utc_date (str): The date of the observation in string format.
        base_frequency (float): The reference base frequency in Hz.
        signal_to_noise (np.ndarray): Signal-to-noise ratio values.
        doppler_noise_hz (np.ndarray): Doppler noise measurements in Hz.
        frequency_detection (np.ndarray): Detected frequency values in Hz.
        first_col_name (Optional[str]): Original name of the first column in the source file.
        second_col_name (Optional[str]): Original name of the second column in the source file.
        fifth_col_name (Optional[str]): Original name of the fifth column in the source file.
    """

    receiving_station_name: str
    utc_datetime: List[datetime]
    utc_date: str
    base_frequency: float
    fdets_sampling_in_seconds: int
    signal_to_noise: np.ndarray
    doppler_noise_hz: np.ndarray
    frequency_detection: np.ndarray
    first_col_name: Optional[str] = ""
    second_col_name: Optional[str] = ""
    fifth_col_name: Optional[str] = ""

    def to_model(self) -> FdetsDataModel:
        """Converts this Domain object back to a Pydantic model for API response."""
        return FdetsDataModel(
            receiving_station_name=self.receiving_station_name,
            utc_datetime=self.utc_datetime,
            utc_date=self.utc_date,
            base_frequency=self.base_frequency,
            fdets_sampling_in_seconds=self.fdets_sampling_in_seconds,
            signal_to_noise=self.signal_to_noise.tolist(),
            doppler_noise_hz=self.doppler_noise_hz.tolist(),
            frequency_detection=self.frequency_detection.tolist(),
            first_col_name=self.first_col_name,
            second_col_name=self.second_col_name,
            fifth_col_name=self.fifth_col_name,
        )
