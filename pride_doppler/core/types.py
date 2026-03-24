import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel


# --- 1. The Pydantic Model (For API & JSON) ---
class FdetsDataModel(BaseModel):
    """
    Pydantic schema for data validation and API serialization.
    """

    receiving_station_name: str
    utc_datetime: List[datetime]
    utc_date: str
    base_frequency: float
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
    """

    receiving_station_name: str
    utc_datetime: List[datetime]
    utc_date: str
    base_frequency: float
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
            signal_to_noise=self.signal_to_noise.tolist(),
            doppler_noise_hz=self.doppler_noise_hz.tolist(),
            frequency_detection=self.frequency_detection.tolist(),
            first_col_name=self.first_col_name,
            second_col_name=self.second_col_name,
            fifth_col_name=self.fifth_col_name,
        )
