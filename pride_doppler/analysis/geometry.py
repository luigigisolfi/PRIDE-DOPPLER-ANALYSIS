import numpy as np
from datetime import datetime
from astroquery.jplhorizons import Horizons
from pride_doppler.core.constants import ID_TO_SITE, STATION_GEODETIC_POSITIONS
from pride_doppler.core.types import FdetsData


def compute_elevation_data(
    data: FdetsData, target_name: str
) -> tuple[list[datetime], np.array, float] | None:
    """
    Queries JPL Horizons for a specific observation.

    Args:
        data (FdetsData): The observation data containing station and time info.
        target_name (str): The JPL Horizons identifier for the target body.

    Returns:
        tuple[list[datetime], np.array, float] | None: A tuple containing the list of 
            datetimes, an array of elevation angles, and the mean elevation. 
            Returns None if the station is unknown.
    """
    station_id = data.receiving_station_name
    site_name = ID_TO_SITE.get(station_id)

    if not site_name or site_name not in STATION_GEODETIC_POSITIONS:
        print(f"Unknown station geodetics for: {station_id}")
        return None, None, 0.0

    # Get Coordinates
    geo = STATION_GEODETIC_POSITIONS[site_name]
    location = {"lon": geo[2], "lat": geo[1], "elevation": geo[0] / 1000.0}

    # Define time range
    start = data.utc_datetime[0].strftime("%Y-%m-%d %H:%M")
    stop = data.utc_datetime[-1].strftime("%Y-%m-%d %H:%M")

    obj = Horizons(
        id=target_name,
        location=location,
        epochs={"start": start, "stop": stop, "step": "1m"},
    )
    eph = obj.ephemerides()

    # Parse results
    times = [datetime.strptime(str(t), "%Y-%b-%d %H:%M") for t in eph["datetime_str"]]
    elevations = np.array(eph["EL"])
    mean_el = np.mean(elevations)

    return times, elevations, mean_el
