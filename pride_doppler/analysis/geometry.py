import numpy as np
from datetime import datetime
from astroquery.jplhorizons import Horizons
from ..core.constants import ID_TO_SITE, STATION_GEODETIC_POSITIONS

def compute_elevation_data(data: FdetsData, target_name: str) -> tuple[list[datetime], np.array, float] | None:
    """
    Queries JPL Horizons for a specific observation.
    Returns:
        - times (list of datetime): Time axis for plotting
        - elevations (np.array): Elevation axis for plotting
        - mean_elevation (float): Scalar for statistics
    """
    station_id = data.receiving_station_name
    site_name = ID_TO_SITE.get(station_id)

    if not site_name or site_name not in STATION_GEODETIC_POSITIONS:
        print(f"Unknown station geodetics for: {station_id}")
        return None, None, 0.0

    # Get Coordinates
    geo = STATION_GEODETIC_POSITIONS[site_name]
    location = {'lon': geo[2], 'lat': geo[1], 'elevation': geo[0]/1000.0}

    # Define time range
    start = data.utc_datetime[0].strftime("%Y-%m-%d %H:%M")
    stop = data.utc_datetime[-1].strftime("%Y-%m-%d %H:%M")

    obj = Horizons(id=target_name, location=location, epochs={'start': start, 'stop': stop, 'step': '1m'})
    eph = obj.ephemerides()

    # Parse results
    times = [datetime.strptime(str(t), "%Y-%b-%d %H:%M") for t in eph['datetime_str']]
    elevations = np.array(eph['EL'])
    mean_el = np.mean(elevations)

    return times, elevations, mean_el
