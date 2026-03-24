import numpy as np
import allantools
from collections import Counter
from astropy.time import Time
from ..core.types import FdetsData


def compute_oadev(
    data: FdetsData, tau_min: float | None, tau_max: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Computes Overlapping Allan Deviation for a single observation.

    Args:
        data: The frequency detection data containing Doppler noise and timestamps.
        tau_min: Minimum tau value (integration time) to include in the results.
        tau_max: Maximum tau value (integration time) to include in the results.

    Returns:
        A tuple containing (taus, oadev, errors) as numpy arrays, or (None, None, None)
        if the data is insufficient or invalid. The oadev is normalized by the
        base frequency.
    """
    if len(data.doppler_noise_hz) < 2:
        return None, None, None

    # 1. Calculate Rate (Sampling Frequency)
    # Convert to Julian Date for robust diffing
    t_jd = [Time(t).jd for t in data.utc_datetime]
    diffs = np.diff(t_jd)

    # Find most common time difference
    if len(diffs) == 0:
        return None, None, None

    most_common_diff = Counter(diffs).most_common(1)
    if not most_common_diff or most_common_diff[0][0] == 0:
        return None, None, None

    # Convert days to seconds (86400) -> Rate in Hz
    rate_fdets = 1.0 / (most_common_diff[0][0] * 86400.0)

    # 2. Compute OADEV
    taus, oadev, errors, _ = allantools.oadev(
        data=np.array(data.doppler_noise_hz) / data.base_frequency,
        rate=rate_fdets,
        data_type="freq",
        taus="decade",
    )
    # 3. Filter Taus
    if tau_min is not None or tau_max is not None:
        mask = np.ones_like(taus, dtype=bool)
        if tau_min:
            mask &= taus >= tau_min
        if tau_max:
            mask &= taus <= tau_max

        taus = taus[mask]
        oadev = oadev[mask]
        errors = errors[mask]

    return taus, oadev, errors
