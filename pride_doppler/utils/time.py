"""
pride_doppler/utils/time.py

Handles time format parsing, MJD conversions, and string formatting.
Reproduces logic from original Utilities class.
"""

from datetime import datetime, timedelta, timezone
import re
import math

# MJD Reference: November 17, 1858
MJD_EPOCH = datetime(1858, 11, 17, 0, 0, 0)


def mjd_to_utc(mjd: float) -> datetime.date:
    """
    Converts Modified Julian Date (MJD) to a UTC date object.

    Args:
        mjd (float): The Modified Julian Date.

    Returns:
        datetime.date: The corresponding UTC date.
    """
    return (MJD_EPOCH + timedelta(days=mjd)).date()


def utc_to_mjd(utc_date_str: str) -> float:
    """
    Converts a UTC date string (YYYY.MM.DD) to MJD.

    Args:
        utc_date_str (str): Date string in 'YYYY.MM.DD' format.

    Returns:
        float: The Modified Julian Date.
    """
    try:
        utc_date = datetime.strptime(utc_date_str, "%Y.%m.%d")
    except ValueError:
        # Fallback for different separators if needed, though original specified dot
        utc_date = datetime.strptime(utc_date_str.replace("-", "."), "%Y.%m.%d")

    delta = utc_date - MJD_EPOCH
    return delta.days + (delta.seconds / 86400.0)


def mjd_utc_seconds_to_utc(mjd: float, utc_seconds: float) -> datetime:
    """
    Converts MJD day + seconds into a full UTC datetime.
    Exact reproduction of original Utilities.mjd_utc_seconds_to_utc logic.

    Args:
        mjd (float): Modified Julian Date (integer part usually).
        utc_seconds (float): Seconds passed since the start of the MJD day.

    Returns:
        datetime: Combined UTC datetime object.
    """
    # Convert MJD to JD
    jd = mjd + 2400000.5

    # JD to Gregorian calculation
    jd_days = int(jd)
    jd_fraction = jd - jd_days

    # Reference date for Julian Day 0 (Re-deriving based on original logic)
    # Original code added days to 1858-11-17 based on JD difference
    utc_date = MJD_EPOCH + timedelta(days=jd_days - 2400000.5)

    # Handle fractional day
    utc_time = utc_date + timedelta(days=jd_fraction)

    # Add seconds
    final_utc = utc_time + timedelta(seconds=utc_seconds)
    return final_utc


def format_observation_time(
    observation_date_str: str, time_in_seconds: float
) -> datetime:
    """
    Combines 'YYYY.MM.DD' string and seconds-of-day into a datetime.

    Args:
        observation_date_str (str): Date in 'YYYY.MM.DD' format.
        time_in_seconds (float): Seconds of the day.

    Returns:
        datetime: The combined datetime object.
    """
    obs_date = datetime.strptime(observation_date_str, "%Y.%m.%d")

    # Split seconds
    int_seconds, frac_seconds = divmod(time_in_seconds, 1)

    # Convert to H:M:S
    hours, remainder = divmod(int(int_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    microseconds = round(frac_seconds * 1_000_000)

    time_delta = timedelta(
        hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds
    )
    full_dt = obs_date + time_delta

    # Strip microseconds to match original string format behavior if necessary
    # or return strict object. Returning object is safer.
    return full_dt


def parse_datetime_flexible(t: str) -> datetime:
    """
    Attempts to parse a time string using multiple common formats.

    Args:
        t (str): The time string to parse.

    Returns:
        datetime: Parsed datetime object.

    Raises:
        ValueError: If the string does not match any supported formats.
    """
    formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ]

    t_str = str(t)
    for fmt in formats:
        try:
            return datetime.strptime(t_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Time format not recognized: {t}")


def list_yymm(start_date: datetime, end_date: datetime) -> dict:
    """
    Return a dictionary {yymm: [list of yymmdd]} for each day between start and end.

    Args:
        start_date (datetime): The beginning of the range.
        end_date (datetime): The end of the range.

    Returns:
        dict: Mapping of YYMM strings to lists of YYMMDD strings.
    """
    if start_date > end_date:
        raise ValueError("Start date must be before end date.")

    current = start_date
    yymm_dict = {}

    while current <= end_date:
        # Ensure timezone consistency
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)

        yymmdd = current.strftime("%y%m%d")
        yymm = yymmdd[:4]

        if yymm not in yymm_dict:
            yymm_dict[yymm] = []

        yymm_dict[yymm].append(yymmdd)
        current += timedelta(days=1)

    return yymm_dict
