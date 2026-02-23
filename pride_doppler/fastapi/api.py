import os
import shutil
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from pride_doppler.visualization import plots
import matplotlib
matplotlib.use('Agg')
from pride_doppler.core import constants

# --- Imports ---
from pride_doppler.analysis import allan, geometry, processing
from pride_doppler.io import fdets
from pride_doppler.core.types import FdetsDataModel

# NEW IMPORTS
from pride_doppler.utils import time as time_utils
from pride_doppler.utils import images as image_utils

app = FastAPI(title="Pride Doppler Analysis API", version="1.2.0")


fdets_data_model_example_juice = {
    "receiving_station_name": "Ef",
    "utc_datetime": [
        "2024-03-05T05:43:05",
        "2024-03-05T05:43:15",
        "2024-03-05T05:43:25",
        "2024-03-05T05:43:35",
        "2024-03-05T05:43:45",
    ],
    "utc_date": "2024.03.06",
    "base_frequency": 8432000000,
    "signal_to_noise": [
        4961.871533528503,
        4005.14286150054,
        4700.20676471372,
        4065.376371852537,
        3919.3006915153474,
    ],
    "doppler_noise_hz": [
        -0.00010400124006082478,
        0.0005793954644559562,
        -0.000565530892345123,
        0.0007632124302290322,
        -0.0009541448399659203,
    ],
    "frequency_detection": [
        4045815.116328046,
        4045809.669872214,
        4045804.2085546986,
        4045798.742057931,
        4045793.268412489,
    ],
    "first_col_name": " UTC Time     ",
    "second_col_name": "    Signal-to-Noise     ",
    "fifth_col_name": "   Doppler noise [Hz]  "
}

fdets_data_model_example_mex = {
        "receiving_station_name": "On",
        "utc_datetime": [
            "2024-03-05T10:00:00",
            "2024-03-05T10:01:00",
            "2024-03-05T10:02:00",
            "2024-03-05T10:03:00",
            "2024-03-05T10:04:00"
        ],
        "utc_date": "2024.03.05",
        "base_frequency": 8420000000.0,
        "signal_to_noise": [
            50.5,
            51.2,
            50.8,
            51.0,
            50.9
        ],
        "doppler_noise_hz": [
            0.001,
            0.002,
            -0.001,
            0.001,
            0.0
        ],
        "frequency_detection": [
            1000.0,
            1005.0,
            1010.0,
            1015.0,
            1020.0
        ],
        "first_col_name": "UTC Time",
        "second_col_name": "SNR",
        "fifth_col_name": "Doppler Noise"
    }

fdets_data_model_example_juice_filtered = {
    "receiving_station_name": "Ef",
    "utc_datetime": [
        "2024-03-05T05:43:05",
        "2024-03-05T05:43:15",
        "2024-03-05T05:43:25",
    ],
    "utc_date": "2024.03.06",
    "base_frequency": 8432000000,
    "signal_to_noise": [
        4961.871533528503,
        4005.14286150054,
        4700.20676471372,
    ],
    "doppler_noise_hz": [
        -0.00010400124006082478,
        0.0005793954644559562,
        -0.000565530892345123,
    ],
    "frequency_detection": [
        4045815.116328046,
        4045809.669872214,
        4045804.2085546986,
    ],
    "first_col_name": " UTC Time     ",
    "second_col_name": "    Signal-to-Noise     ",
    "fifth_col_name": "   Doppler noise [Hz]  "
}


@app.get("/meta/stations")
def get_stations():
    """
    Returns a dictionary of valid Station IDs (e.g., 'Ef', 'On') mapped to their full names.
    Useful for populating UI dropdowns.
    """
    return constants.ID_TO_SITE

@app.get("/meta/spacecraft")
def get_spacecraft_data():
    """
    Returns reference data for supported spacecraft (frequencies, antenna codes, etc.).
    """
    return constants.SPACECRAFT_DATA

@app.get("/meta/horizons-targets")
def get_horizons_targets():
    """
    Returns the mapping of internal mission names to JPL Horizons IDs.
    Example: 'mex' -> '-41'
    """
    return constants.HORIZONS_TARGETS

@app.get("/meta/experiments")
def get_experiments():
    """
    Returns the catalogue of defined experiments and their time ranges.
    """
    return constants.EXPERIMENTS

class AllanRequest(BaseModel):
    data: FdetsDataModel = Field(
        ...,
        description = 'JSON',
        examples = [fdets_data_model_example_juice]
    )
    tau_min: float | None = Field(10, description = 'Minimum tau for Allan Deviation calculation', examples = [10, 20])
    tau_max: float | None = Field(1000, description = 'Minimum tau for Allan Deviation calculation', examples = [1000, 2000])

class AllanResponse(BaseModel):
    taus: List[float]
    oadev: List[float]
    errors: List[float]

class ElevationRequest(BaseModel):
    data: FdetsDataModel = Field(..., description = 'JSON', examples = [fdets_data_model_example_mex] )
    target_name: str = Field(..., description = 'Horizons Target Name or Code', examples = ['Mars Express'])

class ElevationResponse(BaseModel):
    times: List[datetime]
    elevations: List[float]
    mean_elevation: float

class FilterRequest(BaseModel):
    data_list: List[FdetsDataModel] = Field(..., description = 'JSON', examples = [[fdets_data_model_example_juice, fdets_data_model_example_mex]])
    threshold: float = 3.5

class PlotParamsRequest(BaseModel):
    data: FdetsDataModel

class PlotAllanRequest(BaseModel):
    data_list: List[FdetsDataModel]
    title: str = "Allan Deviation"

class PlotFilterRequest(BaseModel):
    original_data: FdetsDataModel = Field(..., description = 'JSON', examples = [fdets_data_model_example_juice])
    filtered_data: FdetsDataModel = Field(..., description = 'JSON', examples = [fdets_data_model_example_juice_filtered])

# --- VISUALIZATION ENDPOINTS ---

@app.post("/visualization/parameters")
def plot_parameters(payload: PlotParamsRequest):
    """
    Generates the standard 3-panel plot (SNR, Doppler, Fdets).
    Returns a PNG image.
    """
    domain_data = payload.data.to_domain()

    with tempfile.TemporaryDirectory() as temp_dir:
        # call plot_user_parameters
        # We pass save_dir so it saves the file, and suppress=True to avoid GUI popups
        try:
            plots.plot_user_parameters(
                data=domain_data,
                save_dir=temp_dir,
                suppress=True
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Plotting failed: {str(e)}")

        # Find the generated PNG
        # The function generates a name like "{Station}_{Date}_params.png"
        files = [f for f in os.listdir(temp_dir) if f.endswith(".png")]
        if not files:
            raise HTTPException(status_code=500, detail="Plot generation failed (no file created).")

        file_path = os.path.join(temp_dir, files[0])

        with open(file_path, "rb") as f:
            image_bytes = f.read()

    return Response(content=image_bytes, media_type="image/png")


@app.post("/visualization/allan")
def plot_allan(payload: PlotAllanRequest):
    """
    Generates an Allan Deviation plot for multiple datasets.
    Returns a PNG image.
    """
    domain_list = [d.to_domain() for d in payload.data_list]

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            plots.plot_allan_deviation(
                data_list=domain_list,
                title=payload.title,
                save_dir=temp_dir,
                suppress=True
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Allan Plot failed: {str(e)}")

        # plot_allan_deviation saves as 'allan_deviations.png'
        file_path = os.path.join(temp_dir, "allan_deviations.png")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="Plot file not found.")

        with open(file_path, "rb") as f:
            image_bytes = f.read()

    return Response(content=image_bytes, media_type="image/png")


@app.post("/visualization/filter-comparison")
def plot_filter_comparison(payload: PlotFilterRequest):
    """
    Generates a comparison plot (Original vs Filtered).
    Returns a PNG image.
    """
    orig = payload.original_data.to_domain()
    filt = payload.filtered_data.to_domain()

    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, "comparison.png")

        try:
            plots.plot_filter_comparison(
                original_data=orig,
                filtered_data=filt,
                save_path=save_path,
                suppress=True
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Comparison Plot failed: {str(e)}")

        if not os.path.exists(save_path):
            raise HTTPException(status_code=500, detail="Plot file not found.")

        with open(save_path, "rb") as f:
            image_bytes = f.read()

    return Response(content=image_bytes, media_type="image/png")


@app.get("/utils/time/mjd-to-utc")
def convert_mjd_to_utc(mjd: float):
    """
    Converts Modified Julian Date (MJD) to UTC date.
    Wrapper for pride_doppler.utils.time.mjd_to_utc
    """
    try:
        # mjd_to_utc returns a date object, we convert to string for JSON
        date_obj = time_utils.mjd_to_utc(mjd)
        return {"mjd": mjd, "utc_date": date_obj.isoformat()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/utils/images/combine")
def combine_images(
        files: List[UploadFile] = File(...),
        direction: str = Form("vertical")
):
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Upload at least 2 images to combine.")

    # 1. Use context manager to handle cleanup automatically
    with tempfile.TemporaryDirectory() as temp_dir:
        input_paths = []

        # Save uploaded files
        for idx, file in enumerate(files):
            safe_name = f"input_{idx}_{file.filename}"
            file_path = os.path.join(temp_dir, safe_name)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            input_paths.append(file_path)

        output_filename = "combined_result.png"

        # Process images
        try:
            image_utils.combine_plots(
                image_paths=input_paths,
                output_dir=temp_dir,
                output_file_name=output_filename,
                direction=direction
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

        result_path = os.path.join(temp_dir, output_filename)
        if not os.path.exists(result_path):
            raise HTTPException(status_code=500, detail="Image generation failed silently.")

        # --- FIX STARTS HERE ---
        # Read the file bytes into memory NOW, while the file still exists.
        with open(result_path, "rb") as f:
            image_bytes = f.read()

    # The 'with' block ends here, and temp_dir is deleted.
    # But we still have the 'image_bytes' in memory.

    # Return the bytes directly
    return Response(content=image_bytes, media_type="image/png")

# --- EXISTING ENDPOINTS ---

@app.post("/io/parse", response_model=FdetsDataModel)
def parse_fdets_file(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, file.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            domain_object = fdets.extract_parameters(temp_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Parser error: {str(e)}")

        if domain_object is None:
            raise HTTPException(status_code=400, detail="Parsing failed.")

        return domain_object.to_model()

@app.post("/analysis/oadev", response_model=AllanResponse)
def calculate_allan_deviation(payload: AllanRequest):
    domain_data = payload.data.to_domain()
    taus, oadev, errors = allan.compute_oadev(domain_data, payload.tau_min, payload.tau_max)
    if taus is None: raise HTTPException(status_code=400, detail="Insufficient data.")
    return AllanResponse(taus=taus.tolist(), oadev=oadev.tolist(), errors=errors.tolist())

@app.post("/analysis/elevation", response_model=ElevationResponse)
def compute_elevation_data(payload: ElevationRequest):
    # 1. Validate Station
    # The station name is inside the 'data' object
    station_id = payload.data.receiving_station_name

    # Check if this station exists in your constants
    if station_id not in constants.ID_TO_SITE:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown station ID '{station_id}'. See GET /meta/stations for a list of valid IDs."
        )

    # 2. Validate Geodetics
    # Even if the ID exists, we need coordinates to calculate elevation
    site_name = constants.ID_TO_SITE[station_id]
    if site_name not in constants.STATION_GEODETIC_POSITIONS:
        raise HTTPException(
            status_code=400,
            detail=f"No geodetic coordinates found for '{site_name}'. Cannot compute elevation."
        )

    # 3. Proceed with calculation
    domain_data = payload.data.to_domain()
    times, elevations, mean_el = geometry.compute_elevation_data(domain_data, payload.target_name)

    if times is None:
        raise HTTPException(status_code=404, detail="JPL Horizons query failed.")

    return ElevationResponse(times=times, elevations=elevations.tolist(), mean_elevation=mean_el)

@app.post("/analysis/filter", response_model=List[FdetsDataModel])
def filter_observations(payload: FilterRequest):
    domain_list = [d.to_domain() for d in payload.data_list]
    filtered = processing.filter_data_zscore(domain_list, payload.threshold)
    return [d.to_model() for d in filtered]