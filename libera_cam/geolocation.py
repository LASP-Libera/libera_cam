"""
Geolocation Calculations for the Libera Wide-field of View (WFOV) Camera.

This module provides a clean interface for managing SPICE kernels and performing
geolocation calculations for the Libera camera.
"""

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from cloudpathlib import S3Path
from curryer import spicetime
from curryer import spicierpy as sp
from curryer.compute import geometry, spatial
from curryer.compute.constants import SpatialQualityFlags
from curryer.compute.geometry_fields import PixelField
from curryer.spicierpy.ext import spice_error_message
from dask import delayed
from libera_utils.libera_spice.kernel_manager import KernelManager

from libera_cam.constants import (
    AZIMUTH_ENCODER_FRAMES,
    DEFAULT_GEO_CHUNK_SIZE,
    DEFAULT_SPACECRAFT_OBSERVER,
    GROUND_CAL_PIXEL_MAPPING,
    PIXEL_COUNT_X,
    PIXEL_COUNT_Y,
)
from libera_cam.image_parsing.read_l1a_cam_data import kept_pixels_from_source

logger = logging.getLogger(__name__)

# Per-pixel product fill values (L1B_CAM_product_definition.yml)
_GEO_FILL_LAT_LON = np.float32(-999.0)
_GEO_FILL_ALT = np.float32(-9999.0)
_GEO_FILL_ANGLE = np.float32(-999.0)

_SPICE_BODY_PRODUCTION = "LIBERA_WFOV_CAM"
_SPICE_BODY_JPSS_ONLY = "LIBERA_BASE"
_CAMERA_BODIES = (_SPICE_BODY_PRODUCTION, _SPICE_BODY_JPSS_ONLY)

# curryer geometry fields for the L1B product, as GeometryField enum members (interchangeable
# with their string selectors; ``.columns`` gives the output keys). All are spacecraft-level: they
# resolve through the spacecraft ephemeris and body attitude alone, never the camera frame, so
# one query serves the production and ``jpss_only`` modes alike. The camera's boresight and
# per-pixel geometry are a separate concern and are not computed here.
_SPACECRAFT_FIELDS = (
    geometry.GeometryField.SUBSATELLITE,
    geometry.GeometryField.SUBSOLAR,
    geometry.GeometryField.SC_RADIUS,
    geometry.GeometryField.EARTH_SUN_DISTANCE,
    geometry.GeometryField.SC_POSITION_INERTIAL,
    geometry.GeometryField.SC_VELOCITY_INERTIAL,
    geometry.GeometryField.SATELLITE_ATTITUDE,
)

# curryer column -> (product variable, dtype) for the per-frame scalar fields, matching the product
# definition.
_SCALAR_VARIABLES: dict[str, tuple[str, type]] = {
    "subsatellite_latitude": ("Subsatellite_Latitude", np.float32),
    "subsatellite_longitude": ("Subsatellite_Longitude", np.float32),
    "subsatellite_colatitude": ("Subsatellite_Colatitude", np.float32),
    "subsolar_latitude": ("Subsolar_Latitude", np.float32),
    "subsolar_longitude": ("Subsolar_Longitude", np.float32),
    "subsolar_colatitude": ("Subsolar_Colatitude", np.float32),
    "spacecraft_radius": ("Radius_of_Satellite_from_Center_of_Earth", np.float64),
    "attitude_q0": ("Satellite_Attitude_Q0", np.float32),
    "attitude_q1": ("Satellite_Attitude_Q1", np.float32),
    "attitude_q2": ("Satellite_Attitude_Q2", np.float32),
    "attitude_q3": ("Satellite_Attitude_Q3", np.float32),
}
# curryer vector field -> (product variable, dtype); the three columns stack on the ``EUCLIDEAN_DIM``
# dimension.
_VECTOR_VARIABLES: dict[geometry.GeometryField, tuple[str, type]] = {
    geometry.GeometryField.SC_POSITION_INERTIAL: ("Satellite_Position", np.float64),
    geometry.GeometryField.SC_VELOCITY_INERTIAL: ("Satellite_Velocity", np.float64),
}
_EARTH_SUN_DISTANCE_ATTRIBUTE = "Earth_Sun_Distance_AU"
# Motor azimuth variables: the CK at ``camera_time`` (the first integration), and the CK at the
# second integration, ``_INT_TIME_2_OFFSET`` later.
_AZIMUTH_CAMERA_TIME_VARIABLES = ("Azimuth_Motor_Position_SPICE", "Azimuth_Motor_Position_SPICE_Int_Time_1")
_AZIMUTH_INT_TIME_2_VARIABLE = "Azimuth_Motor_Position_SPICE_Int_Time_2"
# TODO[LIBSDC-816]: replace the fixed offset with the measured second-integration timing.
_INT_TIME_2_OFFSET = np.timedelta64(350, "ms")


@dataclass
class GeolocationKernelConfig:
    """
    Configuration for initializing a KernelManager on a worker node.

    This dataclass is pickleable and can be passed to Dask workers.

    Parameters
    ----------
    dynamic_kernel_sources : sequence of str, pathlib.Path, or cloudpathlib.S3Path, optional
        Manifest-ordered paths to dynamic SPICE kernel files (``.bc`` / ``.bsp``, etc.), including S3 when applicable.
        Each entry is materialized through libera_utils ``KernelFileCache`` inside
        :meth:`libera_utils.libera_spice.kernel_manager.KernelManager.load_libera_dynamic_kernels`.
        Use ``None`` when geolocation kernels are not required.
    jpss_only : bool
        True for JPSS-only kernels with no azimuth CK: the per-pixel geolocation intersects
        against ``LIBERA_BASE`` and the motor azimuth is 0 degrees.
    """

    temp_dir_base: str | Path | None = None
    download_naif_url: str = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
    use_test_naif_url: bool = False
    use_high_precision_earth: bool = True
    cache_timeout_days: int = 7
    dynamic_kernel_sources: Sequence[str | Path | S3Path] | None = None
    jpss_only: bool = False


def _kernel_manager_from_config(config: GeolocationKernelConfig) -> KernelManager:
    """Build the ``KernelManager`` a :class:`GeolocationKernelConfig` describes."""
    return KernelManager(
        temp_dir_base=config.temp_dir_base,
        download_naif_url=config.download_naif_url,
        use_test_naif_url=config.use_test_naif_url,
        use_high_precision_earth=config.use_high_precision_earth,
        cache_timeout_days=config.cache_timeout_days,
    )


def _query_geometry(
    observer: str,
    u_gps_times: np.ndarray,
    fields: tuple[geometry.GeometryField, ...],
    **data_kwargs,
) -> pd.DataFrame:
    """
    One curryer ``GeometryData`` query that the kernels must cover, surfacing SPICE failures as readable errors.

    Extra keyword arguments (e.g. ``attitude_frame``) are forwarded to ``GeometryData``.

    Parameters
    ----------
    observer : str
        SPICE body name for the observer.
    u_gps_times : np.ndarray
        Query times in uGPS.
    fields : tuple of GeometryField
        Fields to compute for this observer.

    Returns
    -------
    pd.DataFrame
        The requested fields' columns, indexed by uGPS.

    Raises
    ------
    RuntimeError
        If the curryer SPICE query fails outright (e.g. an unparsable time or a missing
        kernel), with a parsed, user-facing description of the cause.
    ValueError
        If a SPICE input the fields read (spacecraft ephemeris, attitude, Sun ephemeris) is
        NaN at every requested time, i.e. the kernels do not cover the granule. Per-frame
        gaps come back as NaN.
    """
    try:
        return geometry.GeometryData(observer, **data_kwargs).get_geometry(
            u_gps_times, fields=list(fields), require_coverage=True
        )
    except sp.utils.exceptions.SpiceyError as err:
        raise RuntimeError(f"curryer geometry query failed for {observer!r}: {spice_error_message(err)}") from err


def calculate_spacecraft_geometry(
    kernel_manager: KernelManager,
    timestamps: np.ndarray,
) -> pd.DataFrame:
    """
    Compute the spacecraft-level geometry fields via curryer ``GeometryData``.

    curryer's selective-compute registry queries each SPICE input once, vectorized, with
    coverage gaps surfaced as NaN. The spacecraft observer (:data:`DEFAULT_SPACECRAFT_OBSERVER`)
    yields the fields needing only its ephemeris and body attitude: subsatellite and subsolar
    points, satellite radius, Earth-Sun distance, inertial (J2000) position and velocity, and
    the Earth-fixed attitude quaternion.
    None of them resolve through the camera's instrument frame, so the same call serves the
    production and ``jpss_only`` modes.

    Parameters
    ----------
    kernel_manager : KernelManager
        Kernel manager with the spacecraft SPK and CK and the generic/static kernels furnished.
    timestamps : np.ndarray
        Camera frame times on the L1B output time grid, as ``datetime64[ns]``.

    Returns
    -------
    pd.DataFrame
        Indexed by uGPS; the curryer columns of every field in ``_SPACECRAFT_FIELDS``.

    Raises
    ------
    ValueError
        If a SPICE input these fields read (spacecraft ephemeris, attitude, Sun ephemeris)
        covers no frame.
    RuntimeError
        If the curryer SPICE query fails outright, with a parsed, user-facing description of
        the cause.
    """
    kernel_manager.ensure_known_kernels_are_furnished()
    u_gps_times = spicetime.adapt(pd.DatetimeIndex(timestamps), "iso")
    # The attitude quaternion is Earth-fixed (product convention); the inertial fields keep J2000.
    return _query_geometry(
        DEFAULT_SPACECRAFT_OBSERVER, u_gps_times, _SPACECRAFT_FIELDS, attitude_frame=spatial.EARTH_FRAME
    )


def create_placeholder_spacecraft_geometry(n_samples: int) -> pd.DataFrame:
    """
    Placeholder spacecraft geometry for ``use_geo`` false mode.

    Mirrors :func:`calculate_spacecraft_geometry`'s columns filled with NaN, so the dataset
    assembly always reads a geometry DataFrame and never branches on ``None``.

    Parameters
    ----------
    n_samples : int
        Number of camera frames on the L1B output time grid.

    Returns
    -------
    pd.DataFrame
        One float64 column per ``_SPACECRAFT_FIELDS`` output column, all NaN.
    """
    return pd.DataFrame(
        {column: np.full(n_samples, np.nan) for field in _SPACECRAFT_FIELDS for column in field.columns}
    )


def granule_earth_sun_distance(distances: np.ndarray) -> float:
    """
    Reduce the per-frame Earth-Sun distance to the single granule-level attribute.

    The distance moves by well under 1e-4 AU across a granule, so any covered frame
    represents the whole granule. The median is taken over the covered frames only, so a
    SPICE gap at a fixed index (e.g. the midpoint) cannot poison the attribute.

    Parameters
    ----------
    distances : np.ndarray
        Per-frame Earth-Sun distance in AU, NaN where SPICE had no coverage. At least one frame
        is finite: :func:`calculate_spacecraft_geometry` queries with ``require_coverage``, which
        raises when the Sun ephemeris covers no frame.

    Returns
    -------
    float
        Median Earth-Sun distance over the covered frames, in AU.
    """
    return float(np.nanmedian(distances))


def _assign_spacecraft_geometry(ds: xr.Dataset, geometry_data: pd.DataFrame) -> xr.Dataset:
    """Map the curryer geometry columns onto the product variables; coverage gaps stay NaN."""
    for column, (variable, dtype) in _SCALAR_VARIABLES.items():
        ds[variable] = (("camera_time",), geometry_data[column].to_numpy().astype(dtype))
    for field, (variable, dtype) in _VECTOR_VARIABLES.items():
        vector = geometry_data[list(field.columns)].to_numpy()
        ds[variable] = (("camera_time", "EUCLIDEAN_DIM"), vector.astype(dtype))
    return ds


def calculate_azimuth(
    kernel_manager: KernelManager,
    timestamps: np.ndarray,
) -> np.ndarray:
    """
    Azimuth motor encoder angle at each time, from the azimuth CK.

    The angle is the third 1-2-3 Euler angle of the ``LIBERA_BASE_COORD -> LIBERA_AZ_COORD``
    rotation (:data:`AZIMUTH_ENCODER_FRAMES`), wrapped to ``[0, 360)`` -- the convention
    ``libera_rad`` uses for its ``Azimuth``, validated against the ``libera_utils`` tier-0
    kernel tests. It is the corrected encoder reading the CK was built from, relative to the
    instrument base, not a pointing angle in the spacecraft or orbital frame. curryer's
    ``frame_to_frame_euler`` does the per-sample frame transform and factoring.

    A per-sample CK coverage gap comes back as NaN under ``allow_nans`` and stays NaN. A granule
    with no covered frame at all means the azimuth CK or the Libera frame kernel is not
    furnished, or does not cover the granule, and raises.
    The caller furnishes the kernels (:func:`add_frame_geometry_to_dataset` loads the static
    set, which carries the Libera frame kernel, alongside the CK).

    Parameters
    ----------
    kernel_manager : KernelManager
        Kernel manager with the azimuth CK and the Libera frame kernel furnished.
    timestamps : np.ndarray
        Query times, as ``datetime64[ns]``: the camera frame times, or those times shifted to
        the second integration.

    Returns
    -------
    np.ndarray
        Azimuth in degrees, shape ``(N,)``, dtype float32, in ``[0, 360)`` or NaN where the CK
        does not cover the time.

    Raises
    ------
    ValueError
        If no time has a finite azimuth.
    """
    kernel_manager.ensure_known_kernels_are_furnished()

    u_gps_times = spicetime.adapt(pd.DatetimeIndex(timestamps), "iso")
    base_frame, azimuth_frame = AZIMUTH_ENCODER_FRAMES
    euler = spatial.frame_to_frame_euler(base_frame, azimuth_frame, u_gps_times, sequence=(1, 2, 3), allow_nans=True)

    azimuth = np.mod(euler["euler3"].to_numpy(), 360.0)
    if np.isnan(azimuth).all():
        raise ValueError(
            f"The {base_frame} -> {azimuth_frame} transform has no coverage over any of the {azimuth.size} camera "
            "frame(s); check that the azimuth CK in the kernel manifest covers the granule and that the Libera "
            "frame kernel is furnished."
        )
    azimuth = azimuth.astype(np.float32)
    # The float32 cast rounds angles just below 360 up to 360.0, which is written as 0 to stay in [0, 360).
    azimuth[azimuth == np.float32(360.0)] = 0.0
    return azimuth


def create_placeholder_azimuth(n_samples: int) -> np.ndarray:
    """
    Placeholder motor azimuth for ``use_geo`` false mode.

    Parameters
    ----------
    n_samples : int
        Number of camera frames on the L1B output time grid.

    Returns
    -------
    np.ndarray
        Shape ``(N,)``, dtype float32, all NaN.
    """
    return np.full(n_samples, np.nan, dtype=np.float32)


def create_jpss_only_azimuth(n_samples: int) -> np.ndarray:
    """
    Reference motor azimuth for ``jpss_only`` mode (no azimuth CK).

    Returns 0 degrees per operational convention when the motor kernel is unavailable, the
    same zero-azimuth approximation the ``LIBERA_BASE`` per-pixel geolocation makes.

    Parameters
    ----------
    n_samples : int
        Number of camera frames on the L1B output time grid.

    Returns
    -------
    np.ndarray
        Shape ``(N,)``, dtype float32, all zeros.
    """
    return np.zeros(n_samples, dtype=np.float32)


def _assign_azimuth(ds: xr.Dataset, at_camera_time: np.ndarray, at_int_time_2: np.ndarray) -> xr.Dataset:
    """Write the motor azimuth at ``camera_time`` and at the second integration onto the product variables."""
    for variable in _AZIMUTH_CAMERA_TIME_VARIABLES:
        ds[variable] = (("camera_time",), at_camera_time.copy())
    ds[_AZIMUTH_INT_TIME_2_VARIABLE] = (("camera_time",), at_int_time_2.copy())
    return ds


def _warn_uncovered_frames(variable: str, values: np.ndarray) -> None:
    """Log how many camera frames of a per-frame variable SPICE did not cover (non-finite)."""
    uncovered = ~np.isfinite(values)
    if uncovered.ndim > 1:
        uncovered = uncovered.any(axis=tuple(range(1, uncovered.ndim)))
    n_uncovered = int(uncovered.sum())
    if n_uncovered:
        logger.warning(
            f"{variable} has no SPICE coverage at {n_uncovered} of {uncovered.size} camera frames; written as NaN."
        )


def add_frame_geometry_to_dataset(ds: xr.Dataset, config: GeolocationKernelConfig) -> xr.Dataset:
    """
    Compute the per-frame spacecraft geometry and motor azimuth and add them to the dataset.

    These fields are one value per camera frame, so they are computed eagerly on the client
    rather than through Dask ``map_blocks``: the arrays are (N,) or (N, 3), and a map_blocks
    round-trip would cost more than the calculation. One kernel manager furnishes the kernels
    for both and unloads them within this call, so the SPICE pool is left as it was found.

    The spacecraft geometry reads only the spacecraft ephemeris and attitude, so it is the same
    in both SPICE modes. The motor azimuth is the azimuth CK at ``camera_time``, written to
    ``Azimuth_Motor_Position_SPICE`` and ``Azimuth_Motor_Position_SPICE_Int_Time_1``, and at
    ``camera_time`` plus 350 ms, written to ``Azimuth_Motor_Position_SPICE_Int_Time_2``. In
    ``jpss_only`` mode, which has no azimuth CK, all three are 0 degrees
    (:func:`create_jpss_only_azimuth`). A frame the kernels do not cover is NaN, and a warning
    gives the number of such frames for each variable.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing a ``camera_time`` coordinate.
    config : GeolocationKernelConfig
        Configuration for SPICE kernel management. ``config.jpss_only`` selects the 0 degree
        azimuth in place of the azimuth CK.

    Returns
    -------
    xr.Dataset
        The dataset with the sub-point, satellite radius, inertial position and velocity,
        attitude quaternion and three ``Azimuth_Motor_Position_SPICE`` variables added, and the
        granule-level ``Earth_Sun_Distance_AU`` attribute set.

    Raises
    ------
    ValueError
        If ``ds`` has no ``camera_time`` coordinate, ``config`` names no kernel sources, or the
        kernels do not cover the granule at all (see :func:`calculate_spacecraft_geometry` and,
        outside ``jpss_only`` mode, :func:`calculate_azimuth`).
    RuntimeError
        If the curryer SPICE query fails (see :func:`calculate_spacecraft_geometry`).
    """
    if "camera_time" not in ds.coords:
        raise ValueError("Dataset must have 'camera_time' coordinate.")
    if not config.dynamic_kernel_sources:
        raise ValueError("SPICE kernel sources are required to compute the per-frame geometry")

    times = ds.camera_time.values
    with _kernel_manager_from_config(config) as km:
        km.load_libera_dynamic_kernels(
            config.dynamic_kernel_sources,
            needs_naif_kernels=True,
            needs_static_kernels=True,
        )
        geometry_data = calculate_spacecraft_geometry(km, times)
        if config.jpss_only:
            azimuth = azimuth_int_time_2 = create_jpss_only_azimuth(times.size)
        else:
            azimuth = calculate_azimuth(km, times)
            azimuth_int_time_2 = calculate_azimuth(km, times + _INT_TIME_2_OFFSET)

    ds = _assign_spacecraft_geometry(ds, geometry_data)
    (earth_sun_distance,) = geometry.GeometryField.EARTH_SUN_DISTANCE.columns
    ds.attrs[_EARTH_SUN_DISTANCE_ATTRIBUTE] = granule_earth_sun_distance(geometry_data[earth_sun_distance].to_numpy())
    ds = _assign_azimuth(ds, azimuth, azimuth_int_time_2)
    geometry_variables = [variable for variable, _ in (*_SCALAR_VARIABLES.values(), *_VECTOR_VARIABLES.values())]
    for variable in (*geometry_variables, *_AZIMUTH_CAMERA_TIME_VARIABLES, _AZIMUTH_INT_TIME_2_VARIABLE):
        _warn_uncovered_frames(variable, ds[variable].values)
    return ds


def add_placeholder_frame_geometry_to_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Add NaN placeholder spacecraft geometry and motor azimuth to the dataset.

    Used in ground-data mode, where no SPICE kernels are available. The product definition
    declares these variables and the ``Earth_Sun_Distance_AU`` attribute, and the writer
    rejects a dataset that omits a declared field, so they must be present even unfilled.
    """
    n_samples = ds.sizes["camera_time"]
    ds = _assign_spacecraft_geometry(ds, create_placeholder_spacecraft_geometry(n_samples))
    ds.attrs[_EARTH_SUN_DISTANCE_ATTRIBUTE] = np.nan
    ds = _assign_azimuth(ds, create_placeholder_azimuth(n_samples), create_placeholder_azimuth(n_samples))

    logger.info("use_geo is false: using placeholder spacecraft geometry and motor azimuth.")

    return ds


class FrameGeometry(NamedTuple):
    """Per-pixel geometry of one camera frame, ``(Y, X)`` arrays with the product fills applied.

    Angles are degrees under curryer's conventions (azimuths clockwise from geodetic North in
    [0, 360), zeniths from the local geodetic normal in [0, 180], relative azimuth with the Sun at
    180). Pixels that miss the ellipsoid, or whose epoch SPICE could not resolve, carry the product
    ``_FillValue`` and a non-zero ``quality_flags`` word (curryer ``SpatialQualityFlags``).
    """

    latitude: np.ndarray
    longitude: np.ndarray
    altitude: np.ndarray
    solar_zenith: np.ndarray
    solar_azimuth: np.ndarray
    viewing_zenith: np.ndarray
    viewing_azimuth: np.ndarray
    relative_azimuth: np.ndarray
    quality_flags: np.ndarray


# The per-pixel geometry this product needs, named as curryer ``PixelField`` members the way
# libera_rad names its ``GeometryField`` members. curryer computes only the work these require.
_PIXEL_FIELDS: tuple[PixelField, ...] = (
    PixelField.SURFACE_GEODETIC,
    PixelField.SOLAR_ZENITH,
    PixelField.SOLAR_AZIMUTH,
    PixelField.VIEWING_ZENITH,
    PixelField.VIEWING_AZIMUTH,
    PixelField.RELATIVE_AZIMUTH,
    PixelField.QUALITY_FLAGS,
)

# curryer output column -> (product variable, _FillValue, dtype). The keys are curryer's column
# names, which are also the ``FrameGeometry`` field names, so no rename table is needed.
_PIXEL_VARIABLES: dict[str, tuple[str, float, type]] = {
    "latitude": ("Latitude", _GEO_FILL_LAT_LON, np.float32),
    "longitude": ("Longitude", _GEO_FILL_LAT_LON, np.float32),
    "altitude": ("Altitude", _GEO_FILL_ALT, np.float32),
    "solar_zenith": ("Solar_Zenith_Surface", _GEO_FILL_ANGLE, np.float32),
    "solar_azimuth": ("Solar_Azimuth_Surface_WRT_North", _GEO_FILL_ANGLE, np.float32),
    "viewing_zenith": ("Viewing_Zenith_Surface", _GEO_FILL_ANGLE, np.float32),
    "viewing_azimuth": ("Viewing_Azimuth_Surface_WRT_North", _GEO_FILL_ANGLE, np.float32),
    "relative_azimuth": ("Relative_Azimuth_Surface", _GEO_FILL_ANGLE, np.float32),
}
_KM_TO_M = 1000.0
# Bit 15 of Geolocation_Quality_Flag: geolocation was not run for the pixel, because use_geo is false
# or the camera's onboard pixel mask removed it. Bits 0-14 are curryer's SpatialQualityFlags, which
# reach 0x4000, and 0 reads as good geometry, so "not computed" needs a bit of its own.
_GEO_NOT_COMPUTED_FLAG = np.uint16(1 << 15)
# Fields on [0, 360) that can land on exactly 360.0: a float64 value just under a full turn rounds
# up when cast to float32, and curryer's own wrap of a tiny negative angle can return 360.0 itself.
_AZIMUTH_FIELDS = ("solar_azimuth", "viewing_azimuth", "relative_azimuth")
_FULL_TURN = np.float32(360.0)


def geolocate_frame(
    exposure_ugps: np.ndarray,
    exposure_index: np.ndarray | None,
    spice_body: str,
    pointing_vectors: np.ndarray,
    frame_shape: tuple[int, int] = (PIXEL_COUNT_Y, PIXEL_COUNT_X),
    pixel_mask: np.ndarray | None = None,
) -> FrameGeometry:
    """Per-pixel geometry of one camera frame via curryer ``pixel_geometry``, with product fills.

    The per-frame unit of the geolocation path: pure, one frame in, one frame out, no kernel
    management and no Dask. Every pixel ``pixel_mask`` selects that hits the ellipsoid gets
    geodetic latitude and longitude and the solar and viewing zenith and azimuth at that point;
    nothing is interpolated. curryer sees only the selected pixels' vectors, so the cost scales
    with the pixels the camera kept, and the unselected pixels are written as fill.
    ``altitude`` is the ellipsoid reference and is 0 m at every hit (there is no terrain here;
    that is ``Terrain_Corrected_Altitude``, LIBSDC-814). A frame is geolocated at one or more
    exposure epochs: curryer runs once over all ``K`` epochs and each pixel takes
    the epoch ``exposure_index`` names, which is how the two HDR exposures of a frame will be
    handled once their timing is known (LIBSDC-816). Production passes a single epoch until
    then.

    Parameters
    ----------
    exposure_ugps : np.ndarray
        Exposure epochs of the frame in GPS microseconds, shape ``(K,)`` with ``K >= 1``.
    exposure_index : np.ndarray or None
        Integer index into ``exposure_ugps`` for every pixel, shaped ``frame_shape``. Must be
        None when ``K == 1`` and given when ``K > 1``.
    spice_body : str
        NAIF body whose frame the pointing vectors are expressed in: ``LIBERA_WFOV_CAM`` in
        production, ``LIBERA_BASE`` in ``jpss_only`` mode (the camera vectors applied at zero
        azimuth). No other body is accepted. Its kernels must be loaded.
    pointing_vectors : np.ndarray
        Look vectors in that frame, shape ``(Y * X, 3)``, in row-major order of ``frame_shape``:
        the processing layout ``(y, x)`` of ``image_data`` that packaging transposes to the
        product's ``(CAMERA_PIXEL_COUNT_X, CAMERA_PIXEL_COUNT_Y)``. The bundled
        ``wfov_pixel_vectors.npy`` is stored ``(2048, 2048, 3)`` and read in this order; that
        its first axis is the image row is an assumption of the ground-calibration file, not
        something this function can check.
    frame_shape : tuple[int, int], optional
        ``(Y, X)`` of the frame. Default is the WFOV detector.
    pixel_mask : np.ndarray or None, optional
        Boolean array shaped ``frame_shape``, True for the pixels to geolocate: the pixels the
        camera's onboard pixel mask kept. None geolocates every pixel, the full-frame reference
        computation.

    Returns
    -------
    FrameGeometry
        Float32 fields with the product fill where a pixel misses the ellipsoid, its epoch is
        uncovered, or ``pixel_mask`` excludes it; altitude in meters; azimuths wrapped so the
        float32 rounding of a value just under 360 lands on 0, not 360; ``quality_flags`` as
        uint16, 0 where good, bit 15 alone where ``pixel_mask`` excludes the pixel.

    Raises
    ------
    ValueError
        If ``spice_body`` is not one of the two camera bodies, ``exposure_ugps`` is not a
        non-empty 1-D array, ``exposure_index`` is absent for several epochs, present for one,
        not integer, wrongly shaped or out of range, ``pointing_vectors`` is not
        ``(Y * X, 3)``, or ``pixel_mask`` is not a boolean array shaped ``frame_shape``.
    RuntimeError
        If a curryer quality flag reaches bit 15, which ``Geolocation_Quality_Flag`` reserves for
        pixels whose geolocation was not run.

    Notes
    -----
    Memory: for the full detector at one epoch curryer returns about 440 MB and works in about
    1.05 GB of transients (250 bytes per pixel), so a call peaks near 1.5 GB and scales with
    ``K`` and with the selected pixels; the returned ``FrameGeometry`` is 143 MB per frame
    whatever the mask. Size worker chunks from these.
    """
    if spice_body not in _CAMERA_BODIES:
        raise ValueError(f"Unsupported camera body {spice_body!r}; expected one of {', '.join(_CAMERA_BODIES)}.")
    epochs = np.asarray(exposure_ugps)
    if epochs.ndim != 1 or epochs.size == 0:
        raise ValueError(f"exposure_ugps must be a non-empty 1-D array of epochs, got shape {epochs.shape}")
    n_pixels = int(frame_shape[0]) * int(frame_shape[1])
    vectors = np.asarray(pointing_vectors)
    if vectors.shape != (n_pixels, 3):
        raise ValueError(
            f"pointing_vectors must be shape ({n_pixels}, 3) for frame_shape {frame_shape}, got {vectors.shape}"
        )

    if epochs.size == 1:
        if exposure_index is not None:
            raise ValueError("exposure_index must be None when a single exposure epoch is given")
        flat_index = None
    else:
        if exposure_index is None:
            raise ValueError(f"exposure_index is required to select among {epochs.size} exposure epochs")
        index = np.asarray(exposure_index)
        if index.shape != tuple(frame_shape) or not np.issubdtype(index.dtype, np.integer):
            raise ValueError(
                f"exposure_index must be an integer array shaped {tuple(frame_shape)}, got {index.dtype} {index.shape}"
            )
        if index.min() < 0 or index.max() >= epochs.size:
            raise ValueError(
                f"exposure_index values must lie in [0, {epochs.size}), got [{index.min()}, {index.max()}]"
            )
        flat_index = index.ravel()

    if pixel_mask is None:
        selected = np.arange(n_pixels)
    else:
        mask = np.asarray(pixel_mask)
        if mask.shape != tuple(frame_shape) or mask.dtype != np.bool_:
            raise ValueError(
                f"pixel_mask must be a boolean array shaped {tuple(frame_shape)}, got {mask.dtype} {mask.shape}"
            )
        selected = np.flatnonzero(mask)

    fields = {name: np.full(n_pixels, fill, dtype=dtype) for name, (_, fill, dtype) in _PIXEL_VARIABLES.items()}
    flags = np.full(n_pixels, _GEO_NOT_COMPUTED_FLAG, dtype=np.uint16)
    if selected.size:
        # Fill-and-flag on a SPICE gap is this function's contract (a granule with no covered frame
        # is rejected upstream), so the choice is made here rather than left to curryer's default.
        result = spatial.pixel_geometry(
            epochs, spice_body, vectors[selected], fields=list(_PIXEL_FIELDS), allow_nans=True
        )
        selected_index = None if flat_index is None else flat_index[selected]

        def per_pixel(values: np.ndarray) -> np.ndarray:
            """Select each selected pixel's epoch from a (K, n_selected) curryer array."""
            return values[0] if selected_index is None else values[selected_index, np.arange(selected.size)]

        for name, (_, fill, dtype) in _PIXEL_VARIABLES.items():
            values = per_pixel(result[name])
            if name == "altitude":
                values = values * _KM_TO_M  # 0 at every hit today; kept for a non-zero upstream ``alt``
            values = np.where(np.isfinite(values), values, fill).astype(dtype)
            if name in _AZIMUTH_FIELDS:
                values[values == _FULL_TURN] = 0.0
            fields[name][selected] = values

        selected_flags = per_pixel(result["quality_flags"])
        if selected_flags.max() >= _GEO_NOT_COMPUTED_FLAG:
            raise RuntimeError(
                f"curryer quality flag {selected_flags.max():#x} does not fit bits 0-14 of "
                "Geolocation_Quality_Flag; bit 15 means geolocation was not run"
            )
        flags[selected] = selected_flags
    return FrameGeometry(
        quality_flags=flags.reshape(frame_shape),
        **{name: values.reshape(frame_shape) for name, values in fields.items()},
    )


_GEOLOCATION_FLAG_VARIABLE = "Geolocation_Quality_Flag"
# FrameGeometry field -> (product variable, dtype) for every field the geolocation path writes, in
# FrameGeometry field order (the worker returns its blocks in this order).
_FIELD_VARIABLES: dict[str, tuple[str, type]] = {
    name: (
        (_GEOLOCATION_FLAG_VARIABLE, np.uint16)
        if name == "quality_flags"
        else (_PIXEL_VARIABLES[name][0], _PIXEL_VARIABLES[name][2])
    )
    for name in FrameGeometry._fields
}
_BORESIGHT = np.array([[0.0, 0.0, 1.0]])


def _geometry_chunk_size() -> int:
    """Frames per per-pixel geometry task, from ``LIBERA_CAM_GEO_CHUNK_SIZE``.

    Raises
    ------
    ValueError
        If the value is below 1.
    """
    chunk_size = int(os.getenv("LIBERA_CAM_GEO_CHUNK_SIZE", DEFAULT_GEO_CHUNK_SIZE))
    if chunk_size < 1:
        raise ValueError(f"LIBERA_CAM_GEO_CHUNK_SIZE must be >= 1, got {chunk_size}")
    return chunk_size


def _geometry_spice_body(config: GeolocationKernelConfig) -> str:
    """The body whose frame the pixel vectors are intersected in: the camera, or its base in ``jpss_only`` mode."""
    return _SPICE_BODY_JPSS_ONLY if config.jpss_only else _SPICE_BODY_PRODUCTION


def calculate_chunk_geometry(
    exposure_times: np.ndarray,
    exposure_index: np.ndarray | None,
    pixel_mask: np.ndarray,
    config: GeolocationKernelConfig,
) -> tuple[np.ndarray, ...]:
    """
    Worker task: per-pixel geometry of a chunk of camera frames.

    Furnishes the kernels in a fresh ``KernelManager``, converts the epochs to GPS microseconds
    and calls :func:`geolocate_frame` once per frame into preallocated ``(T, Y, X)`` arrays.
    Frames are serial within a task: CSPICE is not thread-safe, and curryer's float64
    intermediates for one full frame already run to about 1.5 GB, so a whole-chunk call would
    multiply that by ``T``. Parallelism is across tasks.

    Parameters
    ----------
    exposure_times : np.ndarray
        ``(T, K)`` exposure epochs as ``datetime64[ns]``, ``K`` per frame. Production passes
        ``K = 1``, the frame's ``camera_time``; the per-exposure epochs arrive with LIBSDC-816.
    exposure_index : np.ndarray or None
        ``(T, Y, X)`` integer index into each frame's ``K`` epochs, or None when ``K == 1``.
    pixel_mask : np.ndarray
        ``(T, Y, X)`` boolean, True for the pixels each frame's camera mask kept; only those
        are geolocated.
    config : GeolocationKernelConfig
        Kernel sources to furnish; ``jpss_only`` selects the ``LIBERA_BASE`` body.

    Returns
    -------
    tuple of np.ndarray
        One ``(T, Y, X)`` array per :class:`FrameGeometry` field, in field order: the float32
        fields with product fills, then the uint16 ``quality_flags``.

    Raises
    ------
    ValueError
        If ``exposure_times`` is not 2-D, or ``exposure_index`` is given or ``pixel_mask`` is
        shaped other than ``(T, Y, X)``. The per-frame epoch, index, mask and vector checks are
        :func:`geolocate_frame`'s.
    """
    times = np.asarray(exposure_times)
    if times.ndim != 2:
        raise ValueError(f"exposure_times must be (T, K) epochs, got shape {times.shape}")
    n_frames, n_epochs = times.shape
    frame_shape = (PIXEL_COUNT_Y, PIXEL_COUNT_X)
    if exposure_index is not None and exposure_index.shape != (n_frames, *frame_shape):
        raise ValueError(f"exposure_index must be shaped {(n_frames, *frame_shape)}, got {exposure_index.shape}")
    if pixel_mask.shape != (n_frames, *frame_shape):
        raise ValueError(f"pixel_mask must be shaped {(n_frames, *frame_shape)}, got {pixel_mask.shape}")

    # Read through mmap on the worker rather than shipped from the client with every task.
    pointing_vectors = np.load(GROUND_CAL_PIXEL_MAPPING, mmap_mode="r").reshape(-1, 3)
    outputs = tuple(np.empty((n_frames, *frame_shape), dtype=dtype) for _, dtype in _FIELD_VARIABLES.values())
    spice_body = _geometry_spice_body(config)
    with _kernel_manager_from_config(config) as km:
        km.load_libera_dynamic_kernels(
            config.dynamic_kernel_sources,
            needs_naif_kernels=True,
            needs_static_kernels=True,
        )
        ugps = np.asarray(spicetime.adapt(pd.DatetimeIndex(times.ravel()), "iso")).reshape(n_frames, n_epochs)
        for i in range(n_frames):
            frame = geolocate_frame(
                ugps[i],
                None if exposure_index is None else exposure_index[i],
                spice_body,
                pointing_vectors,
                frame_shape=frame_shape,
                pixel_mask=pixel_mask[i],
            )
            for output, values in zip(outputs, frame, strict=True):
                output[i] = values
    return outputs


def calculate_chunk_geometry_from_l1a(
    exposure_times: np.ndarray,
    l1a_source: str,
    l1a_rows: list[int],
    config: GeolocationKernelConfig,
) -> tuple[np.ndarray, ...]:
    """
    Worker task: :func:`calculate_chunk_geometry` on the pixels each frame's camera mask kept.

    The mask is rebuilt here from the L1A rows (:func:`kept_pixels_from_source`, about 0.025 s
    per frame) so the task takes no Dask inputs. distributed holds input-free tasks back until a
    worker has a free thread, and each worker's thread is busy computing and compressing the
    blocks already started, so geometry output cannot pile up ahead of them. A geometry task fed
    by the ``image_data`` or mask tasks gets no such throttling, and its 143 MB per frame of
    output piles up in worker memory and spill.

    Parameters
    ----------
    exposure_times : np.ndarray
        ``(T, 1)`` exposure epochs as ``datetime64[ns]``.
    l1a_source : str
        The L1A file, ``l1a_row.attrs["source"]``.
    l1a_rows : list[int]
        The ``T`` frames' L1A rows, values of ``l1a_row``.
    config : GeolocationKernelConfig
        Kernel sources to furnish.

    Returns
    -------
    tuple of np.ndarray
        As :func:`calculate_chunk_geometry`.

    Raises
    ------
    IndexError
        If an L1A row lies outside the file (:func:`kept_pixels_from_source`).
    ValueError
        As :func:`calculate_chunk_geometry`.
    """
    return calculate_chunk_geometry(exposure_times, None, kept_pixels_from_source(l1a_source, l1a_rows), config)


def _require_frame_coverage(config: GeolocationKernelConfig, timestamps: np.ndarray, spice_body: str) -> None:
    """
    Furnish the kernels once on the client and reject a granule of which no frame is covered.

    Loading here materializes every kernel into the libera_utils cache serially before the
    workers start; ``KernelFileCache`` has no lock, so concurrent tasks fetching the same file
    could corrupt it. Coverage is then probed along the body's +Z axis (the camera boresight in
    production) at every frame: an epoch SPICE cannot resolve flags every pixel
    ``CALC_ELLIPS_INSUFF_DATA`` whatever its vector, so one pixel per frame tells. Uncovered
    frames inside a covered granule are the workers' business (fill and flag); a granule with no
    covered frame means the kernels miss it altogether.

    Raises
    ------
    ValueError
        If no frame is covered.
    """
    with _kernel_manager_from_config(config) as km:
        km.load_libera_dynamic_kernels(
            config.dynamic_kernel_sources,
            needs_naif_kernels=True,
            needs_static_kernels=True,
        )
        ugps = np.asarray(spicetime.adapt(pd.DatetimeIndex(timestamps), "iso"))
        probe = spatial.pixel_geometry(ugps, spice_body, _BORESIGHT, fields=[PixelField.QUALITY_FLAGS], allow_nans=True)

    uncovered = (probe["quality_flags"][:, 0] & int(SpatialQualityFlags.CALC_ELLIPS_INSUFF_DATA)) != 0
    if uncovered.all():
        raise ValueError(
            f"SPICE kernels cover none of the {uncovered.size} camera frame(s) for {spice_body!r}; "
            "check that the manifest kernels span the granule."
        )
    if uncovered.any():
        logger.warning(
            "%d of %d camera frame(s) have no SPICE coverage for %r; they are written as fill with the "
            "CALC_ELLIPS_INSUFF_DATA flag.",
            int(uncovered.sum()),
            uncovered.size,
            spice_body,
        )


def add_geolocation_to_dataset(ds: xr.Dataset, config: GeolocationKernelConfig) -> xr.Dataset:
    """
    Lazily compute the per-pixel geometry of every frame and add it to the dataset.

    Every pixel the camera kept, the
    :func:`~libera_cam.image_parsing.read_l1a_cam_data.kept_pixels` of its frame, is geolocated
    at its frame's epoch through :func:`geolocate_frame`; nothing is subsampled or interpolated. The
    pixels the onboard pixel mask removed are written as fill with bit 15 of the flag set, so
    the cost and the stored volume follow the kept pixels. The work is split into tasks of
    ``LIBERA_CAM_GEO_CHUNK_SIZE`` frames along ``camera_time``, independent of the
    decompression chunking; each task rebuilds its frames' masks from the L1A file
    (:func:`calculate_chunk_geometry_from_l1a`), returns one ``(T, Y, X)`` block per field and the blocks
    are concatenated into lazy arrays under the product variable names. The kernels are
    furnished once on the client first, which populates the kernel cache ahead of the workers
    and rejects a granule the kernels do not cover at all.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a ``camera_time`` coordinate (``datetime64[ns]``) and the integer
        ``l1a_row`` on ``camera_time`` with the L1A file in its ``source`` attribute, as
        :func:`~libera_cam.image_parsing.read_l1a_cam_data.read_l1a_cam_data` returns it.
    config : GeolocationKernelConfig
        Kernel sources (required) and ``jpss_only`` (``LIBERA_BASE`` body: the camera vectors
        at zero azimuth, no azimuth CK).

    Returns
    -------
    xr.Dataset
        The dataset with the eight float32 per-pixel fields (``Latitude``, ``Longitude``,
        ``Altitude`` and the five surface angles) and the uint16 ``Geolocation_Quality_Flag``
        added as lazy ``(camera_time, y, x)`` arrays.

    Raises
    ------
    ValueError
        If ``ds`` has no ``camera_time`` coordinate, no integer ``l1a_row`` with a ``source``
        attribute, ``config`` names no kernel sources, ``LIBERA_CAM_GEO_CHUNK_SIZE`` is below 1, or the
        kernels cover none of the frames.

    Notes
    -----
    Each frame is geolocated at one epoch, its ``camera_time``. The worker already takes
    ``(T, K)`` epochs and a per-pixel ``(T, Y, X)`` index, so the two exposures of a frame get
    their own epochs once their timing is known. TODO[LIBSDC-816]
    """
    if "camera_time" not in ds.coords:
        raise ValueError("Dataset must have 'camera_time' coordinate.")
    if (
        "l1a_row" not in ds
        or not np.issubdtype(ds["l1a_row"].dtype, np.integer)
        or not isinstance(ds["l1a_row"].attrs.get("source"), str)
    ):
        raise ValueError("Dataset must have the integer 'l1a_row' with a 'source' attribute, from read_l1a_cam_data.")
    if not config.dynamic_kernel_sources:
        raise ValueError("SPICE kernel sources are required for per-pixel geolocation")
    chunk_size = _geometry_chunk_size()

    timestamps = ds.camera_time.values
    _require_frame_coverage(config, timestamps, _geometry_spice_body(config))

    n_frames = timestamps.size
    exposure_times = timestamps[:, None]
    frame_shape = (PIXEL_COUNT_Y, PIXEL_COUNT_X)
    l1a_source = ds["l1a_row"].attrs["source"]
    l1a_rows = ds["l1a_row"].values
    chunk_geometry = delayed(calculate_chunk_geometry_from_l1a, nout=len(_FIELD_VARIABLES), pure=False)
    blocks: dict[str, list[da.Array]] = {name: [] for name in _FIELD_VARIABLES}
    for start in range(0, n_frames, chunk_size):
        stop = min(start + chunk_size, n_frames)
        outputs = chunk_geometry(exposure_times[start:stop], l1a_source, l1a_rows[start:stop].tolist(), config)
        for (name, (_, dtype)), output in zip(_FIELD_VARIABLES.items(), outputs, strict=True):
            blocks[name].append(da.from_delayed(output, shape=(stop - start, *frame_shape), dtype=dtype))
    for name, (variable, _) in _FIELD_VARIABLES.items():
        ds[variable] = (("camera_time", "y", "x"), da.concatenate(blocks[name], axis=0))

    logger.info("Per-pixel geometry scheduled for %d frame(s) in tasks of %d.", n_frames, chunk_size)
    return ds


def add_jpss_only_geolocation_to_dataset(ds: xr.Dataset, config: GeolocationKernelConfig) -> xr.Dataset:
    """Per-pixel geometry against ``LIBERA_BASE`` (camera vectors at zero azimuth) from the JPSS kernels alone."""
    return add_geolocation_to_dataset(ds, replace(config, jpss_only=True))


def add_placeholder_geolocation_to_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Add the per-pixel geolocation variables as fill values, for ``use_geo`` false mode.

    Ground-data processing has no SPICE kernels, but the writer rejects a dataset missing a
    declared variable, so every field of the geolocation path is written as its fill: the float
    fields as their product ``_FillValue`` and ``Geolocation_Quality_Flag`` with bit 15 set
    (geolocation not run), since 0 would read as good geometry. The arrays are lazy and share
    ``image_data``'s time chunks.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing ``camera_time``, ``y`` and ``x`` dimensions.

    Returns
    -------
    xr.Dataset
        The dataset with the eight float32 fields and the uint16 flag variable added.

    Raises
    ------
    ValueError
        If ``ds`` has no Dask-backed ``image_data`` to take the time chunks from.
    """
    if "image_data" not in ds or not isinstance(ds["image_data"].data, da.Array):
        raise ValueError("Placeholder geolocation takes its time chunks from a Dask-backed 'image_data' variable.")
    shape = (ds.sizes["camera_time"], PIXEL_COUNT_Y, PIXEL_COUNT_X)
    chunks = (ds["image_data"].chunks[0], (PIXEL_COUNT_Y,), (PIXEL_COUNT_X,))
    fills = {name: fill for name, (_, fill, _) in _PIXEL_VARIABLES.items()} | {"quality_flags": _GEO_NOT_COMPUTED_FLAG}
    for name, (variable, dtype) in _FIELD_VARIABLES.items():
        ds[variable] = (("camera_time", "y", "x"), da.full(shape, fills[name], dtype=dtype, chunks=chunks))

    logger.info("use_geo is false: per-pixel geolocation written as fill values.")

    return ds
