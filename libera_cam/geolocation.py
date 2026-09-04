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
    SPACECRAFT_OBSERVERS,
)

logger = logging.getLogger(__name__)

# Product fill values (L1B_CAM_product_definition.yml)
_GEO_FILL_LAT_LON = np.float32(-999.0)
_GEO_FILL_ALT = np.float32(-9999.0)
_FILL_VALUE = -999.0  # product default: angles, quaternion components, velocity
_FILL_DISTANCE = -9999.0  # radius and position
_FILL_EARTH_SUN_DISTANCE = -999.0

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

# curryer column -> (product variable, _FillValue, dtype) for the per-frame scalar fields, matching
# the product definition.
_SCALAR_VARIABLES: dict[str, tuple[str, float, type]] = {
    "subsatellite_latitude": ("Subsatellite_Latitude", _FILL_VALUE, np.float32),
    "subsatellite_longitude": ("Subsatellite_Longitude", _FILL_VALUE, np.float32),
    "subsatellite_colatitude": ("Subsatellite_Colatitude", _FILL_VALUE, np.float32),
    "subsolar_latitude": ("Subsolar_Latitude", _FILL_VALUE, np.float32),
    "subsolar_longitude": ("Subsolar_Longitude", _FILL_VALUE, np.float32),
    "subsolar_colatitude": ("Subsolar_Colatitude", _FILL_VALUE, np.float32),
    "spacecraft_radius": ("Radius_of_Satellite_from_Center_of_Earth", _FILL_DISTANCE, np.float64),
    "attitude_q0": ("Satellite_Attitude_Q0", _FILL_VALUE, np.float32),
    "attitude_q1": ("Satellite_Attitude_Q1", _FILL_VALUE, np.float32),
    "attitude_q2": ("Satellite_Attitude_Q2", _FILL_VALUE, np.float32),
    "attitude_q3": ("Satellite_Attitude_Q3", _FILL_VALUE, np.float32),
}
# curryer vector field -> (product variable, _FillValue, dtype); the three columns stack on the
# ``EUCLIDEAN_DIM`` dimension.
_VECTOR_VARIABLES: dict[geometry.GeometryField, tuple[str, float, type]] = {
    geometry.GeometryField.SC_POSITION_INERTIAL: ("Satellite_Position", _FILL_DISTANCE, np.float64),
    geometry.GeometryField.SC_VELOCITY_INERTIAL: ("Satellite_Velocity", _FILL_VALUE, np.float64),
}
_EARTH_SUN_DISTANCE_ATTRIBUTE = "Earth_Sun_Distance_AU"
_AZIMUTH_VARIABLE = "Azimuth"


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


def _apply_fill(values: np.ndarray, fill: float, dtype: type) -> np.ndarray:
    """Replace SPICE coverage gaps (NaN) with the product ``_FillValue``."""
    return np.where(np.isfinite(values), values, fill).astype(dtype)


def _validate_observer(observer: str, allowed: tuple[str, ...], role: str) -> None:
    """Reject an observer frame that is not a known Libera frame for this role.

    A typo'd frame already fails inside SPICE, but a *valid* frame used in the wrong role
    (the WFOV camera as the spacecraft, say) would silently produce correct-looking geometry
    for the wrong body. This turns that into an error at the call site.

    Parameters
    ----------
    observer : str
        Requested SPICE frame name.
    allowed : tuple[str, ...]
        Frame names valid for this role.
    role : str
        Human-readable role name, used in the error message.

    Raises
    ------
    ValueError
        If ``observer`` is not in ``allowed``.
    """
    if observer not in allowed:
        raise ValueError(
            f"Unsupported {role} observer {observer!r}; expected one of {', '.join(allowed)}. "
            "Geometry for other frames is not supported by the L1B product definition."
        )


def _query_geometry(
    observer: str,
    u_gps_times: np.ndarray,
    fields: tuple[geometry.GeometryField, ...],
    require_coverage: bool,
    coverage_fields: tuple[geometry.GeometryField, ...] = (),
    **data_kwargs,
) -> pd.DataFrame:
    """
    One curryer ``GeometryData`` query, surfacing SPICE failures as readable errors.

    Extra keyword arguments (e.g. ``attitude_frame``) are forwarded to ``GeometryData``.

    Parameters
    ----------
    observer : str
        SPICE body name for the observer.
    u_gps_times : np.ndarray
        Query times in uGPS.
    fields : tuple of GeometryField
        Fields to compute for this observer.
    require_coverage : bool
        If True, raise when the coverage fields are entirely NaN -- the kernels do not cover
        the granule at all (a misconfiguration).
    coverage_fields : tuple of GeometryField, optional
        Fields whose all-NaN state signals no coverage; defaults to every requested field. The
        spacecraft observer restricts this to an ephemeris-derived field, since the subsolar
        point and Earth-Sun distance are computed from the Sun ephemeris alone and stay finite
        even when the spacecraft kernels miss the granule.

    Returns
    -------
    pd.DataFrame
        The requested fields' columns, indexed by uGPS.

    Raises
    ------
    RuntimeError
        If the curryer SPICE query fails outright (e.g. an unparsable time or a missing
        kernel), or -- when ``require_coverage`` -- if it returns no coverage at all. Both
        carry a parsed, user-facing description of the cause.
    """
    try:
        result = geometry.GeometryData(observer, **data_kwargs).get_geometry(u_gps_times, fields=list(fields))
    except sp.utils.exceptions.SpiceyError as err:
        raise RuntimeError(f"curryer geometry query failed for {observer!r}: {spice_error_message(err)}") from err
    if require_coverage:
        coverage_columns = [column for field in (coverage_fields or fields) for column in field.columns]
        if bool(result[coverage_columns].isna().to_numpy().all()):
            raise RuntimeError(
                f"curryer geometry returned no coverage for observer {observer!r} over the granule; "
                "check that the SPICE kernels cover the requested times."
            )
    return result


def calculate_spacecraft_geometry(
    kernel_manager: KernelManager,
    timestamps: np.ndarray,
    spacecraft_observer: str = DEFAULT_SPACECRAFT_OBSERVER,
) -> pd.DataFrame:
    """
    Compute the spacecraft-level geometry fields via curryer ``GeometryData``.

    curryer's selective-compute registry queries each SPICE input once, vectorized, with
    coverage gaps surfaced as NaN. The spacecraft observer yields the fields needing only its
    ephemeris and body attitude: subsatellite and subsolar points, satellite radius, Earth-Sun
    distance, inertial (J2000) position and velocity, and the Earth-fixed attitude quaternion.
    None of them resolve through the camera's instrument frame, so the same call serves the
    production and ``jpss_only`` modes.

    Parameters
    ----------
    kernel_manager : KernelManager
        Kernel manager with the spacecraft SPK and CK and the generic/static kernels furnished.
    timestamps : np.ndarray
        Camera frame times on the L1B output time grid, as ``datetime64[ns]``.
    spacecraft_observer : str
        SPICE frame for the spacecraft, one of :data:`SPACECRAFT_OBSERVERS`. Default
        ``"JPSS4_SC"``.

    Returns
    -------
    pd.DataFrame
        Indexed by uGPS; the curryer columns of every field in ``_SPACECRAFT_FIELDS``.

    Raises
    ------
    ValueError
        If ``spacecraft_observer`` is not a known Libera spacecraft frame.
    RuntimeError
        If the curryer SPICE query fails outright, or returns no coverage at all. Both carry a
        parsed, user-facing description of the cause.
    """
    _validate_observer(spacecraft_observer, SPACECRAFT_OBSERVERS, "spacecraft")
    kernel_manager.ensure_known_kernels_are_furnished()
    u_gps_times = spicetime.adapt(pd.DatetimeIndex(timestamps), "iso")
    # Spacecraft fields resolve in every mode, so all-NaN means the kernels miss the granule. The
    # attitude quaternion is Earth-fixed (product convention); the inertial fields keep J2000.
    return _query_geometry(
        spacecraft_observer,
        u_gps_times,
        _SPACECRAFT_FIELDS,
        require_coverage=True,
        coverage_fields=(geometry.GeometryField.SUBSATELLITE,),
        attitude_frame=spatial.EARTH_FRAME,
    )


def create_placeholder_spacecraft_geometry(n_samples: int) -> pd.DataFrame:
    """
    Placeholder spacecraft geometry for ``use_geo`` false mode.

    Mirrors :func:`calculate_spacecraft_geometry`'s columns filled with the product fill values,
    so the dataset assembly always reads a geometry DataFrame and never branches on ``None``.
    Distance fields (radius, position) use -9999; everything else the product default -999.

    Parameters
    ----------
    n_samples : int
        Number of camera frames on the L1B output time grid.

    Returns
    -------
    pd.DataFrame
        One column per ``_SPACECRAFT_FIELDS`` output column, filled with the fill value.
    """
    default_fill = np.full(n_samples, _FILL_VALUE, dtype=np.float32)
    distance_fill = np.full(n_samples, _FILL_DISTANCE, dtype=np.float64)
    distance_columns = {
        geometry.GeometryField.SC_RADIUS.columns[0],
        *geometry.GeometryField.SC_POSITION_INERTIAL.columns,
    }
    data = {
        column: (distance_fill if column in distance_columns else default_fill)
        for field in _SPACECRAFT_FIELDS
        for column in field.columns
    }
    return pd.DataFrame(data)


def granule_earth_sun_distance(distances: np.ndarray) -> float:
    """
    Reduce the per-frame Earth-Sun distance to the single granule-level attribute.

    The distance moves by well under 1e-4 AU across a granule, so any covered frame
    represents the whole granule. The median is taken over the covered frames only, so a
    SPICE gap at a fixed index (e.g. the midpoint) cannot poison the attribute.

    Parameters
    ----------
    distances : np.ndarray
        Per-frame Earth-Sun distance in AU, NaN where SPICE had no coverage.

    Returns
    -------
    float
        Median Earth-Sun distance over the covered frames, in AU.

    Raises
    ------
    RuntimeError
        If no frame has a finite distance. The Sun ephemeris comes from the generic NAIF
        kernels, so this means they are not furnished rather than a per-frame gap.
    """
    if not np.isfinite(distances).any():
        raise RuntimeError("Earth-Sun distance has no coverage over the granule; check the generic NAIF kernels.")
    return float(np.nanmedian(distances))


def _assign_spacecraft_geometry(ds: xr.Dataset, geometry_data: pd.DataFrame) -> xr.Dataset:
    """Map the curryer geometry columns onto the product variables, filling coverage gaps."""
    for column, (variable, fill, dtype) in _SCALAR_VARIABLES.items():
        ds[variable] = (("camera_time",), _apply_fill(geometry_data[column].to_numpy(), fill, dtype))
    for field, (variable, fill, dtype) in _VECTOR_VARIABLES.items():
        vector = geometry_data[list(field.columns)].to_numpy()
        ds[variable] = (("camera_time", "EUCLIDEAN_DIM"), _apply_fill(vector, fill, dtype))
    return ds


def add_spacecraft_geometry_to_dataset(ds: xr.Dataset, config: GeolocationKernelConfig) -> xr.Dataset:
    """
    Compute the spacecraft-level geometry fields and add them to the dataset.

    Unlike the per-pixel geolocation, these fields are one value per camera frame, so they
    are computed eagerly on the client rather than through Dask ``map_blocks``: the arrays
    are (N,) or (N, 3), and a map_blocks round-trip would cost more than the calculation.
    Kernels are furnished and unloaded within this call so the SPICE pool is left as it was
    found.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing a ``camera_time`` coordinate.
    config : GeolocationKernelConfig
        Configuration for SPICE kernel management. ``config.jpss_only`` is irrelevant here:
        no field in this set resolves through the instrument frame.

    Returns
    -------
    xr.Dataset
        The dataset with the sub-point, satellite radius, inertial position and velocity, and
        attitude quaternion variables added, and the granule-level ``Earth_Sun_Distance_AU``
        attribute set.

    Raises
    ------
    ValueError
        If ``ds`` has no ``camera_time`` coordinate or ``config`` names no kernel sources.
    RuntimeError
        If the curryer SPICE query fails or the kernels do not cover the granule at all
        (see :func:`calculate_spacecraft_geometry`).
    """
    if "camera_time" not in ds.coords:
        raise ValueError("Dataset must have 'camera_time' coordinate.")
    if not config.dynamic_kernel_sources:
        raise ValueError("SPICE kernel sources are required to compute spacecraft geometry")

    with _kernel_manager_from_config(config) as km:
        km.load_libera_dynamic_kernels(
            config.dynamic_kernel_sources,
            needs_naif_kernels=True,
            needs_static_kernels=True,
        )
        geometry_data = calculate_spacecraft_geometry(km, ds.camera_time.values)

    ds = _assign_spacecraft_geometry(ds, geometry_data)
    (earth_sun_distance,) = geometry.GeometryField.EARTH_SUN_DISTANCE.columns
    ds.attrs[_EARTH_SUN_DISTANCE_ATTRIBUTE] = granule_earth_sun_distance(geometry_data[earth_sun_distance].to_numpy())
    return ds


def add_placeholder_spacecraft_geometry_to_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Add product fill-value placeholder spacecraft geometry to the dataset.

    Used in ground-data mode, where no SPICE kernels are available. The product definition
    declares these variables and the ``Earth_Sun_Distance_AU`` attribute, and the writer
    rejects a dataset that omits a declared field, so they must be present even unfilled.
    """
    ds = _assign_spacecraft_geometry(ds, create_placeholder_spacecraft_geometry(ds.sizes["camera_time"]))
    ds.attrs[_EARTH_SUN_DISTANCE_ATTRIBUTE] = _FILL_EARTH_SUN_DISTANCE

    logger.info("use_geo is false: using placeholder spacecraft geometry.")

    return ds


def calculate_azimuth(
    kernel_manager: KernelManager,
    timestamps: np.ndarray,
    fill_value: float = _FILL_VALUE,
) -> np.ndarray:
    """
    Azimuth motor encoder angle at each camera frame, from the azimuth CK.

    The angle is the third 1-2-3 Euler angle of the ``LIBERA_BASE_COORD -> LIBERA_AZ_COORD``
    rotation (:data:`AZIMUTH_ENCODER_FRAMES`), wrapped to ``[0, 360)`` -- the convention
    ``libera_rad`` uses for its ``Azimuth``, validated against the ``libera_utils`` tier-0
    kernel tests. It is the corrected encoder reading the CK was built from, relative to the
    instrument base, not a pointing angle in the spacecraft or orbital frame. curryer's
    ``frame_to_frame_euler`` does the per-sample frame transform and factoring.

    Every per-sample SPICE failure -- a CK coverage gap, but equally a CK or frame kernel that
    is not furnished -- comes back as NaN under ``allow_nans`` and is written as ``fill_value``;
    a granule with no covered frame is logged as a warning, not raised. The caller furnishes the
    kernels (:func:`add_azimuth_to_dataset` loads the static set, which carries the Libera frame
    kernel, alongside the CK), and the granule-level coverage check is tracked under
    LIBSDC-788, matching ``libera_rad``.

    Parameters
    ----------
    kernel_manager : KernelManager
        Kernel manager with the azimuth CK and the Libera frame kernel furnished.
    timestamps : np.ndarray
        Camera frame times on the L1B output time grid, as ``datetime64[ns]``.
    fill_value : float
        Product fill value for frames the CK does not cover.

    Returns
    -------
    np.ndarray
        Azimuth in degrees, shape ``(N,)``, dtype float32, in ``[0, 360)`` or ``fill_value``.
    """
    kernel_manager.ensure_known_kernels_are_furnished()

    # TODO[LIBSDC-788]: CK coverage check via KernelManager; an uncovered granule currently fills.

    u_gps_times = spicetime.adapt(pd.DatetimeIndex(timestamps), "iso")
    base_frame, azimuth_frame = AZIMUTH_ENCODER_FRAMES
    euler = spatial.frame_to_frame_euler(base_frame, azimuth_frame, u_gps_times, sequence=(1, 2, 3), allow_nans=True)

    azimuth = np.mod(euler["euler3"].to_numpy(), 360.0)
    if np.isnan(azimuth).all():
        logger.warning(
            "Azimuth CK returned no coverage over %d camera frame(s); Azimuth is written as fill.", azimuth.size
        )
    return _apply_fill(azimuth, fill_value, np.float32)


def create_placeholder_azimuth(n_samples: int, fill_value: float = _FILL_VALUE) -> np.ndarray:
    """
    Placeholder motor azimuth for ``use_geo`` false mode.

    Parameters
    ----------
    n_samples : int
        Number of camera frames on the L1B output time grid.
    fill_value : float
        Product fill value for ``Azimuth``.

    Returns
    -------
    np.ndarray
        Shape ``(N,)``, dtype float32, filled with ``fill_value``.
    """
    return np.full(n_samples, fill_value, dtype=np.float32)


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


def add_azimuth_to_dataset(ds: xr.Dataset, config: GeolocationKernelConfig) -> xr.Dataset:
    """
    Compute the motor azimuth from the azimuth CK and add it to the dataset as ``Azimuth``.

    One value per camera frame, computed eagerly on the client like the spacecraft geometry.
    Kernels are furnished and unloaded within this call.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing a ``camera_time`` coordinate.
    config : GeolocationKernelConfig
        Configuration for SPICE kernel management; its sources must include the azimuth CK.

    Returns
    -------
    xr.Dataset
        The dataset with the ``Azimuth`` variable added.

    Raises
    ------
    ValueError
        If ``ds`` has no ``camera_time`` coordinate or ``config`` names no kernel sources.
    """
    if "camera_time" not in ds.coords:
        raise ValueError("Dataset must have 'camera_time' coordinate.")
    if not config.dynamic_kernel_sources:
        raise ValueError("SPICE kernel sources are required to compute the motor azimuth")

    with _kernel_manager_from_config(config) as km:
        km.load_libera_dynamic_kernels(
            config.dynamic_kernel_sources,
            needs_naif_kernels=True,
            needs_static_kernels=True,
        )
        azimuth = calculate_azimuth(km, ds.camera_time.values)

    ds[_AZIMUTH_VARIABLE] = (("camera_time",), azimuth)
    return ds


def add_placeholder_azimuth_to_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Add the product fill-value ``Azimuth`` for ``use_geo`` false mode."""
    ds[_AZIMUTH_VARIABLE] = (("camera_time",), create_placeholder_azimuth(ds.sizes["camera_time"]))
    return ds


def add_jpss_only_azimuth_to_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Add the zero-degree reference ``Azimuth`` for ``jpss_only`` mode."""
    ds[_AZIMUTH_VARIABLE] = (("camera_time",), create_jpss_only_azimuth(ds.sizes["camera_time"]))
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


# FrameGeometry field -> (curryer PixelGeometry attribute, product variable, _FillValue, dtype).
_PIXEL_VARIABLES: dict[str, tuple[str, str, float, type]] = {
    "latitude": ("lat", "Latitude", _GEO_FILL_LAT_LON, np.float32),
    "longitude": ("lon", "Longitude", _GEO_FILL_LAT_LON, np.float32),
    "altitude": ("alt", "Altitude", _GEO_FILL_ALT, np.float32),
    "solar_zenith": ("solar_zenith", "Solar_Zenith_Surface", _FILL_VALUE, np.float32),
    "solar_azimuth": ("solar_azimuth", "Solar_Azimuth_Surface_WRT_North", _FILL_VALUE, np.float32),
    "viewing_zenith": ("viewing_zenith", "Viewing_Zenith_Surface", _FILL_VALUE, np.float32),
    "viewing_azimuth": ("viewing_azimuth", "Viewing_Azimuth_Surface_WRT_North", _FILL_VALUE, np.float32),
    "relative_azimuth": ("relative_azimuth", "Relative_Azimuth_Surface", _FILL_VALUE, np.float32),
}
_KM_TO_M = 1000.0
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
) -> FrameGeometry:
    """Per-pixel geometry of one camera frame via curryer ``pixel_geometry``, with product fills.

    The per-frame unit of the geolocation path: pure, one frame in, one frame out, no kernel
    management and no Dask. Every pixel that hits the ellipsoid gets geodetic latitude and
    longitude and the solar and viewing zenith and azimuth at that point; nothing is
    interpolated. ``altitude`` is the ellipsoid reference and is 0 m at every hit (there is no
    terrain here; that is ``Terrain_Corrected_Altitude``, LIBSDC-814). A frame is geolocated at
    one or more exposure epochs: curryer runs once over all ``K`` epochs and each pixel takes
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

    Returns
    -------
    FrameGeometry
        Float32 fields with the product fill where a pixel misses the ellipsoid or its epoch
        is uncovered; altitude in meters; azimuths wrapped so the float32 rounding of a value
        just under 360 lands on 0, not 360; ``quality_flags`` as uint16, 0 where good.

    Raises
    ------
    ValueError
        If ``spice_body`` is not one of the two camera bodies, ``exposure_ugps`` is not a
        non-empty 1-D array, ``exposure_index`` is absent for several epochs, present for one,
        not integer, wrongly shaped or out of range, or ``pointing_vectors`` is not
        ``(Y * X, 3)``.
    RuntimeError
        If a curryer quality flag does not fit the uint16 word this function narrows the flags
        to.

    Notes
    -----
    Memory: for the full detector at one epoch curryer returns about 440 MB and works in about
    1.05 GB of transients (250 bytes per pixel), so a call peaks near 1.5 GB and scales with
    ``K``; the returned ``FrameGeometry`` is 143 MB per frame. Size worker chunks from these.
    """
    _validate_observer(spice_body, _CAMERA_BODIES, "camera")
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

    # Fill-and-flag on a SPICE gap is this function's contract (a granule with no covered frame
    # is rejected upstream), so the choice is made here rather than left to curryer's default.
    result = spatial.pixel_geometry(epochs, spice_body, vectors, allow_nans=True)
    pixel_ids = None if flat_index is None else np.arange(n_pixels)

    def per_pixel(values: np.ndarray) -> np.ndarray:
        """Select each pixel's epoch from a (K, Y * X) curryer array."""
        return values[0] if flat_index is None else values[flat_index, pixel_ids]

    fields = {}
    for name, (curryer_name, _, fill, dtype) in _PIXEL_VARIABLES.items():
        values = per_pixel(getattr(result, curryer_name))
        if name == "altitude":
            values = values * _KM_TO_M  # 0 at every hit today; kept for a non-zero upstream ``alt``
        values = _apply_fill(values, fill, dtype)
        if name in _AZIMUTH_FIELDS:
            values[values == _FULL_TURN] = 0.0
        fields[name] = values.reshape(frame_shape)

    flags = per_pixel(result.quality_flags)
    if flags.max() > np.iinfo(np.uint16).max:
        raise RuntimeError(f"curryer quality flag {flags.max():#x} does not fit the uint16 word FrameGeometry uses")
    return FrameGeometry(quality_flags=flags.astype(np.uint16).reshape(frame_shape), **fields)


_GEOLOCATION_FLAG_VARIABLE = "Geolocation_Quality_Flag"
# Bit 15 of Geolocation_Quality_Flag: geolocation was not run (use_geo false). Bits 0-14 are curryer's
# SpatialQualityFlags, which reach 0x4000, and 0 reads as good geometry, so "not computed" needs a bit
# of its own.
_GEO_NOT_COMPUTED_FLAG = np.uint16(1 << 15)
# FrameGeometry field -> (product variable, dtype) for every field the geolocation path writes, in
# FrameGeometry field order (the worker returns its blocks in this order).
_FIELD_VARIABLES: dict[str, tuple[str, type]] = {
    name: (
        (_GEOLOCATION_FLAG_VARIABLE, np.uint16)
        if name == "quality_flags"
        else (_PIXEL_VARIABLES[name][1], _PIXEL_VARIABLES[name][3])
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
        If ``exposure_times`` is not 2-D, or ``exposure_index`` is given with a shape other than
        ``(T, Y, X)``. The per-frame epoch, index and vector checks are :func:`geolocate_frame`'s.
    """
    times = np.asarray(exposure_times)
    if times.ndim != 2:
        raise ValueError(f"exposure_times must be (T, K) epochs, got shape {times.shape}")
    n_frames, n_epochs = times.shape
    frame_shape = (PIXEL_COUNT_Y, PIXEL_COUNT_X)
    if exposure_index is not None and exposure_index.shape != (n_frames, *frame_shape):
        raise ValueError(f"exposure_index must be shaped {(n_frames, *frame_shape)}, got {exposure_index.shape}")

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
            )
            for output, values in zip(outputs, frame, strict=True):
                output[i] = values
    return outputs


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
    RuntimeError
        If no frame is covered.
    """
    with _kernel_manager_from_config(config) as km:
        km.load_libera_dynamic_kernels(
            config.dynamic_kernel_sources,
            needs_naif_kernels=True,
            needs_static_kernels=True,
        )
        ugps = np.asarray(spicetime.adapt(pd.DatetimeIndex(timestamps), "iso"))
        probe = spatial.pixel_geometry(ugps, spice_body, _BORESIGHT, allow_nans=True)

    uncovered = (probe.quality_flags[:, 0] & int(SpatialQualityFlags.CALC_ELLIPS_INSUFF_DATA)) != 0
    if uncovered.all():
        raise RuntimeError(
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

    Every pixel of every frame is geolocated at its own epoch through :func:`geolocate_frame`;
    nothing is masked, subsampled or interpolated. The work is split into tasks of
    ``LIBERA_CAM_GEO_CHUNK_SIZE`` frames along ``camera_time``, independent of the
    decompression chunking; each task returns one ``(T, Y, X)`` block per field and the blocks
    are concatenated into lazy arrays under the product variable names. The kernels are
    furnished once on the client first, which populates the kernel cache ahead of the workers
    and rejects a granule the kernels do not cover at all.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a ``camera_time`` coordinate (``datetime64[ns]``) and ``(camera_time, y, x)``
        image dimensions.
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
        If ``ds`` has no ``camera_time`` coordinate, ``config`` names no kernel sources, or
        ``LIBERA_CAM_GEO_CHUNK_SIZE`` is below 1.
    RuntimeError
        If the kernels cover none of the frames.

    Notes
    -----
    Each frame is geolocated at one epoch, its ``camera_time``. The worker already takes
    ``(T, K)`` epochs and a per-pixel ``(T, Y, X)`` index, so the two exposures of a frame get
    their own epochs once their timing is known. TODO[LIBSDC-816]
    """
    if "camera_time" not in ds.coords:
        raise ValueError("Dataset must have 'camera_time' coordinate.")
    if not config.dynamic_kernel_sources:
        raise ValueError("SPICE kernel sources are required for per-pixel geolocation")
    chunk_size = _geometry_chunk_size()

    timestamps = ds.camera_time.values
    _require_frame_coverage(config, timestamps, _geometry_spice_body(config))

    n_frames = timestamps.size
    exposure_times = timestamps[:, None]
    frame_shape = (PIXEL_COUNT_Y, PIXEL_COUNT_X)
    chunk_geometry = delayed(calculate_chunk_geometry, nout=len(_FIELD_VARIABLES), pure=False)
    blocks: dict[str, list[da.Array]] = {name: [] for name in _FIELD_VARIABLES}
    for start in range(0, n_frames, chunk_size):
        stop = min(start + chunk_size, n_frames)
        outputs = chunk_geometry(exposure_times[start:stop], None, config)
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
    fills = {name: fill for name, (_, _, fill, _) in _PIXEL_VARIABLES.items()} | {
        "quality_flags": _GEO_NOT_COMPUTED_FLAG
    }
    for name, (variable, dtype) in _FIELD_VARIABLES.items():
        ds[variable] = (("camera_time", "y", "x"), da.full(shape, fills[name], dtype=dtype, chunks=chunks))

    logger.info("use_geo is false: per-pixel geolocation written as fill values.")

    return ds
