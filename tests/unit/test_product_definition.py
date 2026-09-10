"""Tests for the bundled L1B product definition."""

import tomllib
from pathlib import Path

import yaml
from libera_utils.io.product_definition import LiberaDataProductDefinition

from libera_cam.config import product_config_path
from libera_cam.version import version as libera_cam_version


def test_product_definition_algorithm_version_is_dynamic():
    """algorithm_version must be null in the YAML so it is injected from the package at write time."""
    definition = yaml.safe_load(product_config_path.read_text())
    assert definition["attributes"]["algorithm_version"] is None


def test_spacecraft_geometry_fields_are_declared_per_frame():
    """The spacecraft-level fields are one value per camera frame, not per pixel."""
    definition = yaml.safe_load(product_config_path.read_text())

    # Dynamic attribute, injected at write time like algorithm_version.
    assert definition["attributes"]["Earth_Sun_Distance_AU"] is None

    per_frame = [
        "Subsatellite_Latitude",
        "Subsatellite_Longitude",
        "Subsatellite_Colatitude",
        "Subsolar_Latitude",
        "Subsolar_Longitude",
        "Subsolar_Colatitude",
        "Radius_of_Satellite_from_Center_of_Earth",
        "Satellite_Attitude_Q0",
        "Satellite_Attitude_Q1",
        "Satellite_Attitude_Q2",
        "Satellite_Attitude_Q3",
        "Azimuth",
    ]
    for name in per_frame:
        assert definition["variables"][name]["dimensions"] == ["CAMERA_TIME"]

    for name in ["Satellite_Position", "Satellite_Velocity"]:
        assert definition["variables"][name]["dimensions"] == ["CAMERA_TIME", "EUCLIDEAN_DIM"]
        assert definition["variables"][name]["dtype"] == "float64"


def test_surface_angles_are_declared_per_pixel_with_rad_conventions():
    """The surface-angle set matches libera_rad: degrees, -999 fill, azimuths on [0, 360], zeniths on [0, 180]."""
    definition = yaml.safe_load(product_config_path.read_text())["variables"]
    per_pixel = ["CAMERA_TIME", "CAMERA_PIXEL_COUNT_X", "CAMERA_PIXEL_COUNT_Y"]
    expected_ranges = {
        "Solar_Zenith_Surface": [0, 180],
        "Viewing_Zenith_Surface": [0, 180],
        "Relative_Azimuth_Surface": [0, 360],
        "Viewing_Azimuth_Surface_WRT_North": [0, 360],
        "Solar_Azimuth_Surface_WRT_North": [0, 360],
    }
    for name, valid_range in expected_ranges.items():
        variable = definition[name]
        assert variable["dimensions"] == per_pixel
        assert variable["dtype"] == "float32"
        assert variable["attributes"]["units"] == "degrees"
        assert variable["attributes"]["valid_range"] == valid_range
        assert variable["attributes"]["_FillValue"] == -999


def test_geolocation_quality_flag_is_a_per_pixel_uint16_bitmask():
    """curryer's flags fill bits 0-14 and bit 15 marks not-run; no _FillValue, so readers keep the integer dtype."""
    variable = yaml.safe_load(product_config_path.read_text())["variables"]["Geolocation_Quality_Flag"]
    assert variable["dimensions"] == ["CAMERA_TIME", "CAMERA_PIXEL_COUNT_X", "CAMERA_PIXEL_COUNT_Y"]
    assert variable["dtype"] == "uint16"
    assert "_FillValue" not in variable["attributes"]
    assert "[b15]" in variable["attributes"]["value_meaning"]


def test_per_pixel_geolocation_variables_declare_shuffle():
    """Byte shuffling ahead of the mandatory gzip 4 shrinks the geolocation set without changing a value.

    Asserted through the loaded definition rather than the YAML text: libera_utils merges the
    declaration as ``{**encoding, **DEFAULT_ENCODING}``, so this is the encoding that reaches the file.
    """
    definition = LiberaDataProductDefinition.from_yaml(product_config_path)
    shuffled = [
        "Latitude",
        "Longitude",
        "Altitude",
        "Terrain_Corrected_Latitude",
        "Terrain_Corrected_Longitude",
        "Terrain_Corrected_Altitude",
        "Solar_Zenith_Surface",
        "Viewing_Zenith_Surface",
        "Solar_Azimuth_Surface_WRT_North",
        "Viewing_Azimuth_Surface_WRT_North",
        "Relative_Azimuth_Surface",
        "Geolocation_Quality_Flag",
    ]
    for name in shuffled:
        assert definition.variables[name].encoding["shuffle"] is True, name

    # Pinned explicitly on the large non-geolocation fields: netCDF4-python shuffles whenever zlib
    # is on and h5py does not, so leaving these undeclared makes the product engine-dependent.
    assert definition.variables["Radiance"].encoding["shuffle"] is False
    assert definition.variables["Pixel_Counts"].encoding["shuffle"] is True


def test_package_version_matches_pyproject():
    """The installed package version must match pyproject.toml."""
    pyproject_path = Path(__file__).parents[2] / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        pyproject_version = tomllib.load(pyproject_file)["project"]["version"]
    assert libera_cam_version() == pyproject_version
