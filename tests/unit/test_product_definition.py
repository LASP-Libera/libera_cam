"""Tests for the bundled L1B product definition."""

import tomllib
from pathlib import Path

import yaml

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


def test_package_version_matches_pyproject():
    """The installed package version must match pyproject.toml."""
    pyproject_path = Path(__file__).parents[2] / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        pyproject_version = tomllib.load(pyproject_file)["project"]["version"]
    assert libera_cam_version() == pyproject_version
