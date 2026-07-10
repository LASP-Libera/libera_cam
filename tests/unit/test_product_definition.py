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

    assert "Earth_Sun_Distance_AU" in definition["attributes"]

    expected = [
        "Subsatellite_Latitude",
        "Subsatellite_Longitude",
        "Subsatellite_Colatitude",
        "Subsolar_Latitude",
        "Subsolar_Longitude",
        "Subsolar_Colatitude",
        "Radius_of_Satellite_from_Center_of_Earth",
    ]
    for name in expected:
        assert definition["variables"][name]["dimensions"] == ["CAMERA_TIME"]


def test_package_version_matches_pyproject():
    """The installed package version must match pyproject.toml."""
    pyproject_path = Path(__file__).parents[2] / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        pyproject_version = tomllib.load(pyproject_file)["project"]["version"]
    assert libera_cam_version() == pyproject_version
