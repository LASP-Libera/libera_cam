import sys
from pathlib import Path

import pytest

pytest_plugins = ["tests.plugins.aws_fixtures"]


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch for autouse AWS fixtures."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="session")
def test_data_path():
    """Returns the Path to the test_data directory"""
    return Path(sys.modules[__name__.split(".")[0]].__file__).parent / "test_data"


@pytest.fixture
def generate_input_manifest(tmp_path, test_data_path):
    """Build and write an input manifest from DITL test data.

    Returns a callable: ``generate_input_manifest(configuration={})`` writes a
    manifest with the given ``configuration`` dict and returns its path.
    """
    from libera_utils.io.manifest import Manifest, ManifestType

    ditl_data_path = test_data_path / "DITL_3min"
    filenames = (
        ditl_data_path / "LIBERA_L1A_WFOV-SCI-DECODED_V5-8-4_20280212T080001_20280212T080342_R26157025734.nc",
        ditl_data_path / "LIBERA_SPICE_AZROT-CK_V5-8-4_20280212T033106_20280212T040027_R26157131711.bc",
        ditl_data_path / "LIBERA_SPICE_JPSS-CK_V5-4-2_20280212T000000_20280212T235959_R26006200637.bc",
        ditl_data_path / "LIBERA_SPICE_JPSS-SPK_V5-4-2_20280212T000000_20280212T235959_R26006200633.bsp",
    )

    def _build(configuration: dict | None = None):
        input_manifest = Manifest(
            manifest_type=ManifestType.INPUT,
            files=[],
            configuration=dict(configuration) if configuration is not None else {},
        )
        input_manifest.add_files(*filenames)
        return input_manifest.write(tmp_path)

    return _build


@pytest.fixture
def test_ditl_l1a_file_path(test_data_path):
    """Returns the Path to a sample L1A NetCDF file from the Day in the Life (DITL) test data"""
    return (
        test_data_path
        / "DITL_short"
        / "LIBERA_L1A_WFOV-SCI-DECODED_V5-4-2_20280215T135304_20280215T142141_R26021133743.nc"
    )


@pytest.fixture
def local_data_path():
    """Returns the Path to the calibration_data directory"""
    return Path(sys.modules[__name__.split(".")[0]].__file__).parent.parent / "libera_cam" / "ground_calibration_data"
