"""Tests for GeolocationKernelConfig dynamic_kernel_sources (including mocked S3)."""

from pathlib import Path

import pytest
from cloudpathlib import S3Path
from libera_utils.libera_spice import spice_utils
from libera_utils.libera_spice.kernel_manager import KernelManager

from libera_cam.geolocation import GeolocationKernelConfig, prefetch_kernels


def _ditl_dynamic_kernel_files(test_data_path: Path, limit: int = 2) -> list[Path]:
    kernel_dir = test_data_path / "DITL_3min"
    kernel_files = sorted(
        [p for p in kernel_dir.iterdir() if p.is_file() and p.suffix in {".bc", ".bsp"}],
        key=lambda p: p.name,
    )
    assert kernel_files, f"No dynamic kernels found under {kernel_dir}"
    return kernel_files[:limit]


@pytest.mark.parametrize("source_wrapper", ["s3_uri", "S3Path"])
def test_dynamic_kernel_s3_sources_materialize_into_cache(
    monkeypatch,
    tmp_path,
    test_data_path,
    write_file_to_s3,
    source_wrapper,
):
    """Manifest-style s3:// strings and S3Path objects materialize into the user cache."""
    kernel_files = _ditl_dynamic_kernel_files(test_data_path)
    bucket = "libera-cam-test-kernels"

    s3_sources: list[str | S3Path] = []
    for kernel_file in kernel_files:
        uri = f"s3://{bucket}/ditl/{kernel_file.name}"
        write_file_to_s3(kernel_file, uri)
        if source_wrapper == "s3_uri":
            s3_sources.append(uri)
        else:
            s3_sources.append(S3Path(uri))

    monkeypatch.setattr(spice_utils.caching, "get_local_cache_dir", lambda: tmp_path)

    config = GeolocationKernelConfig(
        dynamic_kernel_sources=s3_sources,
        use_test_naif_url=False,
        cache_timeout_days=7,
    )
    prefetch_kernels(config)

    km = KernelManager(cache_timeout_days=7)
    km.load_libera_dynamic_kernels(
        config.dynamic_kernel_sources,
        needs_naif_kernels=True,
        needs_static_kernels=True,
    )

    for kernel_file in kernel_files:
        cached = tmp_path / kernel_file.name
        assert cached.is_file(), f"Expected cached kernel missing: {kernel_file.name}"
        assert cached.read_bytes() == kernel_file.read_bytes()
