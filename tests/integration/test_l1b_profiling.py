"""Diagnostic L1B profiling smoke tests for CI visibility."""

import os
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr
from dask.diagnostics import CacheProfiler, Profiler, ResourceProfiler, visualize
from libera_utils.io.manifest import Manifest

from libera_cam.l1b import algorithm

pytestmark = pytest.mark.integration


def func_prefix(task_repr: str) -> str:
    """Strip profiler task hex address suffix to get the base function name."""
    return task_repr.rsplit("-", 1)[0]


def log_profiling(prof: Profiler) -> None:
    """Print a normalized Dask task timing summary for CI and local logs."""
    records = []
    for result in prof.results:
        func_repr = result.key[0] if isinstance(result.key, tuple) else result.key
        duration = result.end_time - result.start_time
        records.append({"func": func_prefix(func_repr), "duration": duration})
    df = pd.DataFrame(records)

    summary = df.groupby("func")["duration"].sum()
    total_time = summary.sum()
    summary_normalized = (summary / total_time).sort_values(ascending=False)
    elapsed_time = prof.end_time - prof.start_time

    lines = [
        "",
        "=" * 60,
        "L1B Dask profiling summary",
        "=" * 60,
    ]
    for func, elapsed in summary_normalized.items():
        lines.append(f"  {func}: {elapsed:.3f}")
    lines.append(f"  total_time={total_time:.3f}, elapsed_time={elapsed_time:.3f}")
    lines.append("=" * 60)
    lines.append("")

    summary_text = "\n".join(lines)
    print(summary_text)

    github_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if github_summary:
        with open(github_summary, "a", encoding="utf-8") as summary_file:
            summary_file.write("### L1B Dask profiling summary\n\n```\n")
            summary_file.write(summary_text)
            summary_file.write("\n```\n")


@pytest.mark.profiling
def test_l1b_profiling_smoke(generate_input_manifest, tmp_path, monkeypatch):
    """Run L1B on DITL data and print Dask profiling output; fail only on exceptions."""
    monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
    monkeypatch.setenv("DASK_SCHEDULER", "synchronous")

    manifest_path = generate_input_manifest()
    run_visualize = os.environ.get("L1B_PROFILE_VISUALIZE") == "1"
    prof_dir = tmp_path / "prof"

    if run_visualize:
        prof_dir.mkdir(parents=True, exist_ok=True)
        with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
            output_manifest_path = algorithm(Namespace(manifest=manifest_path))
        log_profiling(prof)
        visualize([prof, rprof, cprof], filename=str(prof_dir / "dask_profile.html"))
    else:
        with Profiler() as prof:
            output_manifest_path = algorithm(Namespace(manifest=manifest_path))
        log_profiling(prof)

    output_manifest = Manifest.from_file(output_manifest_path)
    nc_files = [file for file in output_manifest.files if Path(file.filename).suffix == ".nc"]
    assert len(nc_files) == 1

    with xr.open_dataset(nc_files[0].filename) as data_product:
        assert "Radiance" in data_product
