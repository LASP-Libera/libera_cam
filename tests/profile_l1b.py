#! /usr/bin/env python3

import logging
import os
import shutil
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import pandas as pd
import xarray as xr
from dask.diagnostics import CacheProfiler, Profiler, ResourceProfiler, visualize
from libera_utils.io.manifest import Manifest, ManifestType

from libera_cam.l1b import algorithm

# set logging to debug for the test
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_test_data_path():
    """Returns the Path to the test_data directory"""
    return Path(sys.modules[__name__.split(".")[0]].__file__).parent / "test_data"


def generate_input_manifest(tmp_path, test_data_path):
    """Generate test manifest from the DITL_short data in test_data"""
    ditl_data_path = test_data_path / "DITL_short"

    filenames = [
        ditl_data_path / "LIBERA_L1A_WFOV-SCI-DECODED_V5-4-2_20280215T135304_20280215T142141_R26021133743.nc",
        ditl_data_path / "LIBERA_SPICE_AZROT-CK_V5-5-1_20280215T135304_20280215T142141_R26021234221.bc",
        ditl_data_path / "LIBERA_SPICE_JPSS-CK_V5-4-2_20280215T000000_20280215T220000_R26006200700.bc",
        ditl_data_path / "LIBERA_SPICE_JPSS-SPK_V5-4-2_20280215T000000_20280215T220000_R26006200656.bsp",
    ]

    input_manifest = Manifest(manifest_type=ManifestType.INPUT, files=filenames)

    input_manifest_file_path = input_manifest.write(tmp_path)

    return input_manifest_file_path


def func_prefix(task_repr: str) -> str:
    """Strip profiler task hex address suffix to get the base function name.
    This depends on a specific suffix format but works for all task types in the dask
    graph as currently defined for L1B processing."""
    return task_repr.rsplit("-", 1)[0]


def log_profiling(prof):
    # Build a DataFrame from profiler results
    records = []
    for r in prof.results:
        func_repr = r.key[0] if isinstance(r.key, tuple) else r.key
        duration = r.end_time - r.start_time
        records.append({"func": func_prefix(func_repr), "duration": duration})
    df = pd.DataFrame(records)

    # Create a normalized summary of time usage by each dask task type
    summary = df.groupby("func")["duration"].sum()
    total_time = summary.sum()
    summary_normalized = (summary / total_time).sort_values(ascending=False)
    elapsed_time = prof.end_time - prof.start_time

    # Log the timing summary
    for func, elapsed in summary_normalized.items():
        print(f"{func}: {elapsed:.3f}")
    print(f"total_time={total_time:.3f}, elapsed_time={elapsed_time:.3f}")


def run_l1b(input_manifest_file_path: str | Path, work_path: str | Path, run_visualize: bool = False):
    """Run L1B processing for input data specified by manifest

    Parameters
    ----------
    input_manifest_file_path: str | Path
        Path to the input manifest file
    work_path: str | Path
        Path to the work directory, which is used as the PROCESSING_PATH env variable
        as well as the dropbox output path
    run_visualize: bool
        Set to true to generate profile html and run bokeh visualization
    """
    os.environ["PROCESSING_PATH"] = str(work_path)

    algo_inputs = Namespace(manifest=input_manifest_file_path)
    logger.info(algo_inputs)

    if os.environ["DASK_SCHEDULER"] != "distributed":
        with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
            output_manifest_path = algorithm(algo_inputs)
        log_profiling(prof)
        if run_visualize:
            # Visualize the profiler results
            visualize([prof, rprof, cprof])
    else:
        # the dask distributed dashboard handles the profiling
        output_manifest_path = algorithm(algo_inputs)

    output_manifest_obj = Manifest.from_file(output_manifest_path)

    # Check we can open the output dataset and summarize it briefly
    for file in output_manifest_obj.files:
        if Path(file.filename).suffix == ".nc":
            data_product = xr.open_dataset(file.filename)
            print(data_product)


def main(
    input_path: Path,
    work_path: Path = Path("/data"),
    scheduler: str = "synchronous",
    num_workers: int = 1,
    memory_limit: str = "8GB",
    chunk_size: int = 50,
    run_visualize: bool = False,
):
    # copy input_path over the standard file in DITL_short
    dest_path = (
        get_test_data_path()
        / "DITL_short/LIBERA_L1A_WFOV-SCI-DECODED_V5-4-2_20280215T135304_20280215T142141_R26021133743.nc"
    )
    shutil.copyfile(input_path, dest_path)
    input_manifest_file_path = generate_input_manifest(work_path, get_test_data_path())

    os.environ["LIBERA_CAM_CHUNK_SIZE"] = str(chunk_size)
    os.environ["DASK_SCHEDULER"] = scheduler
    os.environ["DASK_NUM_WORKERS"] = str(num_workers)
    os.environ["DASK_MEMORY_LIMIT"] = memory_limit

    start = datetime.now()
    run_l1b(input_manifest_file_path, work_path, run_visualize=run_visualize)
    end = datetime.now()
    logger.info(f"Fully processed L1B in {(end - start).total_seconds()} seconds")


if __name__ == "__main__":
    # output and most temporary files go here
    gwork_path = Path("/data")
    # gwork_path = Path.home() / "data"

    # to create 2 GB ram_disk: diskutil erasevolume HFS+ 'ram_disk' $(hdiutil attach -nomount ram://4194304)
    # to destroy: diskutil eject /Volumes/ram_disk
    # 2GB is only enough for 1 or 2 copies
    # gwork_path = Path("/Volumes/ram_disk")

    # use only synchronous, processes, or distributed scheduler
    main(
        Path("/data/LIBERA_L1A_WFOV-SCI-DECODED_V5-4-2_20280215T135304_20280215T142141_R26021133743_10.nc"),
        work_path=gwork_path,
        scheduler="distributed",
        num_workers=4,
        memory_limit="4GB",
        chunk_size=10,
        # run_visualize=True
    )
