"""Module for the Libera WFOV camera L1b processing CLI

libera-cam
"""

# Standard
import argparse

# Local
from libera_cam import l1b
from libera_cam.version import version as libera_cam_version


def main(cli_args: list = None):
    """Main CLI entrypoint that runs the function inferred from the specified subcommand"""
    args = parse_cli_args(cli_args)
    args.func(args)


def print_version_info(*args):
    """Print CLI version information"""
    print(f"Libera WFOV camera science data processing CLI\n\tVersion {libera_cam_version()}")


def parse_cli_args(cli_args: list):
    """Parse CLI arguments

    Parameters
    ----------
    cli_args : list
        List of CLI arguments to parse

    Returns
    -------
    Namespace
        Parsed arguments in a Namespace object
    """
    parser = argparse.ArgumentParser(prog="libera-cam", description="Libera WFOV camera science data processing CLI")
    parser.add_argument(
        "--version",
        action="store_const",
        dest="func",
        const=print_version_info,
        help="print current version of the CLI",
    )

    parser.set_defaults(func=l1b.algorithm)
    parser.add_argument("manifest", type=str, help="input manifest file")
    parser.add_argument("-v", "--verbose", action="store_true", help="set DEBUG level logging output")

    parsed_args = parser.parse_args(cli_args)
    return parsed_args
