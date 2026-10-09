"""Tests for cli module"""

# Installed
import argparse
import logging
from unittest.mock import patch

import pytest

# Local
from libera_cam import cli, l1b


@pytest.mark.parametrize(
    ("cli_args", "parsed"),
    [
        (
            ["-v", "input_manifest.json"],
            argparse.Namespace(func=l1b.algorithm, manifest="input_manifest.json", verbose=True),
        ),
    ],
)
def test_parse_cli_args(cli_args, parsed):
    assert dict(vars(cli.parse_cli_args(cli_args))) == dict(vars(parsed))


@pytest.mark.parametrize(("cli_args", "level"), [(["m.json"], logging.INFO), (["-v", "m.json"], logging.DEBUG)])
def test_main_configures_logging_before_running(cli_args, level):
    calls = []
    with (
        patch.object(cli, "configure_task_logging", side_effect=lambda *a, **k: calls.append(("logging", k))),
        patch.object(l1b, "algorithm", side_effect=lambda args: calls.append(("algorithm", args.manifest))),
    ):
        cli.main(cli_args)
    assert calls == [("logging", {"console_log_level": level}), ("algorithm", "m.json")]
