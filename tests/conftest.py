# Root-level conftest

import os

import pytest
#
# Loads the benchmark plugin (CLI options, markers, fixtures).
# The plugin lives in a file deliberately NOT named conftest.py to
# avoid pluggy's duplicate-registration error.

pytest_plugins = [
    'tests.pytest_random_plugin',
    'tests.pytest_benchmark_plugin',
]


def _precompile_only():
    return os.getenv("TILEKERNELS_PRECOMPILE_ONLY", "").lower() in ("1", "true", "yes", "on")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if _precompile_only() and report.when == "call":
        report.outcome = "skipped"
        report.longrepr = "precompile-only: kernel compilation only"
