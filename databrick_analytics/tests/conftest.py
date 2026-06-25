"""
Pytest fixtures for the databrick_analytics test suite.

Two fixture paths:

1. local_spark — a plain local SparkSession, always available.
   Used by tests/test_taxis.py for offline unit tests.
   Runs in CI with no Databricks credentials needed.

2. spark — a DatabricksSession for integration tests that hit the live workspace.
   Automatically skipped when DATABRICKS_HOST / DATABRICKS_TOKEN are not set,
   so 'pytest tests/' works locally without a cluster.
"""
from __future__ import annotations

import os
import pathlib
import json
import csv

import pytest
from pyspark.sql import SparkSession


# ---------------------------------------------------------------------------
# Local Spark fixture — always available, no cluster needed
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def local_spark() -> SparkSession:
    """
    Session-scoped local SparkSession for offline unit tests.

    local[2] uses two threads — enough parallelism to exercise window
    functions while keeping CI startup time under ~15 s.

    spark.ui.enabled=false prevents port-binding flakiness in CI environments
    where multiple jobs may share the same host.

    shuffle.partitions=2 avoids creating 200 near-empty shuffle files for
    the tiny in-memory DataFrames used in unit tests.
    """
    return (
        SparkSession.builder
        .appName("databrick-analytics-unit-tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# Databricks fixture — skipped when credentials are unavailable
# ---------------------------------------------------------------------------

_databricks_creds_present = bool(
    os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN")
)

try:
    from databricks.connect import DatabricksSession
    from databricks.sdk import WorkspaceClient
    _databricks_available = True
except ImportError:
    _databricks_available = False


@pytest.fixture()
def spark() -> SparkSession:
    """
    DatabricksSession for integration tests that need a live cluster.

    Skipped automatically when DATABRICKS_HOST and DATABRICKS_TOKEN are not
    set, so the full test suite can run in CI without credentials.
    To run integration tests locally, configure your ~/.databrickscfg or set
    those environment variables before invoking pytest.
    """
    if not _databricks_available or not _databricks_creds_present:
        pytest.skip(
            "Databricks credentials not found — set DATABRICKS_HOST and "
            "DATABRICKS_TOKEN to run integration tests."
        )
    return DatabricksSession.builder.getOrCreate()


# ---------------------------------------------------------------------------
# Fixture file loader — uses local_spark
# ---------------------------------------------------------------------------

@pytest.fixture()
def load_fixture(local_spark: SparkSession):
    """
    Load a JSON or CSV fixture from the fixtures/ directory into a DataFrame.

    Usage:
        def test_something(load_fixture):
            df = load_fixture("sample_trips.json")
            assert df.count() >= 1
    """
    def _loader(filename: str):
        path = pathlib.Path(__file__).parent.parent / "fixtures" / filename
        suffix = path.suffix.lower()
        if suffix == ".json":
            rows = json.loads(path.read_text())
            return local_spark.createDataFrame(rows)
        if suffix == ".csv":
            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            return local_spark.createDataFrame(rows)
        raise ValueError(f"Unsupported fixture type: {filename}")

    return _loader
