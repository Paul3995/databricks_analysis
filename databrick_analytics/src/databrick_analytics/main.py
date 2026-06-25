"""
Pipeline entry point for Databricks jobs.

Imports the Databricks SparkSession here — not in taxis.py — so the
transformation module stays importable in local pytest without a cluster.
"""
import argparse
import logging

from databricks.sdk.runtime import spark

from databrick_analytics import taxis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="NYC Taxi analytics pipeline on Databricks")
    parser.add_argument("--catalog", required=True, help="Unity Catalog catalog name")
    parser.add_argument("--schema", required=True, help="Unity Catalog schema name")
    args = parser.parse_args()

    spark.sql(f"USE CATALOG {args.catalog}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {args.catalog}.{args.schema}")

    taxis.run_pipeline(spark=spark, catalog=args.catalog, schema=args.schema)


if __name__ == "__main__":
    main()
