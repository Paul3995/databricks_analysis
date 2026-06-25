"""
PySpark transformation layer for the NYC Taxi sample dataset.

All public functions are pure: DataFrame in → DataFrame out, no side effects.
This separation means every function can be unit-tested locally with plain
PySpark — no Databricks cluster or DatabricksSession required.

The SparkSession is never imported here; callers inject it, keeping the
module portable between local tests and a live Databricks runtime.

Design decisions are commented inline so they are defensible in interviews.
"""
from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_trips(spark: SparkSession, table: str = "samples.nyctaxi.trips") -> DataFrame:
    """
    Read NYC taxi trips from a Unity Catalog table.

    Args:
        spark: Active SparkSession (injected by the caller).
        table: Fully-qualified table name. Defaults to the Databricks sample.

    Returns:
        Raw trips DataFrame.
    """
    logger.info("Reading trips from %s", table)
    return spark.read.table(table)


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

def clean_trips(df: DataFrame) -> DataFrame:
    """
    Drop nulls on critical columns and remove business-logic outliers.

    Outlier bounds are derived from the TLC data dictionary:
    - Distance < 0.1 mi: GPS noise or cancelled trips
    - Distance > 100 mi: data-entry error for an NYC metro context
    - Fare ≤ 0: refunds or meter errors — not meaningful revenue
    - Fare > 500: well above TLC maximums; treated as corrupt data
    - Dropoff before pickup: timestamp corruption — must be excluded or
      duration and speed calculations will produce negative values

    Args:
        df: Raw trips DataFrame.

    Returns:
        Cleaned DataFrame with outliers removed.
    """
    critical_cols = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_distance",
        "fare_amount",
        "total_amount",
    ]
    return (
        df
        .dropna(subset=critical_cols)
        .filter(
            (F.col("trip_distance") > 0.1) & (F.col("trip_distance") < 100)
            & (F.col("fare_amount") > 0)   & (F.col("fare_amount") < 500)
            & (F.col("tpep_dropoff_datetime") > F.col("tpep_pickup_datetime"))
        )
    )


def add_derived_columns(df: DataFrame) -> DataFrame:
    """
    Enrich each trip row with computed business metrics.

    Columns added:
    - trip_duration_minutes: float — used for speed & experience analysis
    - speed_mph: float — useful for congestion studies; None when duration ≤ 0
    - tip_pct: float — tip as % of base fare; 0.0 when fare_amount is zero
    - pickup_date: date — used for time-series aggregations
    - pickup_hour: int (0–23) — used for hourly demand analysis
    - day_of_week: string — Monday–Sunday, used for weekly seasonality

    speed_mph and tip_pct use F.when to guard against division by zero rather
    than try/except — PySpark transforms execute as a distributed SQL DAG, so
    Python exceptions cannot catch arithmetic errors at row level.

    Args:
        df: Cleaned trips DataFrame.

    Returns:
        Enriched DataFrame with derived columns appended.
    """
    duration_minutes = (
        F.col("tpep_dropoff_datetime").cast("long")
        - F.col("tpep_pickup_datetime").cast("long")
    ) / 60.0

    return (
        df
        .withColumn("trip_duration_minutes", F.round(duration_minutes, 2))
        .withColumn(
            "speed_mph",
            # Guard: only compute speed when the trip has positive duration.
            F.when(
                duration_minutes > 0,
                F.round(F.col("trip_distance") / (duration_minutes / 60.0), 2),
            ).otherwise(F.lit(None).cast("double")),
        )
        .withColumn(
            "tip_pct",
            # Guard: avoid division by zero on zero-fare trips.
            F.when(
                F.col("fare_amount") > 0,
                F.round(F.col("tip_amount") / F.col("fare_amount") * 100, 2),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn("pickup_date", F.to_date("tpep_pickup_datetime"))
        .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
        .withColumn("day_of_week", F.date_format("tpep_pickup_datetime", "EEEE"))
    )


def compute_hourly_revenue(df: DataFrame) -> DataFrame:
    """
    Aggregate trip data to hourly granularity.

    Business question: When during the day does the most revenue flow?
    Use case: surge-pricing windows, driver shift scheduling, demand forecasting.

    revenue_per_trip normalises by volume so a busy-but-cheap hour
    (e.g. morning commute) is not confused with a genuinely high-value hour
    (e.g. late-night airport runs).

    Args:
        df: Enriched trips DataFrame (must include pickup_hour,
            trip_duration_minutes, tip_pct).

    Returns:
        One row per hour (0–23) sorted by hour, with revenue and volume metrics.
    """
    return (
        df
        .groupBy("pickup_hour")
        .agg(
            F.count("*").alias("total_trips"),
            F.round(F.sum("total_amount"), 2).alias("total_revenue"),
            F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
            F.round(F.avg("trip_distance"), 2).alias("avg_distance_miles"),
            F.round(F.avg("trip_duration_minutes"), 1).alias("avg_duration_minutes"),
            F.round(F.avg("tip_pct"), 2).alias("avg_tip_pct"),
            F.round(F.sum("total_amount") / F.count("*"), 2).alias("revenue_per_trip"),
        )
        .orderBy("pickup_hour")
    )


def compute_pickup_zone_revenue(df: DataFrame) -> DataFrame:
    """
    Aggregate revenue by pickup zip code.

    Business question: Which pickup areas generate the most revenue?
    Use case: fleet allocation, driver incentive zone bonuses.

    trips_per_revenue_dollar inverts the question — a zip with many cheap
    short trips has a worse ratio than one with fewer long-distance trips.
    This separates high-volume/low-value zones from low-volume/high-value ones
    (e.g. airport runs vs. local hops), which is the more actionable insight
    for fleet operators.

    Args:
        df: Enriched trips DataFrame (must include pickup_zip, tip_pct,
            trip_duration_minutes).

    Returns:
        One row per pickup zip, ordered by total_revenue descending.
        Null pickup_zip rows are excluded — they cannot be acted on.
    """
    return (
        df
        .filter(F.col("pickup_zip").isNotNull())
        .groupBy("pickup_zip")
        .agg(
            F.count("*").alias("total_trips"),
            F.round(F.sum("total_amount"), 2).alias("total_revenue"),
            F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
            F.round(F.avg("trip_distance"), 2).alias("avg_distance_miles"),
            F.round(F.avg("trip_duration_minutes"), 1).alias("avg_duration_minutes"),
            F.round(F.avg("tip_pct"), 2).alias("avg_tip_pct"),
            F.round(F.count("*") / F.sum("total_amount"), 4).alias(
                "trips_per_revenue_dollar"
            ),
        )
        .orderBy(F.desc("total_revenue"))
    )


def compute_daily_zone_stats(df: DataFrame) -> DataFrame:
    """
    Aggregate trip data to daily × pickup_zip granularity.

    This intermediate aggregation feeds the rolling-revenue window below.
    Pre-aggregating here reduces the window computation from operating on
    millions of raw trip rows to at most (distinct_days × distinct_zips)
    rows — orders of magnitude smaller and proportionally faster.

    Args:
        df: Enriched trips DataFrame (must include pickup_date, pickup_zip).

    Returns:
        One row per (pickup_date, pickup_zip). Null pickup_zip excluded.
    """
    return (
        df
        .filter(F.col("pickup_zip").isNotNull())
        .groupBy("pickup_date", "pickup_zip")
        .agg(
            F.count("*").alias("total_trips"),
            F.round(F.sum("total_amount"), 2).alias("total_revenue"),
            F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
        )
    )


def add_rolling_revenue(df: DataFrame) -> DataFrame:
    """
    Compute a 7-day rolling average revenue per pickup zip.

    Window spec rationale:
    - partitionBy(pickup_zip): each zip gets its own independent window so
      a slow week in one area does not dilute a busy week in another.
    - orderBy(pickup_date cast to long): rangeBetween requires a numeric
      ORDER BY column, so the date is cast to Unix seconds.
    - rangeBetween(-6 * 86400, 0): include today plus the 6 preceding
      calendar days (7 days total).

    We use rangeBetween rather than rowsBetween(-6, 0) because rowsBetween
    counts preceding *rows*, which breaks silently when a zip has missing
    days (e.g. no trips on a holiday). rangeBetween works on actual timestamp
    values, so it always spans exactly 7 calendar days regardless of data gaps.

    Args:
        df: Daily zone stats DataFrame (output of compute_daily_zone_stats).

    Returns:
        Same DataFrame with rolling_7d_avg_revenue column appended.
    """
    seconds_per_day = 86_400
    window = (
        Window
        .partitionBy("pickup_zip")
        .orderBy(F.col("pickup_date").cast("long"))
        .rangeBetween(-6 * seconds_per_day, 0)
    )
    return df.withColumn(
        "rolling_7d_avg_revenue",
        F.round(F.avg("total_revenue").over(window), 2),
    )


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(spark: SparkSession, catalog: str, schema: str) -> None:
    """
    Execute the full analytics pipeline end-to-end.

    Stages:
        1. Read raw trips from the Databricks sample catalog
        2. Clean & validate (drop nulls, filter business-logic outliers)
        3. Enrich with derived columns (duration, speed, tip %, time parts)
        4. Write enriched trip-level table to Unity Catalog
        5. Aggregate to hourly revenue summary → write
        6. Aggregate to pickup zone revenue summary → write
        7. Aggregate to daily × zone stats, compute 7-day rolling window → write

    The SparkSession is injected rather than imported from databricks.sdk.runtime
    so this function is callable from local pytest (plain local Spark) and from
    Databricks jobs (cluster Spark) with no code change — a standard dependency
    injection pattern that keeps business logic free of infrastructure concerns.

    Args:
        spark: Active SparkSession.
        catalog: Unity Catalog catalog name (e.g. "dev_databricks_analytics").
        schema: Unity Catalog schema name (e.g. "md").
    """
    logger.info("=== NYC Taxi pipeline starting — %s.%s ===", catalog, schema)

    raw = read_trips(spark)
    raw_count = raw.count()
    logger.info("Raw trips read: %s rows", f"{raw_count:,}")

    clean = clean_trips(raw)
    clean_count = clean.count()
    logger.info(
        "After cleaning: %s rows (%.1f%% retained)",
        f"{clean_count:,}",
        100 * clean_count / raw_count if raw_count else 0,
    )

    enriched = add_derived_columns(clean)

    logger.info("Writing trips_enriched table...")
    enriched.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.trips_enriched")

    logger.info("Computing and writing hourly_revenue table...")
    compute_hourly_revenue(enriched).write.mode("overwrite").saveAsTable(
        f"{catalog}.{schema}.hourly_revenue"
    )

    logger.info("Computing and writing zone_revenue table...")
    compute_pickup_zone_revenue(enriched).write.mode("overwrite").saveAsTable(
        f"{catalog}.{schema}.zone_revenue"
    )

    logger.info("Computing daily zone stats and 7-day rolling revenue...")
    daily = compute_daily_zone_stats(enriched)
    add_rolling_revenue(daily).write.mode("overwrite").saveAsTable(
        f"{catalog}.{schema}.daily_zone_rolling"
    )

    logger.info("=== Pipeline complete ===")
