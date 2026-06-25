"""
Unit tests for the PySpark transformation logic in taxis.py.

All tests use the local_spark fixture (a plain local SparkSession defined in
conftest.py) and small in-memory DataFrames.  No Databricks connection is needed
— these tests run in CI and on any developer laptop with Java installed.

Each test class covers one transformation function and includes:
- A positive case (valid data is kept / correct values produced)
- Negative / boundary cases (bad data is dropped or handled safely)
"""
from __future__ import annotations

import datetime

import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql import types as T

from databrick_analytics.taxis import (
    add_derived_columns,
    add_rolling_revenue,
    clean_trips,
    compute_daily_zone_stats,
    compute_hourly_revenue,
    compute_pickup_zone_revenue,
)

# ---------------------------------------------------------------------------
# Shared test data helpers
# ---------------------------------------------------------------------------

# Two timestamps exactly 30 minutes apart, used across multiple test classes.
_PICKUP  = datetime.datetime(2024, 1, 15, 9, 0, 0)   # Monday 09:00
_DROPOFF = datetime.datetime(2024, 1, 15, 9, 30, 0)  # Monday 09:30

# Minimal schema that mirrors samples.nyctaxi.trips
TRIP_SCHEMA = T.StructType([
    T.StructField("tpep_pickup_datetime",  T.TimestampType()),
    T.StructField("tpep_dropoff_datetime", T.TimestampType()),
    T.StructField("trip_distance",         T.DoubleType()),
    T.StructField("fare_amount",           T.DoubleType()),
    T.StructField("tip_amount",            T.DoubleType()),
    T.StructField("tolls_amount",          T.DoubleType()),
    T.StructField("total_amount",          T.DoubleType()),
    T.StructField("pickup_zip",            T.StringType()),
    T.StructField("dropoff_zip",           T.StringType()),
])


def _valid_row(**overrides) -> dict:
    """Return a minimal valid trip row.  Use keyword overrides to vary specific fields."""
    base = {
        "tpep_pickup_datetime":  _PICKUP,
        "tpep_dropoff_datetime": _DROPOFF,
        "trip_distance":  5.0,
        "fare_amount":   15.0,
        "tip_amount":     3.0,
        "tolls_amount":   0.0,
        "total_amount":  18.0,
        "pickup_zip":  "10001",
        "dropoff_zip": "10022",
    }
    return {**base, **overrides}


def _make_trips(spark: SparkSession, rows: list[dict]) -> object:
    return spark.createDataFrame([Row(**r) for r in rows], schema=TRIP_SCHEMA)


# ---------------------------------------------------------------------------
# clean_trips
# ---------------------------------------------------------------------------

class TestCleanTrips:
    def test_keeps_valid_row(self, local_spark):
        df = _make_trips(local_spark, [_valid_row()])
        assert clean_trips(df).count() == 1

    @pytest.mark.parametrize("distance", [0.0, 0.05, 150.0])
    def test_removes_out_of_range_distance(self, local_spark, distance):
        df = _make_trips(local_spark, [_valid_row(trip_distance=distance)])
        assert clean_trips(df).count() == 0

    @pytest.mark.parametrize("fare", [-5.0, 0.0, 600.0])
    def test_removes_invalid_fare(self, local_spark, fare):
        df = _make_trips(local_spark, [_valid_row(fare_amount=fare)])
        assert clean_trips(df).count() == 0

    def test_removes_dropoff_before_pickup(self, local_spark):
        # Reversed timestamps must be rejected — otherwise duration is negative.
        df = _make_trips(local_spark, [_valid_row(
            tpep_pickup_datetime=_DROPOFF,
            tpep_dropoff_datetime=_PICKUP,
        )])
        assert clean_trips(df).count() == 0

    def test_removes_null_distance(self, local_spark):
        df = _make_trips(local_spark, [_valid_row(trip_distance=None)])
        assert clean_trips(df).count() == 0

    def test_removes_null_fare(self, local_spark):
        df = _make_trips(local_spark, [_valid_row(fare_amount=None)])
        assert clean_trips(df).count() == 0

    def test_removes_null_total_amount(self, local_spark):
        df = _make_trips(local_spark, [_valid_row(total_amount=None)])
        assert clean_trips(df).count() == 0

    def test_keeps_multiple_valid_rows(self, local_spark):
        rows = [_valid_row(trip_distance=d) for d in [1.0, 3.5, 9.0]]
        df = _make_trips(local_spark, rows)
        assert clean_trips(df).count() == 3

    def test_boundary_distance_included(self, local_spark):
        # 0.11 mi is just above the 0.1 threshold — should be kept.
        df = _make_trips(local_spark, [_valid_row(trip_distance=0.11)])
        assert clean_trips(df).count() == 1

    def test_boundary_fare_included(self, local_spark):
        # $0.01 is the minimum valid fare.
        df = _make_trips(local_spark, [_valid_row(fare_amount=0.01)])
        assert clean_trips(df).count() == 1


# ---------------------------------------------------------------------------
# add_derived_columns
# ---------------------------------------------------------------------------

class TestAddDerivedColumns:
    def test_trip_duration_30_minutes(self, local_spark):
        # _PICKUP → _DROPOFF is exactly 30 minutes.
        df = _make_trips(local_spark, [_valid_row()])
        row = add_derived_columns(df).collect()[0]
        assert row["trip_duration_minutes"] == pytest.approx(30.0, abs=0.1)

    def test_speed_10_mph(self, local_spark):
        # 5 miles / 0.5 hours = 10 mph.
        df = _make_trips(local_spark, [_valid_row()])
        row = add_derived_columns(df).collect()[0]
        assert row["speed_mph"] == pytest.approx(10.0, abs=0.1)

    def test_tip_pct_20_percent(self, local_spark):
        # tip=3, fare=15 → 3/15 * 100 = 20 %.
        df = _make_trips(local_spark, [_valid_row()])
        row = add_derived_columns(df).collect()[0]
        assert row["tip_pct"] == pytest.approx(20.0, abs=0.01)

    def test_tip_pct_zero_when_no_tip(self, local_spark):
        df = _make_trips(local_spark, [_valid_row(tip_amount=0.0)])
        row = add_derived_columns(df).collect()[0]
        assert row["tip_pct"] == pytest.approx(0.0, abs=0.01)

    def test_tip_pct_zero_when_fare_is_zero(self, local_spark):
        # Division-by-zero guard: fare_amount=0 → tip_pct must be 0.0, not NaN.
        df = _make_trips(local_spark, [_valid_row(fare_amount=0.01, tip_amount=0.0)])
        row = add_derived_columns(df).collect()[0]
        assert row["tip_pct"] == pytest.approx(0.0, abs=0.01)

    def test_pickup_date_extracted(self, local_spark):
        df = _make_trips(local_spark, [_valid_row()])
        row = add_derived_columns(df).collect()[0]
        assert row["pickup_date"] == datetime.date(2024, 1, 15)

    def test_pickup_hour_extracted(self, local_spark):
        df = _make_trips(local_spark, [_valid_row()])
        row = add_derived_columns(df).collect()[0]
        assert row["pickup_hour"] == 9

    def test_day_of_week_is_monday(self, local_spark):
        # 2024-01-15 is a Monday.
        df = _make_trips(local_spark, [_valid_row()])
        row = add_derived_columns(df).collect()[0]
        assert row["day_of_week"] == "Monday"

    def test_all_derived_columns_present(self, local_spark):
        df = _make_trips(local_spark, [_valid_row()])
        result = add_derived_columns(df)
        expected = {"trip_duration_minutes", "speed_mph", "tip_pct",
                    "pickup_date", "pickup_hour", "day_of_week"}
        assert expected.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# compute_hourly_revenue
# ---------------------------------------------------------------------------

class TestComputeHourlyRevenue:

    ENRICHED_SCHEMA = T.StructType([
        T.StructField("pickup_hour",          T.IntegerType()),
        T.StructField("total_amount",         T.DoubleType()),
        T.StructField("fare_amount",          T.DoubleType()),
        T.StructField("trip_distance",        T.DoubleType()),
        T.StructField("trip_duration_minutes",T.DoubleType()),
        T.StructField("tip_pct",              T.DoubleType()),
    ])

    def _make_enriched(self, spark, rows):
        return spark.createDataFrame([Row(**r) for r in rows], schema=self.ENRICHED_SCHEMA)

    def _base_rows(self):
        return [
            {"pickup_hour": 9,  "total_amount": 18.0, "fare_amount": 15.0,
             "trip_distance": 5.0, "trip_duration_minutes": 30.0, "tip_pct": 20.0},
            {"pickup_hour": 9,  "total_amount": 22.0, "fare_amount": 18.0,
             "trip_distance": 7.0, "trip_duration_minutes": 40.0, "tip_pct": 15.0},
            {"pickup_hour": 10, "total_amount": 10.0, "fare_amount":  8.0,
             "trip_distance": 2.0, "trip_duration_minutes": 12.0, "tip_pct": 10.0},
        ]

    def test_produces_one_row_per_hour(self, local_spark):
        df = self._make_enriched(local_spark, self._base_rows())
        assert compute_hourly_revenue(df).count() == 2

    def test_total_revenue_for_hour_9(self, local_spark):
        df = self._make_enriched(local_spark, self._base_rows())
        rows = {r["pickup_hour"]: r for r in compute_hourly_revenue(df).collect()}
        assert rows[9]["total_revenue"] == pytest.approx(40.0, abs=0.01)

    def test_trip_count_for_hour_9(self, local_spark):
        df = self._make_enriched(local_spark, self._base_rows())
        rows = {r["pickup_hour"]: r for r in compute_hourly_revenue(df).collect()}
        assert rows[9]["total_trips"] == 2

    def test_revenue_per_trip_for_hour_9(self, local_spark):
        # (18 + 22) / 2 trips = 20.0
        df = self._make_enriched(local_spark, self._base_rows())
        rows = {r["pickup_hour"]: r for r in compute_hourly_revenue(df).collect()}
        assert rows[9]["revenue_per_trip"] == pytest.approx(20.0, abs=0.01)


# ---------------------------------------------------------------------------
# compute_pickup_zone_revenue
# ---------------------------------------------------------------------------

class TestComputePickupZoneRevenue:

    ZONE_SCHEMA = T.StructType([
        T.StructField("pickup_zip",            T.StringType()),
        T.StructField("total_amount",          T.DoubleType()),
        T.StructField("fare_amount",           T.DoubleType()),
        T.StructField("trip_distance",         T.DoubleType()),
        T.StructField("trip_duration_minutes", T.DoubleType()),
        T.StructField("tip_pct",               T.DoubleType()),
    ])

    def _make_zone_df(self, spark):
        rows = [
            Row(pickup_zip="10001", total_amount=30.0, fare_amount=25.0,
                trip_distance=5.0, trip_duration_minutes=20.0, tip_pct=18.0),
            Row(pickup_zip="10001", total_amount=20.0, fare_amount=16.0,
                trip_distance=3.0, trip_duration_minutes=12.0, tip_pct=15.0),
            Row(pickup_zip="10002", total_amount=50.0, fare_amount=40.0,
                trip_distance=10.0, trip_duration_minutes=35.0, tip_pct=20.0),
            Row(pickup_zip=None, total_amount=12.0, fare_amount=10.0,
                trip_distance=2.0, trip_duration_minutes=8.0, tip_pct=5.0),
        ]
        return spark.createDataFrame(rows, schema=self.ZONE_SCHEMA)

    def test_excludes_null_zip(self, local_spark):
        result = compute_pickup_zone_revenue(self._make_zone_df(local_spark))
        zips = {r["pickup_zip"] for r in result.collect()}
        assert None not in zips

    def test_correct_trip_count_for_zip(self, local_spark):
        rows = {r["pickup_zip"]: r for r in
                compute_pickup_zone_revenue(self._make_zone_df(local_spark)).collect()}
        assert rows["10001"]["total_trips"] == 2

    def test_correct_total_revenue_for_zip(self, local_spark):
        rows = {r["pickup_zip"]: r for r in
                compute_pickup_zone_revenue(self._make_zone_df(local_spark)).collect()}
        assert rows["10001"]["total_revenue"] == pytest.approx(50.0, abs=0.01)

    def test_ordered_by_revenue_descending(self, local_spark):
        result = compute_pickup_zone_revenue(self._make_zone_df(local_spark)).collect()
        revenues = [r["total_revenue"] for r in result]
        assert revenues == sorted(revenues, reverse=True)


# ---------------------------------------------------------------------------
# add_rolling_revenue
# ---------------------------------------------------------------------------

class TestAddRollingRevenue:

    DAILY_SCHEMA = T.StructType([
        T.StructField("pickup_date",   T.DateType()),
        T.StructField("pickup_zip",    T.StringType()),
        T.StructField("total_trips",   T.LongType()),
        T.StructField("total_revenue", T.DoubleType()),
        T.StructField("avg_fare",      T.DoubleType()),
    ])

    def _three_day_df(self, spark):
        """Three consecutive days for one zip: $1k, $2k, $3k revenue."""
        rows = [
            Row(pickup_date=datetime.date(2024, 1, d), pickup_zip="10001",
                total_trips=100, total_revenue=float(1000 * d), avg_fare=10.0)
            for d in range(1, 4)
        ]
        return spark.createDataFrame(rows, schema=self.DAILY_SCHEMA)

    def test_rolling_column_exists(self, local_spark):
        result = add_rolling_revenue(self._three_day_df(local_spark))
        assert "rolling_7d_avg_revenue" in result.columns

    def test_day1_rolling_equals_own_revenue(self, local_spark):
        # Only one row in the window on day 1 — rolling avg = that day's value.
        rows = add_rolling_revenue(self._three_day_df(local_spark)).orderBy("pickup_date").collect()
        assert rows[0]["rolling_7d_avg_revenue"] == pytest.approx(1000.0, abs=1.0)

    def test_day3_rolling_is_mean_of_all_three(self, local_spark):
        # Days 1–3 all fall within the 7-day window → avg(1000, 2000, 3000) = 2000.
        rows = add_rolling_revenue(self._three_day_df(local_spark)).orderBy("pickup_date").collect()
        assert rows[2]["rolling_7d_avg_revenue"] == pytest.approx(2000.0, abs=1.0)

    def test_independent_windows_per_zip(self, local_spark):
        """Two zips should not bleed into each other's rolling average."""
        zip_a = [
            Row(pickup_date=datetime.date(2024, 1, 1), pickup_zip="AAA",
                total_trips=10, total_revenue=100.0, avg_fare=10.0),
        ]
        zip_b = [
            Row(pickup_date=datetime.date(2024, 1, 1), pickup_zip="BBB",
                total_trips=10, total_revenue=900.0, avg_fare=90.0),
        ]
        df = local_spark.createDataFrame(zip_a + zip_b, schema=self.DAILY_SCHEMA)
        result = {r["pickup_zip"]: r["rolling_7d_avg_revenue"]
                  for r in add_rolling_revenue(df).collect()}
        assert result["AAA"] == pytest.approx(100.0, abs=1.0)
        assert result["BBB"] == pytest.approx(900.0, abs=1.0)


# ---------------------------------------------------------------------------
# compute_daily_zone_stats
# ---------------------------------------------------------------------------

class TestComputeDailyZoneStats:
    def test_aggregates_to_one_row_per_day_zip(self, local_spark):
        rows = [
            _valid_row(pickup_zip="10001"),
            _valid_row(pickup_zip="10001"),
            _valid_row(pickup_zip="10002"),
        ]
        enriched_schema = T.StructType([
            T.StructField("pickup_date",   T.DateType()),
            T.StructField("pickup_zip",    T.StringType()),
            T.StructField("total_amount",  T.DoubleType()),
            T.StructField("fare_amount",   T.DoubleType()),
        ])
        data = [
            Row(pickup_date=datetime.date(2024, 1, 15), pickup_zip="10001",
                total_amount=18.0, fare_amount=15.0),
            Row(pickup_date=datetime.date(2024, 1, 15), pickup_zip="10001",
                total_amount=22.0, fare_amount=18.0),
            Row(pickup_date=datetime.date(2024, 1, 15), pickup_zip="10002",
                total_amount=10.0, fare_amount=8.0),
        ]
        df = local_spark.createDataFrame(data, schema=enriched_schema)
        result = compute_daily_zone_stats(df)
        assert result.count() == 2

    def test_excludes_null_zip(self, local_spark):
        enriched_schema = T.StructType([
            T.StructField("pickup_date",  T.DateType()),
            T.StructField("pickup_zip",   T.StringType()),
            T.StructField("total_amount", T.DoubleType()),
            T.StructField("fare_amount",  T.DoubleType()),
        ])
        data = [
            Row(pickup_date=datetime.date(2024, 1, 15), pickup_zip=None,
                total_amount=15.0, fare_amount=12.0),
        ]
        df = local_spark.createDataFrame(data, schema=enriched_schema)
        assert compute_daily_zone_stats(df).count() == 0
