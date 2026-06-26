# Databricks Analytics Pipeline

A production-style analytics pipeline built on Databricks, using PySpark for transformations and Unity Catalog for storage. It ingests NYC Taxi and Bakehouse sample data, runs it through a multi-stage processing layer, and writes analytics-ready tables that get queried through notebooks. The whole thing deploys automatically to a Databricks workspace via GitHub Actions.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Databricks Asset Bundle                          │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  Python Module  src/databrick_analytics/taxis.py            │    │
│  │                                                              │    │
│  │  read_trips → clean_trips → add_derived_columns             │    │
│  │      → compute_hourly_revenue                               │    │
│  │      → compute_pickup_zone_revenue                          │    │
│  │      → compute_daily_zone_stats → add_rolling_revenue       │    │
│  └────────────────────────┬─────────────────────────────────────┘    │
│                           │ writes to                                │
│  ┌────────────────────────▼─────────────────────────────────────┐    │
│  │  Unity Catalog  {catalog}.{schema}                          │    │
│  │                                                              │    │
│  │  trips_enriched      hourly_revenue    zone_revenue         │    │
│  │  daily_zone_rolling  customer_360      transaction_summary  │    │
│  │  franchise_analytics                                        │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Lakeflow Jobs (resources/)         Notebooks (src/)                │
│  databricks_data_refresh (daily) -> customer_360.ipynb              │
│                                     transaction_summary.ipynb       │
│  franchise_analytics            -> franchise_analytics.ipynb        │
└──────────────────────────────────────────────────────────────────────┘
         |
         | GitHub Actions
         v
  Unit tests (pytest + local PySpark) -> Bundle deploy to prod workspace
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Platform | Databricks Asset Bundles | Infrastructure-as-code for workspace, jobs, and permissions |
| Transformation | PySpark 3.5 | Partitioning, broadcast joins, window functions |
| Analytics DB | Unity Catalog | Centralised governance, fine-grained access control, Delta Lake |
| Orchestration | Lakeflow Jobs | Native Databricks scheduler with task dependency management |
| Testing | pytest + local PySpark | Unit tests that run without a cluster |
| CI/CD | GitHub Actions | Tests on every push; bundle deploy to prod on merge to main |

---

## Project Structure

```
databricks_analysis/
|
+-- .github/
|   +-- workflows/
|       +-- ci.yml                  # Unit tests on every push (no cluster needed)
|       +-- prod_deployment.yml     # Bundle deploy on merge to main
|
+-- databrick_analytics/            # Databricks Asset Bundle root
    +-- databricks.yml              # Bundle config: dev + prod workspace targets
    |
    +-- resources/
    |   +-- databricks_data_refresh.job.yml   # Daily job: customer + transaction
    |   +-- franchise_analytics.job.yml       # Franchise analytics job
    |
    +-- src/
    |   +-- databrick_analytics/
    |   |   +-- taxis.py            # PySpark transforms (pure functions, testable)
    |   |   +-- main.py             # CLI entry point for the taxi pipeline job
    |   |
    |   +-- customer_360.ipynb          # Customer geo + revenue analysis
    |   +-- transaction_summary.ipynb   # Product x payment revenue breakdown
    |   +-- franchise_analytics/
    |       +-- franchise_analytics.ipynb  # Franchise network analysis
    |
    +-- tests/
    |   +-- conftest.py             # local_spark + Databricks fixtures
    |   +-- test_taxis.py           # ~30 offline unit tests for taxis.py
    |   +-- sample_taxis_test.py    # Integration test (requires live cluster)
    |
    +-- pyproject.toml
    +-- databricks.yml
```

---

## How to Run Locally

### Prerequisites

- Python 3.11+
- Java 11+ (`java -version` to confirm — required by PySpark)
- A Databricks workspace with Unity Catalog enabled

### 1. Clone and install

```bash
git clone https://github.com/Paul3995/databricks_analysis.git
cd databricks_analysis/databrick_analytics

# Using uv (recommended)
pip install uv
uv sync --dev

# Or plain pip
pip install -e ".[dev]"
```

### 2. Configure the Databricks CLI

```bash
databricks configure --host https://dbc-3410a8fa-59bb.cloud.databricks.com
# Paste your personal access token when prompted
```

### 3. Deploy to your dev workspace

```bash
databricks bundle deploy --target dev
```

### 4. Run a job

```bash
databricks bundle run databricks_data_refresh --target dev
databricks bundle run franchise_analytics --target dev
```

---

## Running the Tests

### Offline unit tests — no cluster needed

```bash
cd databrick_analytics
pytest tests/test_taxis.py -v
```

Covers: `clean_trips`, `add_derived_columns`, `compute_hourly_revenue`, `compute_pickup_zone_revenue`, `compute_daily_zone_stats`, `add_rolling_revenue`.

### Integration tests — requires a live cluster

```bash
export DATABRICKS_HOST=https://dbc-3410a8fa-59bb.cloud.databricks.com
export DATABRICKS_TOKEN=<your-personal-access-token>
pytest tests/sample_taxis_test.py -v
```

---

## Design Decisions

### Pure functions with injected SparkSession

All transformation logic in `taxis.py` takes `spark` as a parameter rather than importing it from `databricks.sdk.runtime`. This means the same function runs in a local pytest and in a Databricks job without any code change — you just pass a different session. It also makes unit testing straightforward since there's no global state to work around.

### `rangeBetween` vs `rowsBetween` for rolling revenue

The 7-day rolling revenue uses `rangeBetween(-6 * 86400, 0)` rather than `rowsBetween(-6, 0)`. The row-based version counts preceding rows, which silently breaks when a zone has missing days — say, no trips on a public holiday. The range-based version works on actual Unix timestamps, so it always covers exactly 7 calendar days regardless of gaps in the data.

### `DENSE_RANK()` over `RANK()` in notebooks

`DENSE_RANK()` handles ties without leaving gaps in the sequence. `RANK()` produces sequences like 1, 1, 3, 4... which confuse stakeholders reading "Top N markets" reports. `DENSE_RANK()` always produces a clean 1, 2, 3... which is what people actually expect.

### `NULLIF` in percentage calculations

Every division in the SQL notebooks wraps the denominator in `NULLIF(x, 0)`. This returns NULL instead of throwing a divide-by-zero error when a group has zero count. NULL propagates cleanly through aggregations and is much easier to handle downstream than NaN or Infinity.

### LEFT JOIN in customer_360

The customer-to-transaction join is a LEFT JOIN so countries with customers but no transactions still appear in the output. An INNER JOIN would silently drop those markets, making the customer count inconsistent with the raw table — a subtle data quality bug that's easy to miss.

### Dev vs prod targets

Separate targets in `databricks.yml` use different catalog names (`dev_databricks_analytics` vs `pro_databricks_analytics`). Each developer deploys to their own isolated catalog, and the CI/CD only promotes to prod on merge to main. This keeps dev experiments from polluting prod data.

### Two-workflow CI strategy

`ci.yml` runs unit tests on every push using plain PySpark — no credentials needed, no cluster costs. `prod_deployment.yml` only runs on merge to main and requires a service-principal token stored as a GitHub secret. Quality gates stay cheap and fast; the deployment step is gated behind peer review.

---

**Author:** [@Paul3995](https://github.com/Paul3995)
