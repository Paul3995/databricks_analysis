# databrick_analytics

See the [project README](../README.md) for full documentation, architecture, and design decisions.

## Quick start

```bash
# Deploy to dev
databricks bundle deploy --target dev

# Run jobs
databricks bundle run databricks_data_refresh --target dev
databricks bundle run franchise_analytics --target dev

# Run unit tests locally (no cluster needed)
pip install pyspark==3.5.3 pytest pytest-cov
pip install -e . --no-deps
pytest tests/test_taxis.py -v
```
