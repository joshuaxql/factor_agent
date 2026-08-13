# Core Tests

The test suite intentionally covers only behavior maintained by this fork:

- `test_data_pipeline.py`: Tushare normalization and Qlib provider integrity.
- `test_exchange_limit_status.py`: directional limit-status trading rules.
- `test_evaluate.py`: market-return and long-short evaluation behavior.

All tests are offline and use temporary or in-memory data. Run them with:

```bash
python -m pytest -q
```
