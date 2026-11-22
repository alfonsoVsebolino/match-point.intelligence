# E2E Test Suite Ready (`tests/e2e/`)

The independent, requirement-driven, opaque-box E2E test suite (`Tiers 1–4`) is ready for Milestone M4 execution.

## Suite Inventory (`112` Tests Total, `0` Skips, `0` Stubs)
| Tier | File | Marker | Count | Scope |
|------|------|--------|-------|-------|
| Tier 1 | `tests/e2e/test_tier1_features.py` | `tier1` | 47 | Core functional coverage (`>=5` per feature F1–F9) |
| Tier 2 | `tests/e2e/test_tier2_boundaries.py` | `tier2` | 47 | Boundary, error, and corner cases (`>=5` per feature F1–F9) |
| Tier 3 | `tests/e2e/test_tier3_combinations.py` | `tier3` | 12 | Pairwise cross-feature combinations (`>=10` pairs) |
| Tier 4 | `tests/e2e/test_tier4_e2e_scenarios.py` | `tier4` | 6 | End-to-end live `uvicorn` + headless Chromium CDP scenarios |

Supporting harness files:
- `tests/e2e/__init__.py`
- `tests/e2e/conftest.py` (ephemeral `$PORT` live server fixture, `httpx(trust_env=False)`, math/contract oracles)
- `tests/e2e/cdp_runner.mjs` (Node v24 + `/usr/bin/chromium` headless CDP browser driver)

## Execution Command for M4 Gate
Run from `/home/antigravity/alfonsoVsebolino/match-point.intelligence`:
```bash
NO_PROXY="" no_proxy="" .venv/bin/pytest tests/e2e -v
```
And verify existing unit/integration tests remain green alongside E2E:
```bash
NO_PROXY="" no_proxy="" .venv/bin/pytest -v
```

## M4 Pass Criteria
1. **100% Pass Rate**: `112 passed, 0 failed, 0 skipped` across `tests/e2e/`.
2. **Real Browser Health**: Zero Chromium console errors, zero uncaught page exceptions, zero failed local HTTP requests, zero external CDN requests, and `#060913` / neon visual rendering verified.
3. **Clean Workspace**: No leftover artifacts written to repository `data/`.
