"""Tier 4 full end-to-end live server + real headless Chromium browser scenarios."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import time
from typing import Any

import httpx
import pytest

from tests.e2e.conftest import free_port, run_cdp_scenario, start_ephemeral_server

pytestmark = pytest.mark.tier4


def test_scenario_1_ephemeral_port_boot_healthz_and_neon_arena() -> None:
    """Scenario 1: Boot server.py on ephemeral $PORT, check /healthz & airgapped Neon Arena load."""
    port = free_port()
    with start_ephemeral_server(port=port, use_python_entrypoint=True, startup_timeout=25.0) as srv:
        assert srv["port"] == port
        with httpx.Client(base_url=srv["base_url"], trust_env=False, timeout=15.0) as client:
            health_resp = client.get("/healthz")
            assert health_resp.status_code == 200, health_resp.text
            health = health_resp.json()
            assert health["status"] == "ok"
            assert health["models_loaded"] is True
            assert health["players_total"] > 1000
            assert health["players_active"] > 0

        cdp_res = run_cdp_scenario(srv["base_url"], "boot_and_theme")
        assert cdp_res["ok"] is True
        assert cdp_res["errors"] == []
        data = cdp_res["data"]
        assert data["bodyBg"] == "rgb(248, 219, 193)"
        assert data["hasTealInk"] is True
        assert data["hasSignalPink"] is True
        assert data["hasSurfaceCream"] is True
        assert data["hasLegacyNeon"] is False
        assert isinstance(data["tickerText"], str) and len(data["tickerText"].strip()) > 5

        telemetry = cdp_res["telemetry"]
        assert telemetry["consoleErrors"] == []
        assert telemetry["pageExceptions"] == []
        assert telemetry["externalRequests"] == []
        assert telemetry["httpErrors"] == []
        assert telemetry["failedRequests"] == []


def test_scenario_2_h2h_odds_swap_and_31_feature_matrix(live_server: dict[str, Any]) -> None:
    """Scenario 2: Custom H2H + Power-Method odds + Swap symmetry + 31-Feature Matrix in Chromium."""
    cdp_res = run_cdp_scenario(live_server["base_url"], "h2h_odds_swap_matrix")
    assert cdp_res["ok"] is True
    assert cdp_res["errors"] == []

    data = cdp_res["data"]
    assert "Alcaraz" in data["p1Initial"]
    assert "Sinner" in data["p2Initial"]
    assert data["featureRows"] >= 31
    assert data["arcade"]["ovrCount"] >= 2
    assert data["arcade"]["hasRadarGeometry"] is True
    assert data["arcade"]["radarAxesMatched"] >= 6
    assert data["arcade"]["hasTreeShap"] is True

    before_pair = data["beforePair"]
    after_pair = data["afterPair"]
    assert len(before_pair) == 2 and len(after_pair) == 2
    assert before_pair[0] + before_pair[1] == pytest.approx(100.0, abs=0.5)
    assert after_pair[0] + before_pair[0] == pytest.approx(100.0, abs=1.2)


def test_scenario_3_alltime_legacy_and_divergence_gauge(live_server: dict[str, Any]) -> None:
    """Scenario 3: All-Time roster unlock, Federer vs Nadal Clay H2H & Sinner vs Djokovic divergence."""
    cdp_res = run_cdp_scenario(live_server["base_url"], "alltime_legacy_and_divergence")
    assert cdp_res["ok"] is True
    assert cdp_res["errors"] == []

    data = cdp_res["data"]
    assert data["expandedCount"] > data["initialOptionCount"]
    assert data["expandedCount"] > 1000
    assert data["divergenceFound"] is True


def test_scenario_4_historical_backtracker_cascading_flow(live_server: dict[str, Any]) -> None:
    """Scenario 4: Historical Backtracker Split -> Tournament -> Matchup -> Inspect in Chromium."""
    cdp_res = run_cdp_scenario(live_server["base_url"], "historical_backtracker_flow")
    assert cdp_res["ok"] is True
    assert cdp_res["errors"] == []

    data = cdp_res["data"]
    assert data["featureRows"] >= 31
    assert data["arcade"]["ovrCount"] >= 2
    assert data["arcade"]["hasRadarGeometry"] is True
    assert data["arcade"]["hasTreeShap"] is True


def test_scenario_5_daily_schedule_refresh_and_filters(live_server: dict[str, Any]) -> None:
    """Scenario 5: Daily Schedule Board, Refresh button, Series/Edge filters & ticker in Chromium."""
    cdp_res = run_cdp_scenario(live_server["base_url"], "daily_schedule_filters_and_ticker")
    assert cdp_res["ok"] is True
    assert cdp_res["errors"] == []

    data = cdp_res.get("data") or {}
    telemetry = cdp_res.get("telemetry") or {}
    tested_filters = data.get("testedFilters")
    if tested_filters is None:
        tested_filters = telemetry.get("testedFilters", [])
    assert isinstance(tested_filters, list) and len(tested_filters) >= 1, (
        f"Expected at least 1 schedule filter pill/toggle tested in Scenario 5, got {tested_filters!r}"
    )
    assert telemetry.get("consoleErrors") == []
    assert telemetry.get("pageExceptions") == []
    assert telemetry.get("externalRequests") == []
    assert telemetry.get("httpErrors") == []
    assert telemetry.get("failedRequests") == []

    with httpx.Client(base_url=live_server["base_url"], trust_env=False, timeout=15.0) as client:
        sched_resp = client.get("/api/v1/schedule/live")
        assert sched_resp.status_code == 200, sched_resp.text
        sched = sched_resp.json()
        assert "kpis" in sched and "fixtures" in sched


def test_scenario_6_multi_client_concurrent_live_http_load(live_server: dict[str, Any]) -> None:
    """Scenario 6: 12 concurrent live HTTP roundtrips with 100% 200 OK, zero thread leaks & p95 < 2.0s."""
    base_url = live_server["base_url"]

    # Ensure server is warm before measuring concurrent warm latency
    with httpx.Client(base_url=base_url, trust_env=False, timeout=15.0) as warm_client:
        assert warm_client.get("/healthz").status_code == 200

    tasks: list[tuple[str, str, dict[str, Any] | None]] = [
        ("GET", "/healthz", None),
        ("GET", "/healthz", None),
        ("GET", "/api/v1/meta", None),
        ("GET", "/api/v1/meta", None),
        (
            "POST",
            "/api/v1/predict",
            {"p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam", "odd_1": 1.85, "odd_2": 2.05},
        ),
        (
            "POST",
            "/api/v1/predict",
            {"p1": "Sinner J.", "p2": "Alcaraz C.", "surface": "Hard", "series": "Grand Slam", "odd_1": 2.05, "odd_2": 1.85},
        ),
        (
            "POST",
            "/api/v1/predict",
            {"p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam", "odd_1": 1.85, "odd_2": 2.05},
        ),
        (
            "POST",
            "/api/v1/predict",
            {"p1": "Sinner J.", "p2": "Alcaraz C.", "surface": "Hard", "series": "Grand Slam", "odd_1": 2.05, "odd_2": 1.85},
        ),
        ("GET", "/api/v1/fixtures/historical", None),
        ("GET", "/api/v1/fixtures/historical", None),
        ("GET", "/api/v1/schedule/live", None),
        ("GET", "/api/v1/schedule/live", None),
    ]

    def _execute(idx_and_spec: tuple[int, tuple[str, str, dict[str, Any] | None]]) -> tuple[int, int, dict[str, Any], float]:
        idx, (method, path, payload) = idx_and_spec
        t0 = time.perf_counter()
        with httpx.Client(base_url=base_url, trust_env=False, timeout=20.0) as cli:
            if method == "GET":
                resp = cli.get(path)
            else:
                resp = cli.post(path, json=payload)
        elapsed = time.perf_counter() - t0
        return idx, resp.status_code, resp.json(), elapsed

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(_execute, enumerate(tasks)))

    results.sort(key=lambda r: r[0])
    latencies: list[float] = []
    for _, status, body, elapsed in results:
        assert status == 200, f"Unexpected HTTP {status}: {body}"
        latencies.append(elapsed)

    # Verify forward + swapped prediction symmetry pairs did not suffer thread-safety leaks
    for fwd_idx, rev_idx in ((4, 5), (6, 7)):
        fwd_body = results[fwd_idx][2]
        rev_body = results[rev_idx][2]
        assert fwd_body["p1"] == "Alcaraz C." and rev_body["p1"] == "Sinner J."
        assert fwd_body["prob_lgb"] + rev_body["prob_lgb"] == pytest.approx(1.0, abs=1e-5)

    sorted_lat = sorted(latencies)
    p95_idx = max(0, math.ceil(0.95 * len(sorted_lat)) - 1)
    p95_latency = sorted_lat[p95_idx]
    assert p95_latency < 2.0, f"p95 warm latency {p95_latency:.3f}s exceeded 2.0s (all={sorted_lat})"
