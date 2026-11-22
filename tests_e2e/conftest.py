"""Shared E2E fixtures, server lifecycle helpers, and opaque-box math/contract oracles.

IMPORTANT: Never import `server` at module top-level so `pytest tests_e2e --collect-only`
succeeds cleanly before `server.py` exists.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator

import httpx
import pytest


TESTS_E2E_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_E2E_DIR.parent
CDP_RUNNER_PATH = TESTS_E2E_DIR / "cdp_runner.mjs"

RADAR_AXES = (
    "short_form",
    "med_form",
    "surface_mastery",
    "rank_power",
    "clutch_tb",
    "stamina_freshness",
)

EXPECTED_31_FEATURES = (
    "rank_diff",
    "pts_diff",
    "short_winrate_diff",
    "short_tb_winrate_diff",
    "short_comeback_rate_diff",
    "short_bb_rate_diff",
    "short_avg_games_per_set_diff",
    "med_winrate_diff",
    "med_tb_winrate_diff",
    "med_comeback_rate_diff",
    "med_bb_rate_diff",
    "med_avg_games_per_set_diff",
    "med_ret_freq_diff",
    "career_winrate_diff",
    "career_tb_winrate_diff",
    "career_comeback_rate_diff",
    "career_bb_rate_diff",
    "career_avg_games_per_set_diff",
    "career_ret_freq_diff",
    "surface_winrate_diff",
    "days_since_last_diff",
    "matches_14d_diff",
    "matches_30d_diff",
    "form_divergence_diff",
    "h2h_winrate",
    "h2h_match_count",
    "h2h_surface_winrate",
    "h2h_surface_count",
    "surface_encoded",
    "series_encoded",
    "is_grand_slam",
)

_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "NODE_USE_ENV_PROXY",
)


def _strip_proxy_env(env: dict[str, str]) -> dict[str, str]:
    """Remove proxy env vars and disable proxy routing for localhost traffic."""
    for key in _PROXY_ENV_KEYS:
        env.pop(key, None)
    env["NO_PROXY"] = ""
    env["no_proxy"] = ""
    return env


def pytest_configure(config: pytest.Config) -> None:
    """Register E2E tier markers."""
    config.addinivalue_line("markers", "tier1: Tier 1 API, schema, and mathematical oracle tests")
    config.addinivalue_line("markers", "tier2: Tier 2 state, persistence, and integration tests")
    config.addinivalue_line("markers", "tier3: Tier 3 real-process HTTP, startup, and schedule tests")
    config.addinivalue_line("markers", "tier4: Tier 4 headless Chromium CDP browser E2E tests")


def free_port() -> int:
    """Allocate an ephemeral free TCP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_stream_file(handle: Any) -> str:
    """Rewind and read a temporary stdout/stderr file safely."""
    try:
        handle.flush()
        handle.seek(0)
        return handle.read()
    except Exception as exc:  # pragma: no cover - defensive cleanup
        return f"<unable to read stream: {exc}>"


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    """Gracefully terminate a spawned process with SIGKILL fallback."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _build_server_env(port: int, env_overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Construct clean subprocess environment for Uvicorn/FastAPI."""
    env = _strip_proxy_env(os.environ.copy())
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_str = str(REPO_ROOT)
    if existing_pythonpath:
        if repo_str not in existing_pythonpath.split(os.pathsep):
            env["PYTHONPATH"] = f"{repo_str}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = repo_str

    env["PORT"] = str(port)
    if env_overrides:
        for k, v in env_overrides.items():
            env[str(k)] = str(v)
    return env


@contextmanager
def start_ephemeral_server(
    port: int | None = None,
    env_overrides: dict[str, str] | None = None,
    startup_timeout: float = 25.0,
    use_python_entrypoint: bool = False,
) -> Iterator[dict[str, Any]]:
    """Launch a fresh uvicorn server:app instance and tear it down cleanly.

    Supports verifying custom `$PORT` env binding or custom environment overrides.
    Raises RuntimeError with captured stdout/stderr if startup fails or times out.
    """
    if port is not None:
        actual_port = int(port)
    elif env_overrides and "PORT" in env_overrides:
        actual_port = int(env_overrides["PORT"])
    else:
        actual_port = free_port()

    server_py = REPO_ROOT / "server.py"
    if not server_py.is_file():
        raise RuntimeError(f"server.py not found at {server_py}")

    env = _build_server_env(actual_port, env_overrides=env_overrides)
    if use_python_entrypoint:
        cmd = [sys.executable, str(server_py)]
    else:
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(actual_port),
        ]

    stdout_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
    stderr_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=stdout_file,
        stderr=stderr_file,
        text=True,
    )

    base_url = f"http://127.0.0.1:{actual_port}"
    health_url = f"{base_url}/healthz"
    deadline = time.monotonic() + startup_timeout
    last_error: str = "no response yet"

    try:
        with httpx.Client(trust_env=False, timeout=2.0) as client:
            while time.monotonic() < deadline:
                exit_code = proc.poll()
                if exit_code is not None:
                    stdout_text = _read_stream_file(stdout_file)
                    stderr_text = _read_stream_file(stderr_file)
                    raise RuntimeError(
                        f"Server process exited prematurely with code {exit_code} "
                        f"while polling {health_url}.\n"
                        f"CMD: {' '.join(cmd)}\n"
                        f"STDOUT:\n{stdout_text}\nSTDERR:\n{stderr_text}"
                    )
                try:
                    resp = client.get(health_url)
                    if resp.status_code == 200:
                        break
                    last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                except httpx.RequestError as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.2)
            else:
                stdout_text = _read_stream_file(stdout_file)
                stderr_text = _read_stream_file(stderr_file)
                raise RuntimeError(
                    f"Server did not become healthy at {health_url} within {startup_timeout:.1f}s "
                    f"(last error: {last_error}).\n"
                    f"CMD: {' '.join(cmd)}\n"
                    f"STDOUT:\n{stdout_text}\nSTDERR:\n{stderr_text}"
                )

        yield {
            "base_url": base_url,
            "port": actual_port,
            "proc": proc,
        }
    finally:
        _terminate_process(proc)
        stdout_file.close()
        stderr_file.close()


@pytest.fixture(scope="session")
def live_server() -> Iterator[dict[str, Any]]:
    """Session-scoped live FastAPI/Uvicorn server bound to an ephemeral port."""
    try:
        with start_ephemeral_server(startup_timeout=25.0) as server_info:
            yield server_info
    except RuntimeError as exc:
        pytest.fail(str(exc), pytrace=False)


@pytest.fixture()
def http_client(live_server: dict[str, Any]) -> Iterator[httpx.Client]:
    """HTTPX client bound to the live session server with proxy env disabled."""
    with httpx.Client(
        base_url=live_server["base_url"],
        trust_env=False,
        timeout=20.0,
    ) as client:
        yield client


def create_inprocess_client(monkeypatch: pytest.MonkeyPatch | None = None):
    """Lazily import `server` and return a FastAPI TestClient for isolated tests."""
    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    if monkeypatch is not None:
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("no_proxy", "")

    from fastapi.testclient import TestClient

    server_mod = importlib.import_module("server")
    return TestClient(server_mod.app)


# ---------------------------------------------------------------------------
# Opaque-box mathematical & contract oracle helpers
# ---------------------------------------------------------------------------

def solve_power_devig(odd_1: float, odd_2: float) -> tuple[float, float, float, float]:
    """Solve (1/odd_1)**(1/k) + (1/odd_2)**(1/k) == 1.0 via bisection (< 1e-9).

    Returns (implied_p1, implied_p2, overround, power_k).
    """
    if odd_1 <= 1.0 or odd_2 <= 1.0:
        raise ValueError(f"Decimal odds must be > 1.0, got odd_1={odd_1}, odd_2={odd_2}")

    q1 = 1.0 / float(odd_1)
    q2 = 1.0 / float(odd_2)
    overround = q1 + q2

    def f(k: float) -> float:
        inv_k = 1.0 / k
        return (q1 ** inv_k) + (q2 ** inv_k)

    lo = 1e-6
    hi = 100.0
    while f(lo) > 1.0:
        lo *= 0.5
    while f(hi) < 1.0:
        hi *= 2.0

    k = 1.0
    for _ in range(150):
        mid = 0.5 * (lo + hi)
        val = f(mid)
        k = mid
        if abs(val - 1.0) < 1e-12 or (hi - lo) < 1e-12:
            break
        if val < 1.0:
            lo = mid
        else:
            hi = mid

    inv_k = 1.0 / k
    implied_p1 = q1 ** inv_k
    implied_p2 = q2 ** inv_k
    return implied_p1, implied_p2, overround, k


def compute_quarter_kelly_pct(prob: float, odd: float) -> float:
    """Compute Quarter-Kelly bankroll stake % capped at 2.5%."""
    if odd <= 1.0:
        return 0.0
    raw = 0.25 * ((odd - 1.0) * prob - (1.0 - prob)) / (odd - 1.0) * 100.0
    return min(2.5, max(0.0, raw))


def sigmoid(z: float) -> float:
    """Numerically stable logistic sigmoid function."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _is_finite_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def assert_valid_player_card(card: dict[str, Any], expected_name: str | None = None) -> None:
    """Validate an Arcade player card structure and domain bounds."""
    assert isinstance(card, dict), f"player card must be a dict, got {type(card)}"
    name = card.get("name")
    assert isinstance(name, str) and name.strip(), f"invalid card name: {name!r}"
    if expected_name is not None:
        assert name == expected_name, f"expected card name {expected_name!r}, got {name!r}"

    ovr = card.get("ovr")
    assert _is_finite_number(ovr) and 0 <= ovr <= 99, f"invalid ovr: {ovr!r}"

    rank = card.get("rank")
    assert _is_finite_number(rank) and rank >= 0, f"invalid rank: {rank!r}"

    pts = card.get("pts")
    assert _is_finite_number(pts) and pts >= 0, f"invalid pts: {pts!r}"

    for rate_key in ("surface_winrate", "short_winrate", "career_winrate"):
        val = card.get(rate_key)
        assert _is_finite_number(val) and 0.0 <= float(val) <= 1.0, (
            f"invalid {rate_key}={val!r}; expected [0.0, 1.0]"
        )

    days_since_last = card.get("days_since_last")
    assert _is_finite_number(days_since_last) and days_since_last >= 0, (
        f"invalid days_since_last={days_since_last!r}"
    )

    matches_14d = card.get("matches_14d")
    assert _is_finite_number(matches_14d) and matches_14d >= 0, (
        f"invalid matches_14d={matches_14d!r}"
    )

    traits = card.get("traits")
    assert isinstance(traits, list) and all(isinstance(t, str) and t.strip() for t in traits), (
        f"invalid traits: {traits!r}"
    )

    radar = card.get("radar")
    assert isinstance(radar, dict), f"radar must be dict, got {type(radar)}"
    for axis in RADAR_AXES:
        axis_val = radar.get(axis)
        assert _is_finite_number(axis_val) and 0.0 <= float(axis_val) <= 100.0, (
            f"invalid radar[{axis!r}]={axis_val!r}; expected [0, 100]"
        )


def assert_valid_treeshap(
    treeshap: dict[str, Any],
    require_nonzero_shap: bool = True,
) -> None:
    """Validate TreeSHAP tug-of-war contribution payload against the 31-feature contract."""
    assert isinstance(treeshap, dict), f"treeshap must be dict, got {type(treeshap)}"
    assert _is_finite_number(treeshap.get("bias")), f"invalid bias: {treeshap.get('bias')!r}"
    assert _is_finite_number(treeshap.get("p1_total_contrib")), (
        f"invalid p1_total_contrib: {treeshap.get('p1_total_contrib')!r}"
    )

    contributions = treeshap.get("contributions")
    assert isinstance(contributions, list) and len(contributions) > 0, (
        f"treeshap.contributions must be non-empty list, got {contributions!r}"
    )
    assert len(contributions) <= len(EXPECTED_31_FEATURES), (
        f"treeshap.contributions exceeded 31 features: {len(contributions)}"
    )

    expected_set = set(EXPECTED_31_FEATURES)
    seen_features: set[str] = set()
    for idx, item in enumerate(contributions):
        assert isinstance(item, dict), f"contribution[{idx}] must be dict"
        for text_key in ("feature", "label", "category"):
            val = item.get(text_key)
            assert isinstance(val, str) and val.strip(), (
                f"contribution[{idx}].{text_key} invalid: {val!r}"
            )
        feat_name = item["feature"]
        assert feat_name in expected_set, (
            f"contribution[{idx}].feature {feat_name!r} is not one of EXPECTED_31_FEATURES"
        )
        assert feat_name not in seen_features, (
            f"duplicate feature {feat_name!r} in treeshap.contributions"
        )
        seen_features.add(feat_name)

        assert _is_finite_number(item.get("raw_delta")), (
            f"contribution[{idx}].raw_delta invalid: {item.get('raw_delta')!r}"
        )
        assert _is_finite_number(item.get("shap")), (
            f"contribution[{idx}].shap invalid: {item.get('shap')!r}"
        )
        assert item.get("favors") in ("p1", "p2", "neutral"), (
            f"contribution[{idx}].favors invalid: {item.get('favors')!r}"
        )

    if len(contributions) == len(EXPECTED_31_FEATURES):
        shap_sum = sum(float(c["shap"]) for c in contributions)
        assert shap_sum == pytest.approx(float(treeshap["p1_total_contrib"]), abs=1e-4)
    else:
        assert len(contributions) >= 5, (
            f"Expected at least 5 top-K contributions when < 31, got {len(contributions)}"
        )
        if require_nonzero_shap:
            assert any(abs(float(c["shap"])) > 0.0 for c in contributions), (
                "Expected non-zero SHAP magnitudes in top-K contributions"
            )


def assert_valid_features_31(features_31: list[dict[str, Any]]) -> None:
    """Validate full 31-feature matrix payload against exact model feature schema."""
    assert isinstance(features_31, list), f"features_31 must be list, got {type(features_31)}"
    assert len(features_31) == 31, f"expected 31 features, got {len(features_31)}"

    seen_names: set[str] = set()
    for idx, feat in enumerate(features_31):
        assert isinstance(feat, dict), f"features_31[{idx}] must be dict"
        for text_key in ("name", "label", "category"):
            val = feat.get(text_key)
            assert isinstance(val, str) and val.strip(), (
                f"features_31[{idx}].{text_key} invalid: {val!r}"
            )
        seen_names.add(feat["name"])
        assert _is_finite_number(feat.get("value")), (
            f"features_31[{idx}].value invalid: {feat.get('value')!r}"
        )
        assert _is_finite_number(feat.get("shap")), (
            f"features_31[{idx}].shap invalid: {feat.get('shap')!r}"
        )
        assert _is_finite_number(feat.get("lr_coef")), (
            f"features_31[{idx}].lr_coef invalid: {feat.get('lr_coef')!r}"
        )
        assert feat.get("favors") in ("p1", "p2", "neutral"), (
            f"features_31[{idx}].favors invalid: {feat.get('favors')!r}"
        )

    assert seen_names == set(EXPECTED_31_FEATURES), (
        f"features_31 names mismatch; missing={set(EXPECTED_31_FEATURES) - seen_names}, "
        f"unexpected={seen_names - set(EXPECTED_31_FEATURES)}"
    )
    assert any(abs(float(f["lr_coef"])) > 0.0 for f in features_31), (
        "Expected at least one non-zero lr_coef in features_31"
    )


# ---------------------------------------------------------------------------
# Headless Chromium CDP runner bridge
# ---------------------------------------------------------------------------

def run_cdp_scenario(
    base_url: str,
    scenario: str,
    viewport: tuple[int, int] = (1440, 900),
    timeout: int = 60,
) -> dict[str, Any]:
    """Execute a headless Chromium CDP scenario via Node v24 cdp_runner.mjs."""
    width, height = int(viewport[0]), int(viewport[1])
    cmd = [
        "node",
        str(CDP_RUNNER_PATH),
        "--base-url",
        base_url,
        "--scenario",
        scenario,
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    env = _strip_proxy_env(os.environ.copy())

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"CDP scenario {scenario!r} timed out after {timeout}s.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{exc.stdout or ''}\nSTDERR:\n{exc.stderr or ''}"
        ) from exc

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    parsed: dict[str, Any] | None = None
    if stdout:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            for line in reversed(stdout.splitlines()):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        parsed = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue

    if proc.returncode != 0 or not isinstance(parsed, dict) or parsed.get("ok") is not True:
        raise AssertionError(
            f"CDP scenario {scenario!r} failed (exit={proc.returncode}).\n"
            f"PARSED: {json.dumps(parsed, indent=2) if parsed is not None else '<unparseable>'}\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )

    return parsed
