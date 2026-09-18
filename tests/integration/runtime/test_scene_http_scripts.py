from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from market_support_crewai_agent.schemas.base import StrictModel
from market_support_crewai_agent.schemas.reply import ReplyResponse
from scripts.check_scene_http_contract import CheckResult

ROOT = Path(__file__).resolve().parents[3]
SERVE = ROOT / "scripts" / "serve_reply_fake_deps.py"
CHECK = ROOT / "scripts" / "check_scene_http_contract.py"
API_KEY = "fixture-service-key"
ADAPTER_KEY = "fixture-adapter-key"
TENANT = "tenant:http-test"
TCP_ADDRESS = TypeAdapter(tuple[str, int])


class _ErrorDetail(StrictModel):
    code: str
    message: str


class _ErrorResponse(StrictModel):
    detail: _ErrorDetail


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        _host, port = TCP_ADDRESS.validate_python(listener.getsockname())
        return port


def _is_open(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.1)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _wait_for_port(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if _is_open(port):
            return
        if process.poll() is not None:
            stdout, _ = process.communicate(timeout=1)
            pytest.fail(f"server exited before bind: {stdout}")
        time.sleep(0.02)
    pytest.fail(f"server did not bind loopback port {port}")


@contextmanager
def _server(mode: str, tmp_path: Path) -> Generator[tuple[str, int, int]]:
    port, adapter_port = _free_port(), _free_port()
    while adapter_port == port:
        adapter_port = _free_port()
    command = [
        sys.executable,
        str(SERVE),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--adapter-port",
        str(adapter_port),
        "--api-key",
        API_KEY,
        "--adapter-api-key",
        ADAPTER_KEY,
        "--tenant-ref",
        TENANT,
        "--internal-dm-enabled",
        "true",
        "--mode",
        mode,
    ]
    log_path = tmp_path / f"{mode}.server.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(  # noqa: S603 - fixed local test command.
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_for_port(port, process)
            yield f"http://127.0.0.1:{port}", port, process.pid
        finally:
            process.terminate()
            try:
                _ = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                _ = process.wait(timeout=5)
    assert process.poll() is not None
    assert not _is_open(port)
    assert not _is_open(adapter_port)


def _run_checker(base_url: str, output: Path, mode: str = "compatible") -> None:
    command = [
        sys.executable,
        str(CHECK),
        "--base-url",
        base_url,
        "--api-key",
        API_KEY,
        "--tenant-ref",
        TENANT,
    ]
    if mode == "compatible":
        command.extend(("--output", str(output)))
    else:
        command.extend(("--expect-mode", mode, "--output", str(output)))
    result = subprocess.run(  # noqa: S603 - fixed local test command.
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.integration
def test_compatible_checker_uses_real_tcp_and_records_zero_send(
    tmp_path: Path,
) -> None:
    # Given: a fresh production-app process with loopback fake dependencies.
    output = tmp_path / "compatible.json"

    # When: the standalone checker drives the bound TCP socket.
    with _server("compatible", tmp_path) as (base_url, port, pid):
        assert _is_open(port)
        _run_checker(base_url, output)
        assert pid > 0

    # Then: the artifact proves health compatibility and all required scenarios.
    result = CheckResult.model_validate_json(output.read_text(encoding="utf-8"))
    cases = {case.name: case for case in result.cases}
    assert cases["health"].raw_response == (
        '{"status":"ok","service":"market-support-crewai-agent"}'
    )
    group_response = ReplyResponse.model_validate(cases["group_action"].response)
    assert group_response.actions[0].type == "send_weekly_report"
    direct_response = ReplyResponse.model_validate(cases["direct_knowledge"].response)
    assert direct_response.actions == []
    assert cases["late_feedback"].status == 200
    assert cases["feedback_mismatch"].status == 409
    assert result.summary.send_spy_count == 0
    assert result.summary.zero_send is True


@pytest.mark.integration
@pytest.mark.parametrize(
    ("mode", "code"),
    [
        ("tenant-unconfigured", "deployment_identity_unavailable"),
        ("dm-disabled", "internal_dm_disabled"),
        ("adapter-missing-fields", "internal_dm_adapter_incompatible"),
        ("adapter-wrong-scene", "internal_dm_adapter_incompatible"),
        ("adapter-wrong-tenant", "internal_dm_adapter_incompatible"),
        ("adapter-wrong-version", "internal_dm_adapter_incompatible"),
    ],
)
def test_closed_mode_restarts_fail_before_state(
    mode: str,
    code: str,
    tmp_path: Path,
) -> None:
    # Given/When: each immutable mode starts in its own process.
    output = tmp_path / f"{mode}.json"
    with _server(mode, tmp_path) as (base_url, _port, _pid):
        _run_checker(base_url, output, mode)

    # Then: admission fails before reservation, LLM, knowledge, or send work.
    result = CheckResult.model_validate_json(output.read_text(encoding="utf-8"))
    case = result.cases[0]
    error_response = _ErrorResponse.model_validate(case.response)
    assert case.status == 503
    assert error_response.detail.code == code
    assert case.counters.issued_count == 0
    assert case.counters.planner_count == 0
    assert case.counters.composer_count == 0
    assert case.counters.send_spy_count == 0


def test_serve_cli_rejects_non_loopback_and_unknown_mode_before_bind() -> None:
    # Given/When: malformed startup values are passed without starting a child.
    common = [
        sys.executable,
        str(SERVE),
        "--port",
        "18080",
        "--adapter-port",
        "18081",
        "--api-key",
        API_KEY,
        "--adapter-api-key",
        ADAPTER_KEY,
        "--tenant-ref",
        TENANT,
        "--internal-dm-enabled",
        "true",
    ]
    bad_host = subprocess.run(  # noqa: S603 - fixed local test command.
        [*common, "--host", "0.0.0.0", "--mode", "compatible"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    bad_mode = subprocess.run(  # noqa: S603 - fixed local test command.
        [*common, "--host", "127.0.0.1", "--mode", "mutable-control"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    # Then: both inputs fail closed and neither fixed proof port is listening.
    assert bad_host.returncode != 0
    assert "loopback" in (bad_host.stdout + bad_host.stderr).lower()
    assert bad_mode.returncode != 0
    assert not _is_open(18080)
    assert not _is_open(18081)
