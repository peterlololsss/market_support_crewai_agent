from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import pytest

from market_support_crewai_agent.runtime.prompts.direct_provider_io_capture import (
    DirectProviderIoCaptureError,
    DirectProviderIoCaptureV1,
)


@dataclass(frozen=True)
class _ProcessSnapshot:
    stdout: object
    stderr: object
    record_factory: object
    root_handlers: tuple[logging.Handler, ...]
    handler_filters: tuple[tuple[logging.Filter, ...], ...]
    threads: frozenset[threading.Thread]


def test_clean_capture_restores_every_process_resource() -> None:
    fd1 = os.dup(1)
    fd2 = os.dup(2)
    snapshot = _snapshot()
    try:
        with DirectProviderIoCaptureV1() as capture:
            assert capture.phase == "installed"
        assert capture.phase == "closed"
        assert capture.stdout_byte_count == 0
        assert capture.stderr_byte_count == 0
        _assert_restored(snapshot, fd1, fd2)
    finally:
        os.close(fd1)
        os.close(fd2)


def test_stdout_stderr_fd_and_cross_thread_logs_fail_closed_without_raw_state() -> None:
    class RecordHandler(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    handler = RecordHandler()
    root = logging.getLogger()
    root.addHandler(handler)
    snapshot = _snapshot()
    fd1 = os.dup(1)
    fd2 = os.dup(2)
    try:
        with pytest.raises(
            DirectProviderIoCaptureError,
            match="^direct_provider_stdio_violation$",
        ):
            with DirectProviderIoCaptureV1() as capture:
                print("stdout-canary")
                print("stderr-canary", file=sys.stderr)
                os.write(1, b"fd-canary")
                logger_thread = threading.Thread(
                    target=logging.getLogger("provider.test").error,
                    args=("logger-canary",),
                )
                logger_thread.start()
                logger_thread.join(timeout=1)

        assert not hasattr(capture, "captured_text")
        public_state = repr(vars(capture))
        for canary in (
            "stdout-canary",
            "stderr-canary",
            "fd-canary",
            "logger-canary",
        ):
            assert canary not in public_state
        assert handler.records
        assert handler.records[-1].getMessage() == "direct_provider_log_suppressed"
        assert capture.stdout_byte_count > 0
        assert capture.stderr_byte_count > 0
        _assert_restored(snapshot, fd1, fd2)
    finally:
        root.removeHandler(handler)
        os.close(fd1)
        os.close(fd2)


def test_provider_exception_restores_resources_and_remains_primary_when_clean() -> None:
    snapshot = _snapshot()
    fd1 = os.dup(1)
    fd2 = os.dup(2)
    try:
        with pytest.raises(RuntimeError, match="provider failed"):
            with DirectProviderIoCaptureV1() as capture:
                raise RuntimeError("provider failed")
        assert "provider failed" not in repr(vars(capture))
        _assert_restored(snapshot, fd1, fd2)
    finally:
        os.close(fd1)
        os.close(fd2)


def test_overflow_saturates_count_and_fails_closed() -> None:
    sentinel = b"overflow-secret-sentinel"
    payload = (sentinel * 3_000)[:70_000]
    with pytest.raises(
        DirectProviderIoCaptureError,
        match="^direct_provider_stdio_violation$",
    ):
        with DirectProviderIoCaptureV1() as capture:
            _write_all(1, payload)

    assert capture.stdout_byte_count == 65_537
    assert capture.stdout_overflow is True
    assert sentinel.decode() not in repr(vars(capture))


def test_recovered_dup2_fault_restores_equality_but_reports_violation() -> None:
    snapshot = _snapshot()
    fd1 = os.dup(1)
    fd2 = os.dup(2)
    try:
        with pytest.raises(
            DirectProviderIoCaptureError,
            match="^direct_provider_stdio_violation$",
        ):
            with DirectProviderIoCaptureV1() as capture:
                real_dup2 = capture._restore_dup2
                failed = False

                def flaky_dup2(source: int, target: int) -> int:
                    nonlocal failed
                    if target == 1 and not failed:
                        failed = True
                        raise OSError("injected restoration fault")
                    return real_dup2(source, target)

                capture._restore_dup2 = flaky_dup2

        assert capture.restoration_recovered is True
        assert capture.restoration_attempts_fd1 == 2
        assert capture.restoration_attempts_fd2 == 1
        _assert_restored(snapshot, fd1, fd2)
    finally:
        os.close(fd1)
        os.close(fd2)


def test_inherited_writer_timeout_is_a_violation_and_leaves_no_drainer() -> None:
    inherited_writer = -1
    with pytest.raises(
        DirectProviderIoCaptureError,
        match="^direct_provider_stdio_violation$",
    ):
        with DirectProviderIoCaptureV1() as capture:
            inherited_writer = os.dup(1)
    try:
        assert capture.stdout_join_timed_out is True
        assert capture.stdout_thread_alive is False
        assert capture.stderr_thread_alive is False
    finally:
        if inherited_writer >= 0:
            os.close(inherited_writer)


def test_unrecoverable_restoration_exits_seventy_in_isolated_subprocess() -> None:
    script = (
        "from market_support_crewai_agent.runtime.prompts.direct_provider_io_capture "
        "import DirectProviderIoCaptureV1\n"
        "capture = DirectProviderIoCaptureV1()\n"
        "capture.install()\n"
        "def fail(source, target):\n"
        "    raise OSError('unrecoverable')\n"
        "capture._restore_dup2 = fail\n"
        "capture.close()\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 70


def _snapshot() -> _ProcessSnapshot:
    root = logging.getLogger()
    return _ProcessSnapshot(
        stdout=sys.stdout,
        stderr=sys.stderr,
        record_factory=logging.getLogRecordFactory(),
        root_handlers=tuple(root.handlers),
        handler_filters=tuple(tuple(handler.filters) for handler in root.handlers),
        threads=frozenset(threading.enumerate()),
    )


def _assert_restored(snapshot: _ProcessSnapshot, fd1: int, fd2: int) -> None:
    root = logging.getLogger()
    assert os.path.sameopenfile(1, fd1)
    assert os.path.sameopenfile(2, fd2)
    assert sys.stdout is snapshot.stdout
    assert sys.stderr is snapshot.stderr
    assert logging.getLogRecordFactory() is snapshot.record_factory
    assert tuple(root.handlers) == snapshot.root_handlers
    assert tuple(tuple(handler.filters) for handler in root.handlers) == (
        snapshot.handler_filters
    )
    assert frozenset(threading.enumerate()) == snapshot.threads


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        try:
            written = os.write(fd, view)
        except BlockingIOError:
            time.sleep(0.001)
            continue
        view = view[written:]
