from __future__ import annotations

# noqa: SIZE_OK - one process-global fd/log capture state machine and async gate.

import asyncio
import io
import logging
import os
import select
import selectors
import sys
import threading
from types import TracebackType
from typing import Callable, Literal, Self

from market_support_crewai_agent.runtime.prompts.program_models import (
    DirectProviderFatalStateV1,
)


DirectProviderIoCapturePhaseV1 = Literal[
    "created",
    "installed",
    "restoring",
    "closed",
]

_MAX_CAPTURE_BYTES = 65_536
_SATURATED_CAPTURE_BYTES = 65_537
_DRAIN_POLL_SECONDS = 0.05
_DRAIN_JOIN_SECONDS = 1.0
_PROVIDER_GATE = threading.Lock()
_DIRECT_LOG_LOCK = threading.Lock()
_DIRECT_LOG_ACTIVE = False
_STANDARD_LOG_FIELDS = frozenset(logging.makeLogRecord({}).__dict__)


class DirectProviderIoCaptureError(ValueError):
    pass


class _DirectProviderLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if _direct_log_active():
            _redact_log_record(record)
        return True


class _FdBinarySink(io.RawIOBase):
    def __init__(self, fd: int) -> None:
        self._fd = fd

    def writable(self) -> bool:
        return True

    def write(self, data: bytes | bytearray) -> int:
        payload = bytes(data)
        _write_fd_all(self._fd, payload)
        return len(payload)

    def fileno(self) -> int:
        return self._fd


class _FdTextSink(io.TextIOBase):
    def __init__(self, fd: int, template) -> None:
        self._fd = fd
        self._encoding = str(getattr(template, "encoding", None) or "utf-8")
        self._errors = str(getattr(template, "errors", None) or "strict")
        self.buffer = _FdBinarySink(fd)

    @property
    def encoding(self) -> str:
        return self._encoding

    @property
    def errors(self) -> str:
        return self._errors

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("write() argument must be str")
        _write_fd_all(self._fd, text.encode(self._encoding, self._errors))
        return len(text)

    def fileno(self) -> int:
        return self._fd

    def isatty(self) -> bool:
        return False


class DirectProviderIoCaptureV1:
    def __init__(self) -> None:
        self.phase: DirectProviderIoCapturePhaseV1 = "created"
        self.saved_fd1_primary: int | None = None
        self.saved_fd1_emergency: int | None = None
        self.saved_fd2_primary: int | None = None
        self.saved_fd2_emergency: int | None = None
        self.stdout_read_fd: int | None = None
        self.stdout_write_fd: int | None = None
        self.stderr_read_fd: int | None = None
        self.stderr_write_fd: int | None = None
        self.stdout_drainer: threading.Thread | None = None
        self.stderr_drainer: threading.Thread | None = None
        self.stdout_stop_event = threading.Event()
        self.stderr_stop_event = threading.Event()
        self.stdout_byte_count = 0
        self.stderr_byte_count = 0
        self.stdout_overflow = False
        self.stderr_overflow = False
        self.stdout_thread_alive = False
        self.stderr_thread_alive = False
        self.stdout_join_timed_out = False
        self.stderr_join_timed_out = False
        self.restoration_attempts_fd1 = 0
        self.restoration_attempts_fd2 = 0
        self.restoration_recovered = False
        self.process_fatal = False
        self.restoration_error: str | None = None
        self.fatal_state: DirectProviderFatalStateV1 | None = None
        self._owned_fds: set[int] = set()
        self._count_lock = threading.Lock()
        self._stdout = None
        self._stderr = None
        self._installed_stdout: _FdTextSink | None = None
        self._installed_stderr: _FdTextSink | None = None
        self._fd1_inheritable = True
        self._fd2_inheritable = True
        self._record_factory = None
        self._installed_record_factory = None
        self._root_handlers: tuple[logging.Handler, ...] = ()
        self._handler_filters: tuple[tuple[logging.Filter, ...], ...] = ()
        self._log_filter = _DirectProviderLogFilter()
        self._restore_dup2: Callable[[int, int], int] = os.dup2

    def __enter__(self) -> Self:
        self.install()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        del exc_type, exc, tb
        self.close()
        return False

    def install(self) -> None:
        if self.phase != "created":
            raise DirectProviderIoCaptureError("provider_io_capture_reinstall")
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._fd1_inheritable = os.get_inheritable(1)
        self._fd2_inheritable = os.get_inheritable(2)
        try:
            self.saved_fd1_primary = self._dup_owned(1)
            self.saved_fd1_emergency = self._dup_owned(1)
            self.saved_fd2_primary = self._dup_owned(2)
            self.saved_fd2_emergency = self._dup_owned(2)
            self.stdout_read_fd, self.stdout_write_fd = self._pipe_owned()
            self.stderr_read_fd, self.stderr_write_fd = self._pipe_owned()
            self.stdout_drainer = self._start_drainer("stdout")
            self.stderr_drainer = self._start_drainer("stderr")
            self._install_log_redaction()
            self._flush_python_streams()
            os.dup2(self._required_fd(self.stdout_write_fd), 1)
            os.set_inheritable(1, self._fd1_inheritable)
            os.dup2(self._required_fd(self.stderr_write_fd), 2)
            os.set_inheritable(2, self._fd2_inheritable)
            self._close_owned("stdout_write_fd")
            self._close_owned("stderr_write_fd")
            self._installed_stdout = _FdTextSink(1, self._stdout)
            self._installed_stderr = _FdTextSink(2, self._stderr)
            sys.stdout = self._installed_stdout
            sys.stderr = self._installed_stderr
        except BaseException:
            self.phase = "restoring"
            self._abort_installation()
            raise
        self.phase = "installed"

    def close(self) -> None:
        if self.phase == "closed":
            return
        if self.phase != "installed":
            raise DirectProviderIoCaptureError("provider_io_capture_not_installed")
        self.phase = "restoring"
        self._flush_python_streams()
        fd1_restored = self._restore_target(
            target=1,
            primary=self.saved_fd1_primary,
            emergency=self.saved_fd1_emergency,
            attempts_attr="restoration_attempts_fd1",
            inheritable=self._fd1_inheritable,
        )
        fd2_restored = self._restore_target(
            target=2,
            primary=self.saved_fd2_primary,
            emergency=self.saved_fd2_emergency,
            attempts_attr="restoration_attempts_fd2",
            inheritable=self._fd2_inheritable,
        )
        if not fd1_restored or not fd2_restored:
            self._fatal_exit()
        for name in (
            "saved_fd1_primary",
            "saved_fd1_emergency",
            "saved_fd2_primary",
            "saved_fd2_emergency",
            "stdout_write_fd",
            "stderr_write_fd",
        ):
            self._close_owned(name)
        self._restore_process_globals()
        self._finish_drainer("stdout")
        self._finish_drainer("stderr")
        self.stdout_thread_alive = bool(
            self.stdout_drainer is not None and self.stdout_drainer.is_alive()
        )
        self.stderr_thread_alive = bool(
            self.stderr_drainer is not None and self.stderr_drainer.is_alive()
        )
        self._close_owned("stdout_read_fd")
        self._close_owned("stderr_read_fd")
        self.phase = "closed"
        if self._has_violation():
            raise DirectProviderIoCaptureError("direct_provider_stdio_violation")

    def _dup_owned(self, fd: int) -> int:
        duplicate = os.dup(fd)
        os.set_inheritable(duplicate, False)
        self._owned_fds.add(duplicate)
        return duplicate

    def _pipe_owned(self) -> tuple[int, int]:
        read_fd, write_fd = os.pipe()
        for fd in (read_fd, write_fd):
            os.set_inheritable(fd, False)
            os.set_blocking(fd, False)
            self._owned_fds.add(fd)
        return read_fd, write_fd

    def _start_drainer(self, stream: Literal["stdout", "stderr"]) -> threading.Thread:
        thread = threading.Thread(
            target=self._drain,
            args=(stream,),
            name=f"direct-provider-{stream}-drainer",
            daemon=False,
        )
        thread.start()
        return thread

    def _drain(self, stream: Literal["stdout", "stderr"]) -> None:
        read_fd = self.stdout_read_fd if stream == "stdout" else self.stderr_read_fd
        stop_event = (
            self.stdout_stop_event if stream == "stdout" else self.stderr_stop_event
        )
        if read_fd is None:
            self._mark_drainer_error()
            return
        selector = selectors.DefaultSelector()
        try:
            selector.register(read_fd, selectors.EVENT_READ)
            while not stop_event.is_set():
                for _key, _events in selector.select(_DRAIN_POLL_SECONDS):
                    while not stop_event.is_set():
                        try:
                            chunk = os.read(read_fd, 8192)
                        except BlockingIOError:
                            break
                        except OSError:
                            if not stop_event.is_set():
                                self._mark_drainer_error()
                            return
                        if not chunk:
                            return
                        self._record_bytes(stream, len(chunk))
        finally:
            selector.close()

    def _record_bytes(self, stream: Literal["stdout", "stderr"], size: int) -> None:
        with self._count_lock:
            count_attr = f"{stream}_byte_count"
            overflow_attr = f"{stream}_overflow"
            current = getattr(self, count_attr)
            total = current + size
            if total > _MAX_CAPTURE_BYTES:
                setattr(self, count_attr, _SATURATED_CAPTURE_BYTES)
                setattr(self, overflow_attr, True)
            else:
                setattr(self, count_attr, total)

    def _restore_target(
        self,
        *,
        target: int,
        primary: int | None,
        emergency: int | None,
        attempts_attr: str,
        inheritable: bool,
    ) -> bool:
        for source in (primary, primary, emergency):
            setattr(self, attempts_attr, getattr(self, attempts_attr) + 1)
            try:
                source_fd = self._required_fd(source)
                self._restore_dup2(source_fd, target)
                os.set_inheritable(target, inheritable)
                if os.path.sameopenfile(target, source_fd):
                    if getattr(self, attempts_attr) > 1:
                        self.restoration_recovered = True
                        self.restoration_error = "fd_restoration_recovered"
                    return True
            except OSError:
                continue
        return False

    def _restore_process_globals(self) -> None:
        expected_stdout = self._installed_stdout or self._stdout
        expected_stderr = self._installed_stderr or self._stderr
        if sys.stdout is not expected_stdout or sys.stderr is not expected_stderr:
            self.restoration_recovered = True
            self.restoration_error = "process_global_restoration_recovered"
        try:
            sys.stdout = self._stdout
            sys.stderr = self._stderr
            self._restore_log_redaction()
        except (OSError, RuntimeError, TypeError, ValueError):
            self._fatal_exit()
        if sys.stdout is not self._stdout or sys.stderr is not self._stderr:
            self._fatal_exit()

    def _install_log_redaction(self) -> None:
        global _DIRECT_LOG_ACTIVE
        root = logging.getLogger()
        self._record_factory = logging.getLogRecordFactory()
        self._root_handlers = tuple(root.handlers)
        self._handler_filters = tuple(
            tuple(handler.filters) for handler in self._root_handlers
        )

        def record_factory(*args, **kwargs):
            record = self._record_factory(*args, **kwargs)
            if _direct_log_active():
                _redact_log_record(record)
            return record

        self._installed_record_factory = record_factory
        logging.setLogRecordFactory(record_factory)
        for handler in self._root_handlers:
            handler.addFilter(self._log_filter)
        with _DIRECT_LOG_LOCK:
            if _DIRECT_LOG_ACTIVE:
                raise DirectProviderIoCaptureError("provider_io_capture_overlap")
            _DIRECT_LOG_ACTIVE = True

    def _restore_log_redaction(self) -> None:
        global _DIRECT_LOG_ACTIVE
        root = logging.getLogger()
        with _DIRECT_LOG_LOCK:
            _DIRECT_LOG_ACTIVE = False
        if tuple(root.handlers) != self._root_handlers:
            self.restoration_recovered = True
            self.restoration_error = "process_global_restoration_recovered"
            root.handlers[:] = list(self._root_handlers)
        for handler, filters in zip(
            self._root_handlers, self._handler_filters, strict=True
        ):
            handler.filters[:] = list(filters)
        if logging.getLogRecordFactory() is not self._installed_record_factory:
            self.restoration_recovered = True
            self.restoration_error = "process_global_restoration_recovered"
        logging.setLogRecordFactory(self._record_factory)
        if (
            logging.getLogRecordFactory() is not self._record_factory
            or tuple(root.handlers) != self._root_handlers
            or tuple(tuple(handler.filters) for handler in root.handlers)
            != self._handler_filters
        ):
            self._fatal_exit()

    def _finish_drainer(self, stream: Literal["stdout", "stderr"]) -> None:
        thread = self.stdout_drainer if stream == "stdout" else self.stderr_drainer
        if thread is None:
            self._mark_drainer_error()
            return
        thread.join(_DRAIN_JOIN_SECONDS)
        if not thread.is_alive():
            return
        setattr(self, f"{stream}_join_timed_out", True)
        self.restoration_error = "drainer_join_timeout"
        stop_event = (
            self.stdout_stop_event if stream == "stdout" else self.stderr_stop_event
        )
        stop_event.set()
        self._close_owned(f"{stream}_read_fd")
        thread.join(_DRAIN_JOIN_SECONDS)

    def _close_owned(self, attribute: str) -> None:
        fd = getattr(self, attribute)
        if fd is None:
            return
        try:
            os.close(fd)
        except OSError:
            self.restoration_error = "capture_fd_close_error"
        self._owned_fds.discard(fd)
        setattr(self, attribute, None)

    def _flush_python_streams(self) -> None:
        for stream in (self._stdout, self._stderr):
            if stream is None:
                continue
            try:
                stream.flush()
            except (OSError, RuntimeError, ValueError):
                self.restoration_error = "python_stream_flush_error"

    def _abort_installation(self) -> None:
        fd1_restored = self._restore_if_redirected(
            1, self.saved_fd1_primary, self._fd1_inheritable
        )
        fd2_restored = self._restore_if_redirected(
            2, self.saved_fd2_primary, self._fd2_inheritable
        )
        if not fd1_restored or not fd2_restored:
            self._fatal_exit()
        for name in tuple(
            key
            for key in vars(self)
            if key.startswith("saved_fd") or key.endswith("_fd")
        ):
            self._close_owned(name)
        if self._record_factory is not None:
            self._restore_process_globals()
        self.stdout_stop_event.set()
        self.stderr_stop_event.set()
        for thread in (self.stdout_drainer, self.stderr_drainer):
            if thread is not None:
                thread.join(_DRAIN_JOIN_SECONDS)
        self.phase = "closed"

    def _restore_if_redirected(
        self, target: int, backup: int | None, inheritable: bool
    ) -> bool:
        if backup is None:
            return True
        try:
            os.dup2(backup, target)
            os.set_inheritable(target, inheritable)
            return os.path.sameopenfile(target, backup)
        except OSError:
            return False

    def _mark_drainer_error(self) -> None:
        self.restoration_error = "drainer_error"

    def _has_violation(self) -> bool:
        return any(
            (
                self.stdout_byte_count > 0,
                self.stderr_byte_count > 0,
                self.stdout_overflow,
                self.stderr_overflow,
                self.stdout_join_timed_out,
                self.stderr_join_timed_out,
                self.restoration_recovered,
                self.stdout_thread_alive,
                self.stderr_thread_alive,
                bool(self._owned_fds),
                self.restoration_error is not None,
            )
        )

    @staticmethod
    def _required_fd(fd: int | None) -> int:
        if fd is None:
            raise OSError("capture_fd_missing")
        return fd

    def _fatal_exit(self) -> None:
        self.process_fatal = True
        self.restoration_error = "provider_io_restore_unrecoverable"
        self.fatal_state = DirectProviderFatalStateV1(
            exit_code=70,
            reason_code="provider_io_restore_unrecoverable",
        )
        os._exit(70)


class DirectProviderIoCaptureScopeV1:
    def __init__(self) -> None:
        self.capture: DirectProviderIoCaptureV1 | None = None
        self._gate_acquired = False

    async def __aenter__(self) -> DirectProviderIoCaptureV1:
        await _acquire_provider_gate()
        self._gate_acquired = True
        try:
            self.capture = DirectProviderIoCaptureV1()
            self.capture.install()
            return self.capture
        except BaseException:
            self._release_gate()
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        del exc_type, exc, tb
        try:
            if self.capture is not None:
                self.capture.close()
        finally:
            self._release_gate()
        return False

    def _release_gate(self) -> None:
        if self._gate_acquired:
            _PROVIDER_GATE.release()
            self._gate_acquired = False


async def _acquire_provider_gate() -> None:
    acquire_task = asyncio.create_task(asyncio.to_thread(_PROVIDER_GATE.acquire))
    try:
        await asyncio.shield(acquire_task)
    except BaseException:
        acquired = await acquire_task
        if acquired:
            _PROVIDER_GATE.release()
        raise


def _direct_log_active() -> bool:
    with _DIRECT_LOG_LOCK:
        return _DIRECT_LOG_ACTIVE


def _redact_log_record(record: logging.LogRecord) -> None:
    record.msg = "direct_provider_log_suppressed"
    record.args = ()
    record.exc_info = None
    record.exc_text = None
    record.stack_info = None
    for key in tuple(record.__dict__):
        if key not in _STANDARD_LOG_FIELDS:
            del record.__dict__[key]


def _write_fd_all(fd: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        try:
            written = os.write(fd, remaining)
        except BlockingIOError:
            select.select((), (fd,), (), _DRAIN_POLL_SECONDS)
            continue
        if written <= 0:
            raise OSError("provider_stdio_write_failed")
        remaining = remaining[written:]
