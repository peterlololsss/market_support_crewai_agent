from __future__ import annotations

import logging

from market_support_crewai_agent.server.logging_config import (
    APP_LOGGER_NAME,
    configure_app_logging,
)


def test_configure_app_logging_reuses_uvicorn_handler_once(monkeypatch):
    app_logger = logging.getLogger(APP_LOGGER_NAME)
    root_logger = logging.getLogger()
    uvicorn_logger = logging.getLogger("uvicorn.error")
    original_root = _snapshot(root_logger)
    original_app = _snapshot(app_logger)
    original_uvicorn = _snapshot(uvicorn_logger)
    handler = logging.StreamHandler()

    try:
        root_logger.handlers[:] = []
        app_logger.handlers[:] = []
        uvicorn_logger.handlers[:] = [handler]
        monkeypatch.setenv("MARKET_AGENT_LOG_LEVEL", "DEBUG")

        configure_app_logging()
        configure_app_logging()

        assert app_logger.handlers == [handler]
        assert app_logger.level == logging.DEBUG
        assert handler.level == logging.DEBUG
        assert app_logger.propagate is False
    finally:
        _restore(root_logger, original_root)
        _restore(app_logger, original_app)
        _restore(uvicorn_logger, original_uvicorn)


def test_configure_app_logging_uses_existing_root_logging(monkeypatch):
    app_logger = logging.getLogger(APP_LOGGER_NAME)
    root_logger = logging.getLogger()
    original_root = _snapshot(root_logger)
    original_app = _snapshot(app_logger)
    handler = logging.StreamHandler()

    try:
        root_logger.handlers[:] = [handler]
        app_logger.handlers[:] = []
        monkeypatch.setenv("MARKET_AGENT_LOG_LEVEL", "INFO")

        configure_app_logging()

        assert app_logger.handlers == []
        assert app_logger.propagate is True
    finally:
        _restore(root_logger, original_root)
        _restore(app_logger, original_app)


def _snapshot(logger: logging.Logger):
    return list(logger.handlers), logger.level, logger.propagate


def _restore(logger: logging.Logger, snapshot) -> None:
    handlers, level, propagate = snapshot
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate
