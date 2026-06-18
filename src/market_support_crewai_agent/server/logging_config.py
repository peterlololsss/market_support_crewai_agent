from __future__ import annotations

import logging
import os

APP_LOGGER_NAME = "market_support_crewai_agent"
LOG_LEVEL_ENV = "MARKET_AGENT_LOG_LEVEL"


def configure_app_logging() -> None:
    level = _log_level()
    app_logger = logging.getLogger(APP_LOGGER_NAME)
    app_logger.setLevel(level)

    if app_logger.handlers:
        app_logger.propagate = False
    elif logging.getLogger().handlers:
        app_logger.propagate = True
    else:
        app_logger.propagate = False
        handlers = list(logging.getLogger("uvicorn.error").handlers)
        if handlers:
            app_logger.handlers.extend(handlers)
        else:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "[%(levelname)s][%(asctime)s][%(name)s:%(lineno)d] - %(message)s"
                )
            )
            app_logger.addHandler(handler)

    for handler in app_logger.handlers:
        handler.setLevel(level)


def _log_level() -> int:
    value = os.getenv(LOG_LEVEL_ENV, "INFO").strip().upper()
    level = getattr(logging, value, logging.INFO)
    return level if isinstance(level, int) else logging.INFO
