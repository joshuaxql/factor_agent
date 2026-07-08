# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import re
import sys
from contextlib import contextmanager
from time import time
from typing import Any, Dict, Iterable, Optional, Text

from loguru import logger as _logger

from .config import C


TRACE = 5
DEBUG = 10
INFO = 20
WARNING = 30
ERROR = 40
CRITICAL = 50

_LEVEL_NAME_TO_NO = {
    "TRACE": TRACE,
    "DEBUG": DEBUG,
    "INFO": INFO,
    "SUCCESS": 25,
    "WARNING": WARNING,
    "WARN": WARNING,
    "ERROR": ERROR,
    "CRITICAL": CRITICAL,
}
_LEVEL_NO_TO_NAME = {
    TRACE: "TRACE",
    DEBUG: "DEBUG",
    INFO: "INFO",
    25: "SUCCESS",
    WARNING: "WARNING",
    ERROR: "ERROR",
    CRITICAL: "CRITICAL",
}
_DEFAULT_FORMAT = (
    "[{process}:{thread.name}]({time:YYYY-MM-DD HH:mm:ss.SSS}) "
    "{level} - {extra[qlib_name]} - [{file.name}:{line}] - {message}"
)

_global_logger_level = 0


def _patch_record(record: Dict[str, Any]) -> None:
    record["extra"].setdefault("qlib_name", record["name"])


_logger.configure(patcher=_patch_record)


def _normalize_level_no(level: int | str | None) -> int:
    if level is None:
        return INFO
    if isinstance(level, str):
        normalized = level.upper()
        if normalized.isdigit():
            return int(normalized)
        if normalized not in _LEVEL_NAME_TO_NO:
            raise ValueError(f"Unsupported log level: {level}")
        return _LEVEL_NAME_TO_NO[normalized]
    return int(level)


def _normalize_level_for_loguru(level: int | str | None) -> int | str:
    level_no = _normalize_level_no(level)
    return _LEVEL_NO_TO_NAME.get(level_no, level_no)


def _format_compat_message(message: object, *args: object) -> str:
    text = str(message)
    if not args:
        return text
    try:
        return text % args
    except Exception:
        try:
            return text.format(*args)
        except Exception:
            return " ".join([text, *(str(arg) for arg in args)])


class QlibLogger:
    """
    Loguru-backed logger with the small stdlib-logging surface Qlib uses.
    """

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.__level = INFO

    @property
    def level(self) -> int:
        return self.__level

    @property
    def logger(self):
        return _logger.bind(qlib_name=self.module_name)

    def setLevel(self, level: int | str) -> None:  # noqa: N802 - keep stdlib-compatible API
        self.__level = _normalize_level_no(level)

    def getEffectiveLevel(self) -> int:  # noqa: N802 - keep stdlib-compatible API
        return self.__level

    def isEnabledFor(self, level: int | str) -> bool:  # noqa: N802 - keep stdlib-compatible API
        return _normalize_level_no(level) >= max(self.__level, _global_logger_level)

    def log(self, level: int | str = INFO, msg: object = "", *args: object, **kwargs: Any) -> None:
        if "message" in kwargs and msg == "":
            msg = kwargs.pop("message")
        if not self.isEnabledFor(level):
            return
        depth = kwargs.pop("_depth", 1)
        exc_info = kwargs.pop("exc_info", None)
        exception = kwargs.pop("exception", exc_info)
        self.logger.opt(depth=depth, exception=exception).log(
            _normalize_level_for_loguru(level), _format_compat_message(msg, *args), **kwargs
        )

    def trace(self, msg: object, *args: object, **kwargs: Any) -> None:
        self.log(TRACE, msg, *args, _depth=2, **kwargs)

    def debug(self, msg: object, *args: object, **kwargs: Any) -> None:
        self.log(DEBUG, msg, *args, _depth=2, **kwargs)

    def info(self, msg: object, *args: object, **kwargs: Any) -> None:
        self.log(INFO, msg, *args, _depth=2, **kwargs)

    def success(self, msg: object, *args: object, **kwargs: Any) -> None:
        self.log("SUCCESS", msg, *args, _depth=2, **kwargs)

    def warning(self, msg: object, *args: object, **kwargs: Any) -> None:
        self.log(WARNING, msg, *args, _depth=2, **kwargs)

    warn = warning

    def error(self, msg: object, *args: object, **kwargs: Any) -> None:
        self.log(ERROR, msg, *args, _depth=2, **kwargs)

    def exception(self, msg: object, *args: object, **kwargs: Any) -> None:
        kwargs.setdefault("exception", True)
        self.log(ERROR, msg, *args, _depth=2, **kwargs)

    def critical(self, msg: object, *args: object, **kwargs: Any) -> None:
        self.log(CRITICAL, msg, *args, _depth=2, **kwargs)

    fatal = critical

    def bind(self, **kwargs: Any):
        return self.logger.bind(**kwargs)

    def __getattr__(self, name: str) -> Any:
        if name in {"__setstate__"}:
            raise AttributeError
        return getattr(self.logger, name)


class _QLibLoggerManager:
    def __init__(self):
        self._loggers: dict[str, QlibLogger] = {}

    def setLevel(self, level: int | str) -> None:  # noqa: N802 - keep stdlib-compatible API
        for logger in self._loggers.values():
            logger.setLevel(level)

    def __call__(self, module_name: str, level: Optional[int | str] = None) -> QlibLogger:
        """
        Get a logger for a specific module.

        :param module_name: str
            Logic module name.
        :param level: int | str
        :return: QlibLogger
            Logger object.
        """
        if level is None:
            level = C.logging_level

        if not module_name.startswith("qlib."):
            # Add a prefix of qlib. when the requested ``module_name`` doesn't start with ``qlib.``.
            # If the module_name is already qlib.xxx, we do not format here. Otherwise, it will become qlib.qlib.xxx.
            module_name = "qlib.{}".format(module_name)

        module_logger = self._loggers.setdefault(module_name, QlibLogger(module_name))
        module_logger.setLevel(level)
        return module_logger


get_module_logger = _QLibLoggerManager()


class TimeInspector:
    timer_logger = get_module_logger("timer")

    time_marks: list[float] = []

    @classmethod
    def set_time_mark(cls):
        """
        Set a time mark with current time, and this time mark will push into a stack.
        :return: float
            A timestamp for current time.
        """
        _time = time()
        cls.time_marks.append(_time)
        return _time

    @classmethod
    def pop_time_mark(cls):
        """
        Pop last time mark from stack.
        """
        return cls.time_marks.pop()

    @classmethod
    def get_cost_time(cls):
        """
        Get last time mark from stack, calculate time diff with current time.
        :return: float
            Time diff calculated by last time mark with current time.
        """
        cost_time = time() - cls.time_marks.pop()
        return cost_time

    @classmethod
    def log_cost_time(cls, info="Done"):
        """
        Get last time mark from stack, calculate time diff with current time, and log time diff and info.
        :param info: str
            Info that will be logged into stdout.
        """
        cost_time = time() - cls.time_marks.pop()
        cls.timer_logger.info("Time cost: {0:.3f}s | {1}".format(cost_time, info))

    @classmethod
    @contextmanager
    def logt(cls, name="", show_start=False):
        """Log the time of the inside code."""
        if show_start:
            cls.timer_logger.info(f"{name} Begin")
        cls.set_time_mark()
        try:
            yield None
        finally:
            pass
        cls.log_cost_time(info=f"{name} Done")


class LogFilter:
    def __init__(self, param=None):
        self.param = param

    @staticmethod
    def match_msg(filter_str, msg):
        match = False
        try:
            if re.match(filter_str, msg):
                match = True
        except Exception:
            pass
        return match

    def filter(self, record):
        allow = True
        msg = record["message"] if isinstance(record, dict) else getattr(record, "msg", "")
        if isinstance(self.param, str):
            allow = not self.match_msg(self.param, msg)
        elif isinstance(self.param, list):
            allow = not any(self.match_msg(p, msg) for p in self.param)
        return allow


def _iter_filter_patterns(log_config: Dict[Text, Any], filter_names: Iterable[str] | None = None) -> list[str]:
    filters = log_config.get("filters", {})
    if filter_names is None:
        filter_names = filters.keys()

    patterns: list[str] = []
    for filter_name in filter_names:
        filter_config = filters.get(filter_name, {})
        param = filter_config.get("param")
        if isinstance(param, str):
            patterns.append(param)
        elif isinstance(param, list):
            patterns.extend(str(item) for item in param)
    return patterns


def _make_filter(patterns: list[str]):
    def _filter(record: Dict[str, Any]) -> bool:
        return not any(LogFilter.match_msg(pattern, record["message"]) for pattern in patterns)

    return _filter


def _handler_sink(handler_config: Dict[str, Any]):
    handler_class = handler_config.get("class", "")
    if "FileHandler" in handler_class:
        return handler_config.get("filename")
    stream = handler_config.get("stream", "ext://sys.stderr")
    if stream == "ext://sys.stdout":
        return sys.stdout
    return sys.stderr


def _resolve_sink(sink: Any):
    if sink in {"stderr", "ext://sys.stderr"}:
        return sys.stderr
    if sink in {"stdout", "ext://sys.stdout"}:
        return sys.stdout
    return sink


def _handler_format(log_config: Dict[Text, Any], handler_config: Dict[str, Any]) -> str:
    formatter_name = handler_config.get("formatter")
    formatter_config = log_config.get("formatters", {}).get(formatter_name, {})
    format_string = formatter_config.get("format")
    if isinstance(format_string, str) and "%(" not in format_string:
        return format_string
    return _DEFAULT_FORMAT


def set_log_with_config(log_config: Dict[Text, Any]):
    """Configure loguru sinks.

    This accepts the compact subset of the old dictConfig-shaped settings used by Qlib
    and also accepts loguru-style ``handlers`` with sink/level/format fields.
    """
    if not log_config:
        return

    _logger.remove()
    handlers = log_config.get("handlers", {})
    if not handlers:
        _logger.add(sys.stderr, level="DEBUG", format=_DEFAULT_FORMAT)
        return

    for handler_config in handlers.values():
        sink = _resolve_sink(handler_config.get("sink", _handler_sink(handler_config)))
        if sink is None:
            continue
        level = _normalize_level_for_loguru(handler_config.get("level", "DEBUG"))
        patterns = _iter_filter_patterns(log_config, handler_config.get("filters"))
        _logger.add(
            sink,
            level=level,
            format=handler_config.get("format", _handler_format(log_config, handler_config)),
            filter=_make_filter(patterns),
            enqueue=bool(handler_config.get("enqueue", False)),
            backtrace=bool(handler_config.get("backtrace", True)),
            diagnose=bool(handler_config.get("diagnose", False)),
        )


def set_global_logger_level(level: int | str, return_orig_handler_level: bool = False):
    """Set the minimum level for Qlib module loggers."""
    global _global_logger_level
    original = _global_logger_level
    _global_logger_level = _normalize_level_no(level)
    return {"global": original} if return_orig_handler_level else None


@contextmanager
def set_global_logger_level_cm(level: int | str):
    """Set Qlib module loggers' minimum level in a context manager."""
    _handler_level_map = set_global_logger_level(level, return_orig_handler_level=True)
    try:
        yield
    finally:
        global _global_logger_level
        _global_logger_level = _handler_level_map["global"]
