import logging
import os
import sys
import threading
from functools import lru_cache

_thread_lock = threading.RLock()
_default_handler = None


class _Logger(logging.Logger):
    r"""A logger that supports rank0 logging."""

    def info_rank0(self, *args, **kwargs) -> None:
        self.info(*args, **kwargs)

    def warning_rank0(self, *args, **kwargs) -> None:
        self.warning(*args, **kwargs)

    def warning_rank0_once(self, *args, **kwargs) -> None:
        self.warning(*args, **kwargs)


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> "_Logger":
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    r"""Configure root logger using a stdout stream handler with an explicit format."""
    global _default_handler

    with _thread_lock:
        if _default_handler:  # already configured
            return

        formatter = logging.Formatter(
            fmt="[%(levelname)s|%(asctime)s] %(name)s:%(lineno)s >> %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.setFormatter(formatter)
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(logging.INFO)
        library_root_logger.propagate = False


def get_logger() -> _Logger:
    _configure_library_root_logger()
    return _get_library_root_logger()


def info_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.info(*args, **kwargs)


def warning_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.warning(*args, **kwargs)


@lru_cache(None)
def warning_rank0_once(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.warning(*args, **kwargs)


logging.Logger.info_rank0 = info_rank0
logging.Logger.warning_rank0 = warning_rank0
logging.Logger.warning_rank0_once = warning_rank0_once
