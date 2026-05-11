"""
I/O utilities: fault-tolerant printing for consoleless Windows processes.
"""
import os
import builtins
import contextlib


def _safe_print(*args, **_kwargs):
    with contextlib.suppress(OSError):
        _original_print(*args)


if os.name == "nt":
    _original_print = builtins.print
    builtins.print = _safe_print
