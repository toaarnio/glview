"""
I/O utilities: fault-tolerant printing for consoleless Windows processes.
"""
import os
import sys
import builtins
import contextlib


def _safe_print(*args, **_kwargs):
    with contextlib.suppress(OSError):
        _original_print(*args)


class _SafeStream:
    """
    Wrap a text stream so write/flush/close swallow OSError.

    On Windows, the interpreter performs a final flush of sys.stdout and
    sys.stderr during shutdown — after all atexit handlers have run. If the
    underlying console handle has already been torn down (common when launched
    from pythonw.exe, via pipx shims, or when the parent console exits first),
    that flush raises 'OSError: [WinError 1] Incorrect function' and prints a
    spurious "Exception ignored on flushing sys.stdout" message. Routing the
    streams through this wrapper makes those shutdown errors silent.
    """

    def __init__(self, stream):
        self._stream = stream

    def write(self, data):
        try:
            return self._stream.write(data)
        except OSError:
            return 0

    def flush(self):
        with contextlib.suppress(OSError):
            self._stream.flush()

    def close(self):
        with contextlib.suppress(OSError):
            self._stream.close()

    def isatty(self):
        try:
            return self._stream.isatty()
        except OSError:
            return False

    def __getattr__(self, name):
        return getattr(self._stream, name)


if os.name == "nt":
    _original_print = builtins.print
    builtins.print = _safe_print
    if sys.stdout is not None:
        sys.stdout = _SafeStream(sys.stdout)
    if sys.stderr is not None:
        sys.stderr = _SafeStream(sys.stderr)
