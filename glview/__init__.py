"""
Lightning-fast image viewer with smooth zooming & panning.

This package provides the 'glview' command-line application
only. There are no Python modules that you could import and
use in your own application.

https://github.com/toaarnio/glview
"""

from .glview import main
from .version import __version__

__all__ = ["main"]
