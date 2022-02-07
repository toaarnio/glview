"""
Lightning-fast image viewer with smooth zooming & panning.

This package provides the 'glview' command-line application
only. There are no Python modules that you could import and
use in your own application.

https://github.com/toaarnio/glview
"""

from .glview import main

__version__ = "1.1.0"

__all__ = ["main"]
