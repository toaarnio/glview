"""Shared image-processing helpers."""

import numpy as np


def crop_borders(img):
    """Remove fully black border rows and columns from an RGB-like image."""
    span = lambda a: slice(a.argmax(), a.size - a[::-1].argmax())
    nonzero = np.any(img != 0.0, axis=2)
    rowmask = np.any(nonzero, axis=1)
    colmask = np.any(nonzero, axis=0)
    img = img[span(rowmask), :]
    img = img[:, span(colmask)]
    return img
