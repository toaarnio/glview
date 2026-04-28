"""Explicit image slot state used across the loader, UI, and renderer."""

from dataclasses import dataclass
from enum import Enum


class ImageStatus(Enum):
    PENDING = "PENDING"
    LOADED = "LOADED"
    RELEASED = "RELEASED"
    INVALID = "INVALID"


@dataclass
class ImageSlot:
    status: ImageStatus = ImageStatus.PENDING
    revision: int = 0
