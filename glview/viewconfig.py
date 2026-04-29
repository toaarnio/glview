"""Viewer configuration state for rendering and exposure controls."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ViewConfigState:
    debug_mode: int | str = 1
    debug_mode_on: bool = False
    texture_filter: str = "NEAREST"
    cs_in: int = 0
    cs_out: int = 0
    gamma: int = 1
    normalize: int = 0
    ev_range: int = 2
    ev_linear: float = 0.0
    ev: float = 0.0
    gamut_pow: np.ndarray = field(default_factory=lambda: np.ones(3) * 5.0)
    gamut_lim: np.ndarray = field(default_factory=lambda: np.ones(3) * 1.1)
    gamut_thr: np.ndarray = field(default_factory=lambda: np.ones(3) * 0.8)

    def reset_exposure(self):
        self.ev_linear = 0.0
        self.ev = 0.0

    def cycle_gamma(self):
        self.gamma = (self.gamma + 1) % 4

    def cycle_input_colorspace(self):
        self.cs_in = (self.cs_in + 1) % 4

    def cycle_output_colorspace(self):
        self.cs_out = (self.cs_out + 1) % 4

    def toggle_exposure_range(self):
        self.ev_range = (self.ev_range + 6) % 12

    def cycle_normalize(self):
        self.normalize = (self.normalize + 1) % 8

    def toggle_texture_filter(self):
        self.texture_filter = "LINEAR" if self.texture_filter == "NEAREST" else "NEAREST"

    def toggle_debug_mode(self):
        self.debug_mode_on = not self.debug_mode_on

    def update_exposure(self, increase: bool, triangle_wave):
        if increase:
            self.ev_linear += 0.005
        self.ev = triangle_wave(self.ev_linear, self.ev_range)
        return increase
