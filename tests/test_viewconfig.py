import unittest

import numpy as np

from glview.viewconfig import ViewConfigState


class ViewConfigStateTests(unittest.TestCase):

    def test_reset_exposure_clears_linear_and_derived_values(self):
        config = ViewConfigState(ev_linear=0.5, ev=1.2)

        config.reset_exposure()

        self.assertEqual(config.ev_linear, 0.0)
        self.assertEqual(config.ev, 0.0)

    def test_cycle_methods_advance_expected_modes(self):
        config = ViewConfigState(texture_filter="NEAREST", gamma=1, cs_in=0, cs_out=0, normalize=0, ev_range=2)

        config.cycle_gamma()
        config.cycle_input_colorspace()
        config.cycle_output_colorspace()
        config.cycle_normalize()
        config.toggle_exposure_range()
        config.toggle_texture_filter()
        config.toggle_debug_mode()

        self.assertEqual(config.gamma, 2)
        self.assertEqual(config.cs_in, 1)
        self.assertEqual(config.cs_out, 1)
        self.assertEqual(config.normalize, 1)
        self.assertEqual(config.ev_range, 8)
        self.assertEqual(config.texture_filter, "LINEAR")
        self.assertTrue(config.debug_mode_on)

    def test_update_exposure_uses_given_waveform_and_reports_change(self):
        config = ViewConfigState(ev_range=4)

        changed = config.update_exposure(increase=True, triangle_wave=lambda x, amp: x + amp)

        self.assertTrue(changed)
        self.assertAlmostEqual(config.ev_linear, 0.005)
        self.assertAlmostEqual(config.ev, 4.005)

    def test_update_exposure_without_change_still_refreshes_ev(self):
        config = ViewConfigState(ev_linear=0.25, ev_range=2)

        changed = config.update_exposure(increase=False, triangle_wave=lambda x, amp: x - amp)

        self.assertFalse(changed)
        self.assertAlmostEqual(config.ev_linear, 0.25)
        self.assertAlmostEqual(config.ev, -1.75)

    def test_default_gamut_arrays_are_initialized(self):
        config = ViewConfigState()

        np.testing.assert_array_equal(config.gamut_pow, np.ones(3) * 5.0)
        np.testing.assert_array_equal(config.gamut_lim, np.ones(3) * 1.1)
        np.testing.assert_array_equal(config.gamut_thr, np.ones(3) * 0.8)


if __name__ == "__main__":
    unittest.main()
