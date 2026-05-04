"""Side-effect-heavy UI commands extracted from PygletUI."""

import pprint

import imgio
import imsize
import numpy as np
import piexif

from glview.imageutils import crop_borders


class UIOperations:
    """Imperative UI commands that coordinate window, files, and renderer state."""

    def __init__(self, ui):
        self.ui = ui

    def print_exif(self, filespec):
        try:
            exif_all = piexif.load(filespec)
            exif_tags = {tag: name for name, tag in piexif.ExifIFD.__dict__.items() if isinstance(tag, int)}
            image_tags = {tag: name for name, tag in piexif.ImageIFD.__dict__.items() if isinstance(tag, int)}
            exif_dict = {exif_tags[name]: val for name, val in exif_all.pop("Exif").items()}
            image_dict = {image_tags[name]: val for name, val in exif_all.pop("0th").items()}
            merged_dict = {**exif_dict, **image_dict}
            print(f"EXIF data for {filespec}:")
            pprint.pprint(merged_dict)
        except piexif.InvalidImageDataError as exc:
            print(f"Failed to extract EXIF metadata from {filespec}: {exc}")

    def request_exit(self):
        self.ui.running = False
        self.ui.event_loop.has_exit = True

    def toggle_fullscreen(self):
        self.ui.fullscreen = not self.ui.fullscreen
        self.ui.need_redraw = True
        self.ui.window.set_fullscreen(self.ui.fullscreen)
        self.ui.window.set_mouse_visible(not self.ui.fullscreen)

    def reload_visible_images(self):
        for imgidx in self.ui.state.img_per_tile[:self.ui.state.numtiles]:
            self.ui.loader.reload_image(imgidx)
            self.ui.files.mark_pending(imgidx)

    def toggle_debug_mode(self):
        self.ui.config.toggle_debug_mode()
        self.ui._vprint(f"debug rendering mode {self.ui.config.debug_mode}")
        self.ui.need_redraw = True

    def remove_visible_images(self):
        if self.ui.files.mutex.locked():
            return
        indices = self.ui.state.img_per_tile[:self.ui.state.numtiles]
        self.ui.files.drop(indices)
        self.finish_removal()

    def delete_current_image(self):
        if self.ui.files.mutex.locked() or self.ui.state.numtiles != 1:
            return
        imgidx = self.ui.state.img_per_tile[self.ui.state.tileidx]
        self.ui.files.delete(imgidx)
        self.finish_removal()

    def finish_removal(self):
        if self.ui.files.numfiles == 0:
            self.request_exit()
            return
        self.ui.state.repair_after_removal(self.ui.files.numfiles)
        self.ui.window.set_caption(self.ui._caption())
        self.ui.need_redraw = True

    def select_tile(self, tileidx: int):
        self.ui.state.select_tile(tileidx)
        self.ui.need_redraw = True

    def step_active_tile(self, incr: int):
        self.ui.state.step_active_tile(incr, self.ui.files.numfiles)
        self.ui.window.set_caption(self.ui._caption())
        self.ui.need_redraw = True

    def step_all_tiles(self, incr: int):
        self.ui.state.step_all_tiles(incr, self.ui.files.numfiles)
        self.ui.window.set_caption(self.ui._caption())
        self.ui.need_redraw = True

    def show_exif_for_current(self):
        imgidx = self.ui.state.img_per_tile[self.ui.state.tileidx]
        filespec = self.ui.files.filespec(imgidx)
        fileinfo = imsize.read(filespec)
        print(fileinfo)
        self.print_exif(filespec)

    def take_screenshot(self):
        screenshot_uint8 = self.ui.renderer.screenshot(np.uint8)
        screenshot_fp32 = self.ui.renderer.screenshot(np.float32)
        screenshot_uint8 = crop_borders(screenshot_uint8)
        screenshot_fp32 = crop_borders(screenshot_fp32)
        imgio.imwrite(f"screenshot{self.ui.ss_idx:02d}.jpg", screenshot_uint8, maxval=255, verbose=True)
        imgio.imwrite(f"screenshot{self.ui.ss_idx:02d}.pfm", screenshot_fp32, maxval=1.0, verbose=True)
        self.ui.ss_idx += 1
