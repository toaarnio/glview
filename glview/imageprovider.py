""" A multithreaded image file loader. """

import os                      # built-in library
import queue                   # built-in library
import time                    # built-in library
import threading               # built-in library
import traceback               # built-in library
from dataclasses import dataclass  # built-in library
from pathlib import Path       # built-in library
import numpy as np             # pip install numpy
import psutil                  # pip install psutil
import imsize                  # pip install imsize
import imgio                   # pip install imgio
from pqdm.threads import pqdm  # pip install pqdm

from glview.imagestate import ImageStatus


@dataclass(frozen=True)
class LoadRequest:
    idx: int
    token: tuple[int, int]
    filespec: str
    linearize: bool


class ImageProvider:
    """ A multithreaded image file loader. """

    def __init__(self, files, config):
        """ Create a new ImageProvider with the given (hardcoded) FileList instance. """
        self.thread_name = "ImageProviderThread"
        self.files = files
        self.config = config
        self.verbose = bool(config.verbose)
        self.downsample = config.downsample
        self.loader_thread = None
        self.running = False
        self._loader_statuses = [ImageStatus.PENDING] * self.files.numfiles
        self._loader_tokens = [self.files.image_token(i) for i in range(self.files.numfiles)]
        self._update_queue = queue.SimpleQueue()
        self._request_queue = queue.SimpleQueue()
        self.validate_files()

    def start(self):
        """ Start the image loader thread. """
        self._vprint(f"spawning {self.thread_name}...")
        self.running = True
        loader = self._sequential_loader if self.verbose else self._parallel_loader
        loader_func = lambda: loader(downsample=self.downsample)
        self.loader_thread = threading.Thread(target=lambda: self._try(loader_func), name=self.thread_name)
        self.loader_thread.daemon = True  # terminate when main process ends
        self.loader_thread.start()

    def stop(self):
        """ Stop the image loader thread. """
        self._vprint(f"killing {self.thread_name}...")
        self.running = False
        self.loader_thread.join()
        self._vprint(f"{self.thread_name} killed")

    def validate_files(self):
        """ Quickly validate all image files & estimate their memory consumption. """
        size_on_disk = 0.0
        size_in_mem = 0.0
        if self.files.numfiles > 100:
            print("Scanning images & estimating memory consumption...")
        invalid = []
        for idx, entry in enumerate(self.files.entries):
            filespec = entry.filespec
            try:
                info = imsize.read(filespec)
                size_on_disk += info.filesize
                size_in_mem += info.nbytes
            except imsize.ImageFileError as e:
                print(f"{os.path.basename(filespec)}: Skipping: {e}")
                invalid.append(idx)
            except RuntimeError as e:
                print(f"{os.path.basename(filespec)}: Skipping: {e}")
                invalid.append(idx)
            except TypeError:
                print(f"{os.path.basename(filespec)}: Skipping: Unable to determine image dimensions.")
                invalid.append(idx)
        if invalid:
            self.files.drop(invalid)
        size_on_disk /= 1024 ** 2
        size_in_mem /= 1024 ** 2
        ram_available = psutil.virtual_memory().available / 1024**2
        print(f"Found {self.files.numfiles} valid images, consuming {size_on_disk:.0f} MB on disk, "
              f"{size_in_mem:.0f} MB in memory ({ram_available:.0f} MB available).")

    def get_image(self, index):
        """
        Return the image at the given index. The image might not be present,
        in which case the return value is the corresponding slot-state label.
        """
        if self.files.image_status(index) == ImageStatus.LOADED:
            return self.files.get_consumed_image(index)
        return self.files.image_status(index).value

    def get_image_record(self, index):
        """Return the current image payload or state plus the slot identity token."""
        token = self.files.image_token(index)
        image = self.get_image(index)
        return image, token

    def release_image(self, index, token=None):
        """
        Release the image at the given index to (eventually) free up the memory.
        """
        if token is not None and token != self.files.image_token(index):
            return
        self.files.mark_released(index)
        self.files.clear_consumed_image(index)
        self._request_queue.put(("release", index, token or self.files.image_token(index)))

    def reload_image(self, index):
        """Request reloading the image at the given index from disk."""
        self.files.mark_pending(index)
        self._request_queue.put(("reload", index, self.files.image_token(index)))

    def apply_updates(self):
        """Apply pending loader-thread updates on the UI/render thread."""
        loaded_updates, invalid_updates = self._drain_update_queue()
        removed = self._drop_invalid_updates(invalid_updates)
        self._apply_loaded_updates(loaded_updates)
        return removed

    def _drain_update_queue(self):
        loaded_updates = []
        invalid_updates = []
        while True:
            try:
                update = self._update_queue.get_nowait()
            except queue.Empty:
                break
            action, idx, slot_id, revision, *payload = update
            if not self._matches_current_slot(idx, slot_id, revision):
                continue
            if action == "loaded":
                loaded_updates.append((idx, slot_id, revision, payload[0]))
            elif action == "invalid":
                invalid_updates.append((idx, slot_id, revision))
        return loaded_updates, invalid_updates

    def _drop_invalid_updates(self, invalid_updates):
        removed = False
        for idx, slot_id, revision in sorted(invalid_updates, reverse=True):
            if not self._matches_current_slot(idx, slot_id, revision):
                continue
            self.files.drop([idx])
            removed = True
        return removed

    def _apply_loaded_updates(self, loaded_updates):
        for idx, slot_id, revision, img in loaded_updates:
            if not self._matches_current_slot(idx, slot_id, revision):
                continue
            self.files.mark_loaded(idx, img)
            self.files.consume_image(idx, img)

    def _matches_current_slot(self, idx, slot_id, revision):
        if idx >= self.files.numfiles:
            return False
        return (slot_id, revision) == self.files.image_token(idx)

    def _degamma(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply standard sRGB inverse gamma on the given uint8/16 frame, returning
        the result in normalized float16 format.
        """
        assert frame.dtype in [np.uint8, np.uint16], frame.dtype
        t0 = time.time()
        bpp = 8 if frame.dtype == np.uint8 else 16
        lut = np.linspace(0, 1, 2 ** bpp).astype(np.float16)
        srgb_lo = lut / 12.92
        srgb_hi = np.power((lut + 0.055) / 1.055, 2.4)
        threshold_mask = (lut > 0.04045)
        lut = srgb_hi * threshold_mask + srgb_lo * (~threshold_mask)
        frame = lut[frame]
        elapsed = (time.time() - t0) * 1000
        self._vprint(f"Applying inverse sRGB gamma took {elapsed:.1f} ms")
        return frame

    def _parallel_loader(self, downsample):
        ram_total = psutil.virtual_memory().total / 1024**2
        ram_before = psutil.virtual_memory().available / 1024**2
        verbose = self.verbose or self.files.numfiles < 200
        def split():
            nfiles = self.files.numfiles
            splits = [2, 8, 16] + list(range(32, nfiles, 32))
            chunks = np.split(np.arange(nfiles), splits)
            return chunks
        while self.running:  # keep running in the background
            t0 = time.time()
            self._check_ram(2048, wait=True)
            nloaded = 0
            nbytes = 0
            for chunk in split():
                with self.files.mutex:
                    self._sync_loader_state_locked()
                    self._drain_requests_locked()
                    ok = self._check_ram(2048, wait=False)
                    requests = [request for idx in chunk if (request := self._take_pending_request_locked(idx))] if ok else []
                if not requests:
                    continue
                batch = [(request, downsample, verbose) for request in requests]
                images = pqdm(batch, self._load_request, 8, "args", bounded=True,
                              exception_behaviour="immediate", disable=True)
                with self.files.mutex:
                    self._sync_loader_state_locked()
                    self._drain_requests_locked()
                    for request, img in zip(requests, images, strict=True):
                        published = self._publish_result_locked(request, img)
                        if published:
                            nloaded += 1
                            if isinstance(img, np.ndarray):
                                nbytes += img.nbytes
            if nloaded > 0 and nbytes > 1e4:
                elapsed = time.time() - t0
                nbytes = nbytes / 1024**2
                bandwidth = nbytes / elapsed
                ram_after = psutil.virtual_memory().available / 1024**2
                consumed = ram_before - ram_after
                nfiles = self.files.numfiles
                self._print(f"loaded {nloaded}/{nfiles} files, {nbytes:.0f} MB of image data in {elapsed:.1f} seconds ({bandwidth:.1f} MB/sec).")
                self._print(f"consumed {consumed:.0f} MB of system RAM, {ram_after:.0f}/{ram_total:.0f} MB remaining.")
            time.sleep(0.1)

    def _sequential_loader(self, downsample):
        """
        Load all images in sequential order.
        """
        ram_total = psutil.virtual_memory().total / 1024**2
        ram_before = psutil.virtual_memory().available / 1024**2
        while self.running:  # loop until program termination
            idx = 0
            nbytes = 0
            nloaded = 0
            t0 = time.time()
            while self.running:  # load all files
                self._check_ram(2048, wait=True)
                with self.files.mutex:
                    self._sync_loader_state_locked()
                    self._drain_requests_locked()
                    if idx >= self.files.numfiles:
                        break
                    request = self._take_pending_request_locked(idx)
                verbose = self.verbose or self.files.numfiles < 200
                img = self._load_request(request, downsample, verbose) if request else None
                if request is not None:
                    with self.files.mutex:
                        self._sync_loader_state_locked()
                        self._drain_requests_locked()
                        published = self._publish_result_locked(request, img)
                        if published and isinstance(img, np.ndarray):
                            nbytes += img.nbytes
                        nloaded += int(published)
                time.sleep(0.001)
                idx += 1
                with self.files.mutex:
                    if self.files.reindexed:
                        self.files.reindexed = False
                        idx = 0  # restart from beginning
            if nbytes > 1e4:
                elapsed = time.time() - t0
                nbytes = nbytes / 1024**2
                bandwidth = nbytes / elapsed
                ram_after = psutil.virtual_memory().available / 1024**2
                consumed = ram_before - ram_after
                self._print(f"loaded {nloaded} files, {nbytes:.0f} MB of image data in {elapsed:.1f} seconds ({bandwidth:.1f} MB/sec).")
                self._print(f"consumed {consumed:.0f} MB of system RAM, {ram_after:.0f}/{ram_total:.0f} MB remaining.")
            time.sleep(0.1)

    def _load_single(self, idx, downsample, verbose):
        """
        Read image, drop alpha channel, convert to fp32 if maxval != 255;
        if loading fails, mark the slot as "INVALID" and keep going.
        """
        with self.files.mutex:
            request = self._take_pending_request_locked(idx)
        return self._load_request(request, downsample, verbose) if request else None

    def _take_pending_request_locked(self, idx):
        if idx >= len(self._loader_statuses) or self._loader_statuses[idx] != ImageStatus.PENDING:
            return None
        filespec = self.files.filespec(idx)
        suffix = Path(filespec).suffix.lower()
        linearize = suffix in [".jpg", ".jpeg", ".png", ".bmp", ".ppm"]
        return LoadRequest(idx=idx, token=self._loader_tokens[idx], filespec=filespec, linearize=linearize)

    def _load_request(self, request, downsample, verbose):
        try:
            info = imsize.read(request.filespec)
            if info.cfa_raw:
                packing = "unpacked"
                if info.packed_raw:
                    packing = "plain"
                if info.mipi_raw:
                    packing = "mipi"
                width = self.config.width or info.width
                height = self.config.height or info.height
                bpp = self.config.bpp or info.bitdepth
                stride = self.config.stride or info.stride
                packing = self.config.packing or packing
                img, maxval = imgio.rawread(request.filespec, width, height, bpp, stride, packing, verbose=verbose)
            else:
                img, maxval = imgio.imread(request.filespec, verbose=verbose)
            img = img[::downsample, ::downsample]
            img = np.atleast_3d(img)  # {2D, 3D} => 3D
            img = img[:, :, :3]  # scrap alpha channel, if any
            if request.linearize:
                # invert sRGB gamma, as OpenGL texture filtering assumes
                # linear colors; float16 precision is the minimum to avoid
                # clipping dark colors to black
                img = self._degamma(img)
            if img.dtype == np.uint16:
                # uint16 still doesn't work in ModernGL as of 5.7.4;
                # scale to [0, 1] and use float32 instead
                norm = max(maxval, np.max(img))
                img = img.astype(np.float32) / norm
            if img.dtype == np.float64:
                # float64 is not universally supported yet
                img = img.astype(np.float32)
            return img
        except (imgio.ImageIOError, imsize.ImageFileError) as e:
            if not self.verbose:
                self._print(e)
            else:
                self._vprint(e)
            return ImageStatus.INVALID.value

    def _publish_result_locked(self, request, img):
        idx = request.idx
        if idx >= len(self._loader_statuses):
            return False
        if self._loader_statuses[idx] != ImageStatus.PENDING:
            return False
        if request.token != self._loader_tokens[idx]:
            return False
        slot_id, revision = request.token
        if isinstance(img, np.ndarray):
            self._loader_statuses[idx] = ImageStatus.LOADED
            self._update_queue.put(("loaded", idx, slot_id, revision, img))
            return True
        if img == ImageStatus.INVALID.value:
            self._loader_statuses[idx] = ImageStatus.INVALID
            self._update_queue.put(("invalid", idx, slot_id, revision))
            return True
        return False

    def _sync_loader_state_locked(self):
        if len(self._loader_statuses) != self.files.numfiles:
            self._loader_statuses = [self.files.image_status(i) for i in range(self.files.numfiles)]
            self._loader_tokens = [self.files.image_token(i) for i in range(self.files.numfiles)]

    def _drain_requests_locked(self):
        while True:
            try:
                action, idx, token = self._request_queue.get_nowait()
            except queue.Empty:
                break
            if idx >= len(self._loader_statuses):
                continue
            if action == "release":
                if token != self._loader_tokens[idx]:
                    continue
                self._loader_statuses[idx] = ImageStatus.RELEASED
            elif action == "reload":
                if token[0] != self._loader_tokens[idx][0]:
                    continue
                self._loader_statuses[idx] = ImageStatus.PENDING
                self._loader_tokens[idx] = token

    def _check_ram(self, minimum, wait):
        ram_available = lambda: psutil.virtual_memory().available / 1024**2
        while ram_available() < minimum:
            self._print(f"WARNING: Only {ram_available():.0f} MB of RAM remaining. At least {minimum} MB required to continue loading.")
            if wait:
                time.sleep(5.0)
                continue
            break
        return ram_available() >= minimum

    def _try(self, func):
        try:
            func()
        except Exception as e:
            self.running = False
            if self.verbose:
                self._vprint(f"exception in {func.__name__}():")
                traceback.print_exc()
            else:
                self._print(f"{type(e).__name__}: {e}")

    def _print(self, message, **kwargs):
        verbose, self.verbose = self.verbose, True
        self._vprint(message, **kwargs)
        self.verbose = verbose

    def _vprint(self, message, **kwargs):
        if self.verbose:
            prefix = f"[{self.__class__.__name__}/{threading.current_thread().name}]"
            print(f"{prefix} {message}", **kwargs)
