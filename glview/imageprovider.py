""" A multithreaded image file loader. """

import os                      # built-in library
import time                    # built-in library
import threading               # built-in library
import urllib.request          # built-in library
import tempfile                # built-in library
import traceback               # built-in library
from pathlib import Path       # built-in library
import numpy as np             # pip install numpy
import psutil                  # pip install psutil
import imsize                  # pip install imsize
import imgio                   # pip install imgio
from pqdm.threads import pqdm  # pip install pqdm


class ImageProvider:
    """ A multithreaded image file loader. """

    def __init__(self, files, downsample, verbose=False):
        """ Create a new ImageProvider with the given (hardcoded) FileList instance. """
        self.thread_name = "ImageProviderThread"
        self.downsample = downsample
        self.verbose = verbose
        self.files = files
        self.loader_thread = None
        self.running = False
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
        size_on_disk = 0
        size_in_mem = 0
        if len(self.files.filespecs) > 100:
            print("Scanning images & estimating memory consumption...")
        invalid = []
        for idx, filespec in enumerate(self.files.filespecs):
            if "://" not in filespec:
                try:
                    info = imsize.read(filespec)
                    size_on_disk += info.filesize
                    size_in_mem += info.nbytes
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
        in which case the return value is either "PENDING" (not loaded yet),
        "RELEASED" (already released to free up the memory), or "INVALID"
        (tried to load the image but failed).
        """
        return self.files.images[index]

    def release_image(self, index):
        """
        Release the image at the given index to (eventually) free up the memory.
        """
        self.files.images[index] = "RELEASED"

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
            with self.files.mutex:  # drop/delete not allowed while loading
                nloaded = 0
                for chunk in split():
                    ok = self._check_ram(2048, wait=False)
                    if ok:
                        batch = [(idx, downsample, verbose) for idx in chunk]
                        images = pqdm(batch, self._load_single, 8, "args", bounded=True,
                                      exception_behaviour="immediate", disable=True)
                        for idx, img in zip(chunk, images):
                            if img is not None:
                                self.files.images[idx] = img
                                nloaded += 1
            nbytes = np.sum([img.nbytes for img in self.files.images if isinstance(img, np.ndarray)])
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
            t0 = time.time()
            while self.running:  # load all files
                self._check_ram(2048, wait=True)
                with self.files.mutex:  # avoid race conditions
                    if idx < self.files.numfiles:
                        verbose = self.verbose or self.files.numfiles < 200
                        img = self._load_single(idx, downsample, verbose)
                        if img is not None:
                            self.files.images[idx] = img
                            nbytes += img.nbytes if isinstance(img, np.ndarray) else 0
                    else:
                        break
                time.sleep(0.001)
                idx += 1
                if self.files.reindexed:
                    self.files.reindexed = False
                    idx = 0  # restart from beginning
            if nbytes > 1e4:
                elapsed = time.time() - t0
                nbytes = nbytes / 1024**2
                bandwidth = nbytes / elapsed
                ram_after = psutil.virtual_memory().available / 1024**2
                consumed = ram_before - ram_after
                self._print(f"loaded {idx} files, {nbytes:.0f} MB of image data in {elapsed:.1f} seconds ({bandwidth:.1f} MB/sec).")
                self._print(f"consumed {consumed:.0f} MB of system RAM, {ram_after:.0f}/{ram_total:.0f} MB remaining.")
            time.sleep(0.1)

    def _load_single(self, idx, downsample, verbose):
        """
        Read image, drop alpha channel, convert to fp32 if maxval != 255;
        if loading fails, mark the slot as "INVALID" and keep going.
        """
        if isinstance(self.files.images[idx], str) and self.files.images[idx] == "PENDING":
            try:
                filespec = self.files.filespecs[idx]
                if Path(filespec).suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".ppm"]:
                    self.files.linearize[idx] = True
                if not self.files.is_url[idx]:
                    info = imsize.read(filespec)
                    img, maxval = imgio.imread(filespec, width=info.width, height=info.height, bpp=info.bitdepth, verbose=verbose)
                else:
                    data = urllib.request.urlopen(filespec).read()
                    basename = os.path.basename(filespec)
                    with tempfile.NamedTemporaryFile(suffix=f"_{basename}") as tmpfile:
                        tmpfile.write(data)
                        img, maxval = imgio.imread(tmpfile.name, verbose=verbose)
                img = img[::downsample, ::downsample]
                img = np.atleast_3d(img)  # {2D, 3D} => 3D
                img = img[:, :, :3]  # scrap alpha channel, if any
                if self.files.linearize[idx]:
                    # invert sRGB gamma, as OpenGL texture filtering assumes
                    # linear colors; float16 precision is the minimum to avoid
                    # clipping dark colors to black
                    self.files.linearize[idx] = False
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
            except imgio.ImageIOError as e:
                print(f"\n{e}")
                self._vprint(e)
                return "INVALID"
        return None

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
