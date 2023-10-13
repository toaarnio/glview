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

    def __init__(self, files, verbose=False):
        """ Create a new ImageProvider with the given (hardcoded) FileList instance. """
        self.thread_name = "ImageProviderThread"
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
        self.loader_thread = threading.Thread(target=lambda: self._try(loader), name=self.thread_name)
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
            if Path(filespec).suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                self.files.linearize[idx] = True
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
        print(f"Found {self.files.numfiles} valid images, consuming {size_on_disk:.0f} MB on disk, {size_in_mem:.0f} MB in memory.")

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

    def _parallel_loader(self):
        t0 = time.time()
        ram_total = psutil.virtual_memory().total / 1024**2
        ram_before = psutil.virtual_memory().available / 1024**2
        nfiles = self.files.numfiles
        verbose = self.verbose or nfiles < 200
        splits = [2, 8, 16] + list(range(32, nfiles, 32))
        chunks = np.split(np.arange(nfiles), splits)
        with self.files.mutex:  # drop/delete not allowed while loading
            for chunk in chunks:
                batch = [(idx, verbose) for idx in chunk]
                images = pqdm(batch, self._load_single, 8, "args", bounded=True,
                              exception_behaviour="immediate", disable=True)
                for idx, img in zip(chunk, images):
                    self.files.images[idx] = img
                time.sleep(0.01)
        nbytes = np.sum([img.nbytes for img in self.files.images if isinstance(img, np.ndarray)])
        if nbytes > 1e4:
            elapsed = time.time() - t0
            nbytes = nbytes / 1024**2
            bandwidth = nbytes / elapsed
            ram_after = psutil.virtual_memory().available / 1024**2
            consumed = ram_before - ram_after
            nfiles = self.files.numfiles
            self._print(f"loaded {nfiles} files, {nbytes:.0f} MB of image data in {elapsed:.1f} seconds ({bandwidth:.1f} MB/sec).")
            self._print(f"consumed {consumed:.0f} MB of system RAM, {ram_after:.0f}/{ram_total:.0f} MB remaining.")

    def _sequential_loader(self):
        """
        Load all images in sequential order.
        """
        waiting_for_ram = False
        ram_total = psutil.virtual_memory().total / 1024**2
        ram_before = psutil.virtual_memory().available / 1024**2
        while self.running:  # loop until program termination
            idx = 0
            nbytes = 0
            t0 = time.time()
            while self.running:  # load all files
                ram_current = psutil.virtual_memory().available / 1024**2
                if ram_current < 2048:  # less than 2 GB remaining => stop loading
                    if not waiting_for_ram:
                        self._print(f"WARNING: Only {ram_current:.0f} MB of RAM remaining. Free up some memory to load more images.")
                        waiting_for_ram = True  # display the warning only once
                    time.sleep(1.0)  # try again once per second
                    continue
                waiting_for_ram = False
                with self.files.mutex:  # avoid race conditions
                    if idx < self.files.numfiles:
                        if isinstance(self.files.images[idx], str) and self.files.images[idx] == "PENDING":
                            verbose = self.verbose or self.files.numfiles < 200
                            img = self._load_single(idx, verbose)
                            self.files.images[idx] = img
                            nbytes += img.nbytes if isinstance(img, np.ndarray) else 0
                    else:
                        break
                time.sleep(0.001)
                idx += 1
            if nbytes > 1e4:
                elapsed = time.time() - t0
                nbytes = nbytes / 1024**2
                bandwidth = nbytes / elapsed
                ram_after = psutil.virtual_memory().available / 1024**2
                consumed = ram_before - ram_after
                self._print(f"loaded {idx} files, {nbytes:.0f} MB of image data in {elapsed:.1f} seconds ({bandwidth:.1f} MB/sec).")
                self._print(f"consumed {consumed:.0f} MB of system RAM, {ram_after:.0f}/{ram_total:.0f} MB remaining.")
            time.sleep(0.1)

    def _load_single(self, idx, verbose):
        """
        Read image, drop alpha channel, convert to fp32 if maxval != 255;
        if loading fails, mark the slot as "INVALID" and keep going.
        """
        if isinstance(self.files.images[idx], str) and self.files.images[idx] == "PENDING":
            try:
                filespec = self.files.filespecs[idx]
                if not self.files.is_url[idx]:
                    info = imsize.read(filespec)
                    img, maxval = imgio.imread(filespec, width=info.width, height=info.height, bpp=info.bitdepth, verbose=verbose)
                else:
                    data = urllib.request.urlopen(filespec).read()
                    basename = os.path.basename(filespec)
                    with tempfile.NamedTemporaryFile(suffix=f"_{basename}") as tmpfile:
                        tmpfile.write(data)
                        img, maxval = imgio.imread(tmpfile.name, verbose=verbose)
                img = np.atleast_3d(img)  # {2D, 3D} => 3D
                img = img[:, :, :3]  # scrap alpha channel, if any
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
