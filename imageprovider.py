import os                      # built-in library
import time                    # built-in library
import threading               # built-in library
import urllib.request          # built-in library
import tempfile                # built-in library
import numpy as np             # pip install numpy
import psutil                  # pip install psutil
import imsize                  # pip install imsize
import imgio                   # pip install imgio


class ImageProviderMT:


    def __init__(self, files, verbose=False):
        self.thread_name = "ImageProviderThread"
        self.verbose = verbose
        self.files = files
        self.loader_thread = None
        self.running = False
        self.estimate_size()

    def start(self):
        self._vprint(f"spawning {self.thread_name}...")
        self.running = True
        self.loader_thread = threading.Thread(target=lambda: self._try(self._image_loader), name=self.thread_name)
        self.loader_thread.daemon = True  # terminate when main process ends
        self.loader_thread.start()


    def stop(self):
        self._vprint(f"killing {self.thread_name}...")
        self.running = False
        self.loader_thread.join()
        self._vprint(f"{self.thread_name} killed")


    def estimate_size(self):
        size_on_disk = 0
        size_in_mem = 0
        if len(self.files.filespecs) > 100:
            print(f"Scanning images & estimating memory consumption...")
        for filespec in self.files.filespecs:
            if "://" not in filespec:
                info = imsize.read(filespec)
                size_on_disk += info.filesize
                size_in_mem += info.nbytes
        size_on_disk /= 1024 ** 2
        size_in_mem /= 1024 ** 2
        print(f"Found {self.files.numfiles} images, consuming {size_on_disk:.0f} MB on disk, {size_in_mem:.0f} MB in memory.")


    def load_image(self, index):
        return self.files.images[index]


    def release_image(self, index):
        self.files.images[index] = "RELEASED"


    def _image_loader(self):
        waiting_for_ram = False
        ram_total = psutil.virtual_memory().total / 1024**2
        ram_before = psutil.virtual_memory().available / 1024**2
        while self.running:  # loop until program termination
            idx = 0
            nbytes = 0
            t0 = time.time()
            while self.running:  # load all files
                ram_current = psutil.virtual_memory().available / 1024**2
                if ram_current < 512:  # less than 512 MB remaining => stop loading
                    if not waiting_for_ram:
                        self._print(f"WARNING: Only {ram_current:.0f} MB of RAM remaining. Free up some memory to load more images.")
                        waiting_for_ram = True  # display the warning only once
                    time.sleep(1.0)  # try again once per second
                    continue
                waiting_for_ram = False
                with self.files.mutex:  # avoid race conditions
                    if idx < self.files.numfiles:
                        if isinstance(self.files.images[idx], str) and self.files.images[idx] == "PENDING":
                            img = self._load_single(idx)
                            self.files.images[idx] = img
                            nbytes += img.nbytes if isinstance(img, np.ndarray) else 0
                    else:
                        break
                time.sleep(0.01)  # yield some CPU time to the UI thread
                idx += 1
            if nbytes > 1e6:
                elapsed = time.time() - t0
                nbytes = nbytes / 1024**2
                bandwidth = nbytes / elapsed
                ram_after = psutil.virtual_memory().available / 1024**2
                consumed = ram_before - ram_after
                self._print(f"loaded {nbytes:.0f} MB of image data in {elapsed:.1f} seconds ({bandwidth:.1f} MB/sec).")
                self._print(f"consumed {consumed:.0f} MB of system RAM, {ram_after:.0f}/{ram_total:.0f} MB remaining.")
            time.sleep(0.1)


    def _load_single(self, idx):
        """
        Read image, drop alpha channel, convert to fp32 if maxval != 255;
        if loading fails, mark the slot as "INVALID" and keep going.
        """
        try:
            filespec = self.files.filespecs[idx]
            if not self.files.is_url[idx]:
                img, maxval = imgio.imread(filespec, verbose=True)
            else:
                data = urllib.request.urlopen(filespec).read()
                basename = os.path.basename(filespec)
                with tempfile.NamedTemporaryFile(suffix=f"_{basename}") as fp:
                    fp.write(data)
                    img, maxval = imgio.imread(fp.name, verbose=True)
            img = np.atleast_3d(img)  # {2D, 3D} => 3D
            img = img[:, :, :3]  # scrap alpha channel, if any
            if maxval != 255:  # if not uint8, convert to fp16 (due to ModernGL limitations)
                norm = max(maxval, np.max(img))
                img = img.astype(np.float32) / norm
                img = img.astype(np.float16)
            return img
        except imgio.ImageIOError as e:
            print(f"\n{e}")
            self._vprint(e)
            return "INVALID"


    def _try(self, func):
        try:
            func()
        except Exception as e:  # pylint: disable=broad-except
            self.running = False
            if self.verbose:
                import traceback
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
