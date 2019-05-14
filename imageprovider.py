import os                      # built-in library
import time                    # built-in library
import threading               # built-in library
import urllib.request          # built-in library
import tempfile                # built-in library
import numpy as np             # pip install numpy
import psutil                  # pip install psutil
import imgio                   # pip install imgio


class ImageProviderMT(object):

    def __init__(self, files, verbose=False):
        self.thread_name = "ImageProviderThread"
        self.verbose = verbose
        self.files = files
        self.images = [None] * self.files.numfiles
        self.loader_thread = None
        self.running = False

    def start(self):
        self._vprint(f"spawning {self.thread_name}...")
        self.running = True
        self.loader_thread = threading.Thread(target=lambda: self._try(self._image_loader), name=self.thread_name)
        self.loader_thread.start()

    def stop(self):
        self._vprint(f"killing {self.thread_name}...")
        self.running = False
        self.loader_thread.join()
        self._vprint(f"{self.thread_name} killed")

    def load_image(self, index):
        if self.files.is_video[index]:
            raise RuntimeError("File {} is not an image.".format(self.files.filespecs[index]))
        while self.images[index] is None:
            time.sleep(0.01)
            if not self.running:
                raise RuntimeError("ImageProvider terminated, cannot load images anymore.")
        return self.images[index]

    def release_image(self, index):
        self.images[index] = "RELEASED"

    def _image_loader(self):
        ram_total = psutil.virtual_memory().total / 1024**2
        ram_before = psutil.virtual_memory().available / 1024**2
        while self.running:
            t0 = time.time()
            nbytes = 0
            for i, filespec in enumerate(self.files.filespecs):
                time.sleep(0.01)
                if self.running and self.images[i] is None and not self.files.is_video[i]:
                    # read image, drop alpha channel, convert to fp32 if maxval != 255;
                    # if loading fails, mark the slot as "INVALID" and keep going
                    try:
                        if not self.files.is_url[i]:
                            img, maxval = imgio.imread(filespec, verbose=True)
                        else:
                            data = urllib.request.urlopen(filespec).read()
                            basename = os.path.basename(filespec)
                            with tempfile.NamedTemporaryFile(suffix=f"_{basename}") as fp:
                                fp.write(data)
                                img, maxval = imgio.imread(fp.name, verbose=True)
                        img.shape = img.shape[:2] + (-1,)  # {2D, 3D} => 3D
                        img = img[:, :, :3]  # scrap alpha channel, if any
                        if maxval != 255:  # if not uint8, convert to fp32
                            img = img.astype(np.float32, copy=False)
                            norm = max(maxval, np.max(img))
                            img = img / norm
                        self.images[i] = img
                        nbytes += img.nbytes
                    except imgio.ImageIOError as e:
                        print(f"\n{e}")
                        self._vprint(e)
                        self.images[i] = "INVALID"
                if not self.running:
                    break
            if nbytes > 1e6:
                elapsed = time.time() - t0
                nbytes = nbytes / 1024**2
                bandwidth = nbytes / elapsed
                ram_after = psutil.virtual_memory().available / 1024**2
                consumed = ram_before - ram_after
                print(f"[ImageProvider] loaded {nbytes:.0f} MB of image data in {elapsed:.1f} seconds ({bandwidth:.1f} MB/sec).")
                print(f"[ImageProvider] consumed {consumed:.0f} MB of system RAM, {ram_after:.0f}/{ram_total:.0f} MB remaining.")

    def _try(self, func):
        try:
            func()
        except Exception as e:
            self.running = False
            if self.verbose:
                import traceback
                self._vprint(f"exception in {func.__name__}():")
                traceback.print_exc()
            else:
                print(f"[{self.__class__.__name__}/{threading.current_thread().name}] {type(e).__name__}: {e}")

    def _vprint(self, message, **kwargs):
        if self.verbose:
            print(f"[{self.__class__.__name__}/{threading.current_thread().name}] {message}", **kwargs)
