import time                    # built-in library
import threading               # built-in library
import multiprocessing         # built-in library
import numpy as np             # pip install numpy
import psutil                  # pip install psutil
import imgio                   # pip install imgio


#######################################################################################
#
# ImageProviderMT -- ImageProvider running in a separate thread
#
#######################################################################################

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
        if not self.files.is_image[index]:
            raise RuntimeError("File {} is not an image.".format(self.files.filespecs[index]))
        while self.images[index] is None:
            time.sleep(0.01)
            if not self.running:
                raise RuntimeError("ImageProvider terminated, cannot load images anymore.")
        return self.images[index]

    def release_image(self, index):
        self.images[index] = "RELEASED"

    def _image_loader(self):
        self._vprint("total physical memory: {:.1f}M".format(psutil.virtual_memory().total / 1024**2))
        self._vprint("available physical memory: {:.1f}M".format(psutil.virtual_memory().available / 1024**2))
        while self.running:
            t0 = time.time()
            nbytes = 0
            for i, filespec in enumerate(self.files.filespecs):
                time.sleep(0.001)
                if self.images[i] is None and self.files.is_image[i] and self.running:
                    # read image, drop alpha channel, rescale to uint8
                    img, maxval = imgio.imread(filespec, verbose=True)
                    img.shape = img.shape[:2] + (-1,)  # {2D, 3D} => 3D
                    img = img[:, :, :3]  # scrap alpha channel, if any
                    img = (img / maxval * 255 + 0.5) if maxval != 255 else img
                    img = img.astype(np.uint8, copy=False)
                    self.images[i] = img
                    nbytes += img.nbytes
                    self._vprint("available physical memory: {:.1f}M".format(psutil.virtual_memory().available / 1024**2))
            if nbytes > 1e6:
                elapsed = time.time() - t0
                nbytes = nbytes / 1024**2
                bandwidth = nbytes / elapsed
                print("[ImageProvider] copied {:.0f} MB of image data from disk to RAM in {:.1f} seconds ({:.1f} MB/sec).".format(nbytes, elapsed, bandwidth))

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

    def _vprint(self, message):
        if self.verbose:
            print(f"[{self.__class__.__name__}/{threading.current_thread().name}] {message}")
