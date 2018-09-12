import time                    # built-in library
import numpy as np             # pip install numpy
import psutil                  # pip install psutil
import cv2                     # pip install opencv-python


#######################################################################################
#
# StreamProvider
#
#######################################################################################

class StreamProvider(object):
    def __init__(self, files, verbose=False):
        self.verbose = verbose
        self.files = files
        """
        self.cap = cv2.VideoCapture(self.src)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_size = (self.width, self.height)
        self.live_stream = self.num_frames < 0
        self.decoder_thread = None
        self.frame_index = 0
        self.frame_queue = []
        """

    def start(self):
        self._vprint("[StreamProvider] starting...")
        self.running = True
        self.decoder_lock = threading.Lock()
        self.decoder_thread = threading.Thread(target=lambda: self._try(self._frame_loader), name="StreamProviderThread")
        self.decoder_thread.start()
        self._vprint("[StreamProvider] started.")

    def stop(self):
        self._vprint("[StreamProvider] stopping...")
        self.running = False
        self.decoder_thread.join()
        self._vprint("[StreamProvider] stopped.")

    def load_image(self, index):
        while self.images[index] is None:
            time.sleep(0.01)
            if not self.running:
                raise RuntimeError("StreamProvider terminated, cannot load images anymore.")
        return self.images[index]

    def _frame_loader(self):
        while self.running:
            grabbed, frame = self.cap.read()
            if grabbed is True:
                with self.read_lock:
                    slot = len(self.frame_queue)
                    self.frame_queue.append((frame, self.frame_index))
                    self._timeit(t0, "[vcap] pushed frame {} into slot #{}; decoding source frame {}/{}".format(self.frame_index, slot, int(frame_pos), int(self.num_frames)-1))
                    self.frame_index += 1
                if not self.live_stream and self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.num_frames:
                    print("[vcap] End of video reached. Looping back.")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

    def _timeit(self, t0, msg):
        elapsed = (time.time() - t0) * 1000
        self._vprint("{}, took {:.1f} ms".format(msg, elapsed))

    def _vprint(self, message):
        if self.verbose:
            print(message)
