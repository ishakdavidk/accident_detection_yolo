# AsyncImageWriter and global IMAGE_WRITER

from queue import Queue, Full
from threading import Thread
import cv2


class AsyncImageWriter:
    """
    Simple async disk writer for frames.
    Main thread just enqueues (path, image); a worker thread does cv2.imwrite().
    """

    def __init__(self, max_queue=512):
        self.q = Queue(maxsize=max_queue)
        self._worker = Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            path, img = item
            try:
                cv2.imwrite(path, img)
            except Exception as e:
                print(f"[Writer] Failed to write {path}: {e}")
            finally:
                self.q.task_done()

    def save(self, path, img):
        """Enqueue an image for writing; drops frame if queue is full."""
        try:
            self.q.put_nowait((path, img.copy()))
        except Full:
            print("[Writer] queue full; dropping frame:", path)


IMAGE_WRITER = AsyncImageWriter()
