import cv2
import time
from queue import Full, Empty


def frame_reader(camera_source, frame_queue, width, height, fps):
    from config import CAM_FPS  # used for negotiated_fps fallback

    print(f"[Camera] Opening source: {camera_source}")

    if isinstance(camera_source, str) and camera_source.startswith("gst:"):
        pipeline = camera_source[len("gst:"):]
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(camera_source, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("[Camera] CAP_V4L2 failed, retrying with default backend...")
            cap.release()
            cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        print(f"Error: Could not open camera: {camera_source}.")
        frame_queue.put(None)
        return

    if not (isinstance(camera_source, str) and camera_source.startswith("gst:")):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

    ok, test = cap.read()
    if not ok or test is None:
        print("Error: Could not read initial frame.")
        frame_queue.put(None)
        cap.release()
        return

    h, w = test.shape[:2]
    negotiated_fps = cap.get(cv2.CAP_PROP_FPS)
    if negotiated_fps is None or negotiated_fps <= 0:
        negotiated_fps = -1.0
    try:
        nfps_str = f"{negotiated_fps:.1f}"
    except Exception:
        nfps_str = str(negotiated_fps)
    print(f"Camera {camera_source} actual: {w}x{h} @ ~{nfps_str} fps")

    try:
        frame_queue.put_nowait(test)
    except Full:
        pass

    from queue import Full as QFull, Empty as QEmpty
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            frame_queue.put(None)
            break
        try:
            frame_queue.put_nowait(frame)
        except QFull:
            try:
                _ = frame_queue.get_nowait()
            except QEmpty:
                pass
            try:
                frame_queue.put_nowait(frame)
            except QFull:
                pass
    cap.release()
