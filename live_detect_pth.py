import os
os.environ["TORCHVISION_DISABLE_VIDEO"] = "1"  # avoid PyAV/FFmpeg DLL issues on Windows

import cv2
import torch
from ultralytics import YOLO
import numpy as np
from collections import deque


class AccidentDetector:
    def __init__(self, model_path, conf_threshold=0.5, high_confidence=0.7, cooldown_time=3):
        """
        :param model_path: path to YOLO *.pt
        :param conf_threshold: your decision threshold (>= this is 'Accident')
        :param high_confidence: immediate trigger threshold
        :param cooldown_time: frames to keep 'Accident' label after trigger
        """
        self.model = YOLO(model_path)
        self.conf_threshold = float(conf_threshold)
        self.prediction_buffer = deque(maxlen=3)  # Store last 3 predictions (0/1)
        self.accident_cooldown = 0
        self.cooldown_time = int(cooldown_time)
        self.high_confidence = float(high_confidence)

    def detect(self, frame):
        """
        Run detection on a frame.
        - We ask YOLO for *all* boxes (conf ~ 0) so we can *see* low probs.
        - We still *decide* accident using self.conf_threshold (e.g., 0.5).
        Returns: (is_accident, max_conf_accident_class, detected_boxes)
                 detected_boxes: list of (x1, y1, x2, y2, conf) for class 0
        """
        # Request virtually all predictions so sub-0.5 confs are visible
        results = self.model(frame, conf=0.001, verbose=False)[0]

        has_accident = False
        max_conf = 0.0
        detected_boxes = []

        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls)
                if cls == 0:  # Accident class index
                    conf = float(box.conf.item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    detected_boxes.append((x1, y1, x2, y2, conf))
                    max_conf = max(max_conf, conf)

                    # Only COUNT as accident if conf >= threshold
                    if conf >= self.conf_threshold:
                        has_accident = True

        # Update temporal buffer (majority over last few frames)
        self.prediction_buffer.append(1 if has_accident else 0)

        # Decision logic with high-confidence fast path + majority + cooldown
        if max_conf >= self.high_confidence:
            self.accident_cooldown = self.cooldown_time
            return True, max_conf, detected_boxes

        if sum(self.prediction_buffer) >= 2:
            self.accident_cooldown = self.cooldown_time
            return True, max_conf, detected_boxes

        if self.accident_cooldown > 0:
            self.accident_cooldown -= 1
            return True, max_conf, detected_boxes

        return False, max_conf, detected_boxes


def main():
    # ---- Paths ----
    model_path = 'runs/train/best/accident_detection_yolo11s_finetune/weights/epoch77.pt'
    video_path = '../dataset/yt_dashcam_accident/accidentno2.mp4'  # or 0 for webcam

    # ---- Create detector ----
    # conf_threshold is your decision line (Accident only if >= 0.5)
    detector = AccidentDetector(model_path, conf_threshold=0.5, high_confidence=0.7, cooldown_time=3)

    # ---- Video setup ----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video source: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps and fps > 0 else 33

    cv2.namedWindow("Live Detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Prediction Info", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Live Detection", 100, 100)
    cv2.moveWindow("Prediction Info", 100, 500)

    frame_counter = 0
    SKIP_FRAMES = 2  # Only run model every (SKIP_FRAMES+1) frames; display every frame

    last_label = "Normal"
    last_color = (0, 255, 0)
    prob_label = "00.00%"
    detected_boxes = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_counter += 1

            # Run inference on every Nth frame
            if frame_counter % (SKIP_FRAMES + 1) == 1:
                is_accident, confidence, detected_boxes = detector.detect(frame)
                prob_label = f"{confidence * 100:05.2f}%"
                if is_accident:
                    last_label = "Accident"
                    last_color = (0, 0, 255)
                else:
                    last_label = "Normal"
                    last_color = (0, 255, 0)

            # Draw last detections (even on skipped frames)
            # Color by confidence: red if >= decision threshold, gray otherwise
            for x1, y1, x2, y2, conf in detected_boxes:
                color = (0, 0, 255) if conf >= detector.conf_threshold else (128, 128, 128)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Show per-box conf
                txt = f"{conf*100:.1f}%"
                cv2.putText(frame, txt, (x1, max(y1 - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # HUD
            (text_w, text_h), baseline = cv2.getTextSize(
                last_label + " - " + prob_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
            )
            frame_h, frame_w = frame.shape[:2]
            x = max(frame_w - text_w - 50, 10)
            y = max(frame_h - 40, text_h + 10)
            cv2.putText(frame, last_label + " - " + prob_label, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, last_color, 2, cv2.LINE_AA)

            # Info panel
            info_image = np.ones((300, 460, 3), dtype=np.uint8) * 255
            info_lines = [
                f"Frame #: {frame_counter}",
                f"Label: {last_label}",
                f"Accident Probability (max): {prob_label}",
                f"Pred Buffer: {list(detector.prediction_buffer)}",
                f"Cooldown: {detector.accident_cooldown}",
                f"Decision thr: {detector.conf_threshold:.2f}  (>= red box)",
            ]
            y0, dy = 30, 30
            for i, line in enumerate(info_lines):
                y_line = y0 + i * dy
                cv2.putText(info_image, line, (10, y_line),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

            # Show
            display_frame = cv2.resize(frame, (960, 540))
            cv2.imshow("Live Detection", display_frame)
            cv2.imshow("Prediction Info", info_image)

            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
