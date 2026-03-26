import sys
import time
from threading import Thread
from queue import Queue

import cv2
import numpy as np

import config
from led import SysfsGPIOLed
from sdcard import resolve_sd_root, ensure_dir
from writer import IMAGE_WRITER
from detector import RKNNAccidentDetector
from recorder import EventRecorder
from camera import frame_reader
import web_server                     # <-- import the whole module
from web_server import run_stream_server  # optional: convenience import


def main():
    # Initialize threshold on startup
    config.load_threshold()

    led = SysfsGPIOLed()
    led.set(False)

    # SD card + events directory
    sd_root = resolve_sd_root(config.SD_ROOT_HINT)
    output_dir = f"{sd_root}/{config.OUTPUT_SUBDIR}"
    ensure_dir(output_dir)

    # Tell web_server where events live
    web_server.set_events_root(output_dir)

    # Detector
    detector = RKNNAccidentDetector(
        config.MODEL_PATH,
        conf_threshold=config.CONF_THRESHOLD_GLOBAL,
        input_size=config.INPUT_SIZE,
        nms_iou=config.NMS_IOU,
    )

    # Start HTTP streaming server in background (if enabled)
    if config.ENABLE_STREAM:
        server_thread = Thread(target=run_stream_server, daemon=True)
        server_thread.start()
    else:
        print("[Stream] Streaming disabled (ENABLE_STREAM != 1)")

    # Camera reader thread
    frame_queue = Queue(maxsize=config.QUEUE_MAX)
    reader_thread = Thread(
        target=frame_reader,
        args=(config.CAMERA_SOURCE, frame_queue, config.CAM_WIDTH, config.CAM_HEIGHT, config.CAM_FPS),
        daemon=True,
    )
    reader_thread.start()

    # Optional UI windows
    if config.SHOW_UI:
        cv2.namedWindow("Live Detection", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Prediction Info", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Live Detection", 100, 100)
        cv2.moveWindow("Prediction Info", 100, 500)

    # Estimate real FPS
    try:
        if isinstance(config.CAMERA_SOURCE, str) and config.CAMERA_SOURCE.startswith("gst:"):
            real_fps = float(config.CAM_FPS)
        else:
            tmp_cap = cv2.VideoCapture(config.CAMERA_SOURCE, cv2.CAP_V4L2)
            real_fps = float(tmp_cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if real_fps <= 0:
                real_fps = float(config.CAM_FPS)
            tmp_cap.release()
    except Exception:
        real_fps = float(config.CAM_FPS)

    N_BEFORE = max(1, int(round(real_fps * config.PREBUFFER_SEC)))
    N_AFTER  = max(1, int(round(real_fps * config.POSTBUFFER_SEC)))
    GAP_TOL  = max(1, int(round(real_fps * config.EVENT_GAP_SEC)))  # currently unused, but fine

    print(f"[Recorder] Using ~{real_fps:.1f} fps -> PRE {N_BEFORE} frames, POST {N_AFTER} frames, GAP_TOL {GAP_TOL} frames")
    print(f"[Recorder] Saving events to: {output_dir}")

    recorder = EventRecorder(
        output_dir=output_dir,
        n_before=N_BEFORE,
        n_after=N_AFTER,
        save_annotated=config.SAVE_ANNOTATED,
        stitch_video=config.STITCH_VIDEO,
        video_fps=real_fps,
        video_codec=config.VIDEO_CODEC,
        writer=IMAGE_WRITER,
        async_stitch=True,
    )

    frame_counter = 0
    last_label = "Normal"
    last_color = (0, 255, 0)
    prob_label = "00.00%"
    detected_boxes = []

    # Startup LED phase
    startup_on = False
    startup_start_ts = 0.0
    startup_finished = False

    print("Starting detection loop... Press 'q' to quit.")

    try:
        while True:
            frame = frame_queue.get()
            if frame is None:
                print("Received stop signal from frame reader. Exiting with error so systemd can retry.")
                sys.exit(2)

            frame_counter += 1
            display_frame = frame.copy()

            # ----- LED startup phase -----
            if not startup_on and not startup_finished:
                led.set(True)
                startup_start_ts = time.monotonic()
                startup_on = True
            elif startup_on and (time.monotonic() - startup_start_ts >= config.STARTUP_LED_SECONDS):
                led.blink_async_then_off(times=3, interval=0.08)
                startup_on = False
                startup_finished = True
            if startup_finished and not led.is_blinking():
                led.set(False)

            # ----- Inference cadence -----
            just_inferred = False
            hud_is_acc = (last_label == "Accident")

            if frame_counter % (config.SKIP_FRAMES + 1) == 1:
                has_accident_now, confidence, detected_boxes, hud_is_acc = detector.detect(frame)
                prob_label = f"{confidence * 100:05.2f}%"
                last_label = "Accident" if hud_is_acc else "Normal"
                last_color = (0, 0, 255) if hud_is_acc else (0, 255, 0)
                just_inferred = True
            else:
                has_accident_now = (last_label == "Accident")
                confidence = float(prob_label.replace("%", "")) / 100.0 if "%" in prob_label else 0.0

            # Accident LED blink (after startup phase finished, and after persistence)
            if startup_finished and just_inferred and hud_is_acc:
                led.blink_async_then_off(times=3, interval=0.06)

            # Draw boxes
            if detected_boxes:
                for x1, y1, x2, y2, conf in detected_boxes:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # HUD text
            (text_w, text_h), baseline = cv2.getTextSize(
                last_label + " - " + prob_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
            )
            frame_h, frame_w = frame.shape[:2]
            x = frame_w - text_w - 50
            y = frame_h - 40
            cv2.putText(display_frame, last_label + " - " + prob_label, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, last_color, 2)

            # ----- update stream frame -----
            if config.ENABLE_STREAM:
                try:
                    ok_jpeg, jpeg_buf = cv2.imencode(".jpg", display_frame)
                    if ok_jpeg:
                        web_server.latest_jpeg = jpeg_buf.tobytes()
                except Exception:
                    pass

            # ----- recorder logic -----
            recorder.push_prebuffer(frame, display_frame)

            # Start event only when a *persistent* accident (hud_is_acc) is detected
            if just_inferred and hud_is_acc and not recorder.is_active():
                recorder.start_event(frame, display_frame)

            if recorder.is_active():
                # While recording, extend when we still see accident in raw predictions
                if has_accident_now:
                    recorder.extend_post_deadline(1)
                _ = recorder.maybe_record_post(frame, display_frame)

            # ----- UI panes -----
            if config.SHOW_UI:
                info_image = np.ones((340, 540, 3), dtype=np.uint8) * 255
                info_lines = [
                    f"Frame #: {frame_counter}",
                    f"HUD Label: {last_label}",
                    f"Accident Prob: {prob_label}",
                    f"Startup: on={startup_on} finished={startup_finished}",
                    f"LED blinking: {led.is_blinking()}",
                    f"Event Active: {recorder.is_active()}",
                    f"Saving to: {output_dir}",
                    f"Stream: {'ON' if config.ENABLE_STREAM else 'OFF'}",
                    f"URL: http://<IP>:{config.STREAM_PORT}/ (web UI)",
                ]
                y0, dy = 28, 30
                for i, line in enumerate(info_lines):
                    y_txt = y0 + i * dy
                    cv2.putText(info_image, line, (10, y_txt),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                resized = cv2.resize(display_frame, (960, 540))
                cv2.imshow("Live Detection", resized)
                cv2.imshow("Prediction Info", info_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        try:
            led.set(False)
            led.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        if reader_thread.is_alive():
            reader_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
