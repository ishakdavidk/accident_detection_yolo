import os
import glob
import cv2
from datetime import datetime
from collections import deque

from sdcard import ensure_dir
from writer import IMAGE_WRITER


class EventRecorder:
    def __init__(self, output_dir, n_before, n_after,
                 save_annotated=True,
                 stitch_video=True,
                 video_fps=30.0,
                 video_codec='mp4v',
                 writer=None,
                 async_stitch=True):

        self.output_dir = output_dir
        ensure_dir(self.output_dir)
        self.n_before = int(max(0, n_before))
        self.n_after = int(max(1, n_after))
        self.save_annotated = save_annotated
        self.stitch_video = stitch_video
        self.video_fps = float(video_fps)
        self.video_codec = video_codec

        self.prebuffer = deque(maxlen=self.n_before)
        self.active = False
        self.frames_after_needed = 0
        self.event_dir = None
        self.frame_idx = 0

        self.writer = writer or IMAGE_WRITER
        self.async_stitch = async_stitch

    def push_prebuffer(self, frame_raw, frame_annotated):
        self.prebuffer.append((
            frame_annotated.copy() if frame_annotated is not None else None,
            frame_raw.copy()
        ))

    def _new_event_dir(self):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
        path = os.path.join(self.output_dir, ts)
        ensure_dir(path)
        return path

    def start_event(self, trigger_frame_raw, trigger_frame_annotated):
        if self.active:
            return
        self.active = True
        self.frames_after_needed = self.n_after
        self.event_dir = self._new_event_dir()
        self.frame_idx = 0

        # Write prebuffer frames
        for ann, raw in list(self.prebuffer):
            img = ann if (self.save_annotated and ann is not None) else raw
            if img is None:
                continue
            fname = os.path.join(self.event_dir, f"{self.frame_idx:06d}.jpg")
            self.writer.save(fname, img)
            self.frame_idx += 1

        trig = trigger_frame_annotated if self.save_annotated else trigger_frame_raw
        fname = os.path.join(self.event_dir, f"{self.frame_idx:06d}.jpg")
        self.writer.save(fname, trig)
        self.frame_idx += 1

    def maybe_record_post(self, frame_raw, frame_annotated):
        if not self.active:
            return False

        if self.frames_after_needed > 0:
            img = frame_annotated if self.save_annotated else frame_raw
            fname = os.path.join(self.event_dir, f"{self.frame_idx:06d}.jpg")
            self.writer.save(fname, img)
            self.frame_idx += 1
            self.frames_after_needed -= 1

            if self.frames_after_needed == 0:
                ed = self.event_dir
                fps = self.video_fps
                self._reset()

                if self.stitch_video:
                    if self.async_stitch:
                        from threading import Thread
                        Thread(target=self._stitch_video, args=(ed, fps), daemon=True).start()
                    else:
                        try:
                            self._stitch_video(ed, fps)
                        except Exception as e:
                            print(f"[WARN] Video stitching failed for {ed}: {e}")

                return True

        return False

    def extend_post_deadline(self, extra_frames=1):
        if self.active:
            self.frames_after_needed = max(
                1, self.frames_after_needed + int(max(1, extra_frames))
            )

    def is_active(self):
        return self.active

    def _reset(self):
        self.active = False
        self.event_dir = None
        self.frame_idx = 0

    def _stitch_video(self, event_dir, fps):
        """
        Stitch JPG frames in event_dir into an H.264 MP4 using ffmpeg.
        Requires 'ffmpeg' installed on the system.
        """
        import subprocess
        import glob
        import os

        jpgs = sorted(glob.glob(os.path.join(event_dir, "*.jpg")))
        if not jpgs:
            print(f"[Recorder] No JPG frames found in {event_dir}, skipping video.")
            return

        # Our frames are named 000000.jpg, 000001.jpg, ... (frame_idx:06d)
        pattern = os.path.join(event_dir, "%06d.jpg")
        out_path = os.path.join(event_dir, "event.mp4")

        cmd = [
            "ffmpeg",
            "-y",  # overwrite if exists
            "-framerate", str(max(1, int(round(fps)))),
            "-i", pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            out_path,
        ]

        print("[Recorder] Running ffmpeg:", " ".join(cmd))
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"[Recorder] Wrote {out_path}")
        except subprocess.CalledProcessError as e:
            print(f"[Recorder] ffmpeg failed for {event_dir}: {e}")

