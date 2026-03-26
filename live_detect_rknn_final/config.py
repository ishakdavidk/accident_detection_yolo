# env config, thresholds, password, camera source

import os

# ============== General flags ==============
SHOW_UI       = os.getenv("SHOW_UI", "1") == "1"
ENABLE_STREAM = os.getenv("ENABLE_STREAM", "1") == "1"
STREAM_PORT   = int(os.getenv("STREAM_PORT", "8000"))

# ============== Threshold config ==============
THRESH_FILE = "/etc/ondai/threshold.txt"

CONF_THRESHOLD_DEFAULT = float(os.getenv("CONF_THRESHOLD", "0.5"))
CONF_THRESHOLD_GLOBAL = CONF_THRESHOLD_DEFAULT  # runtime value


def load_threshold():
    """Load threshold from file if present; otherwise use default and write file."""
    global CONF_THRESHOLD_DEFAULT, CONF_THRESHOLD_GLOBAL

    import os
    try:
        os.makedirs(os.path.dirname(THRESH_FILE), exist_ok=True)
    except Exception:
        pass

    if os.path.isfile(THRESH_FILE):
        try:
            with open(THRESH_FILE, "r", encoding="utf-8") as f:
                raw = f.read().strip()
                val = float(raw)
                if not (0.0 <= val <= 1.0):
                    raise ValueError("out of range")
                CONF_THRESHOLD_DEFAULT = val
                CONF_THRESHOLD_GLOBAL = val
                print(f"[Threshold] Loaded from {THRESH_FILE}: {val:.3f}")
                return
        except Exception as e:
            print(f"[Threshold] Failed to load {THRESH_FILE}, using default {CONF_THRESHOLD_DEFAULT:.3f}: {e}")

    # If we reach here: use default, write file
    CONF_THRESHOLD_GLOBAL = CONF_THRESHOLD_DEFAULT
    try:
        with open(THRESH_FILE, "w", encoding="utf-8") as f:
            f.write(f"{CONF_THRESHOLD_GLOBAL:.3f}\n")
        print(f"[Threshold] Initialized {THRESH_FILE} with {CONF_THRESHOLD_GLOBAL:.3f}")
    except Exception as e:
        print(f"[Threshold] Failed to write {THRESH_FILE}: {e}")


def save_threshold(new_val: float):
    """Save threshold to file and update global value."""
    global CONF_THRESHOLD_GLOBAL
    import os
    CONF_THRESHOLD_GLOBAL = new_val
    try:
        os.makedirs(os.path.dirname(THRESH_FILE), exist_ok=True)
        with open(THRESH_FILE, "w", encoding="utf-8") as f:
            f.write(f"{CONF_THRESHOLD_GLOBAL:.3f}\n")
        print(f"[Threshold] Saved {CONF_THRESHOLD_GLOBAL:.3f} to {THRESH_FILE}")
    except Exception as e:
        print(f"[Threshold] Failed to save threshold: {e}")


# ============== Device password (persistent) ==============
PASSWORD_FILE = "/etc/ondai/password.txt"
DEFAULT_PASSWORD = "123"


def load_password():
    """Load device password (plain text) from file or initialize with default."""
    import os
    try:
        os.makedirs(os.path.dirname(PASSWORD_FILE), exist_ok=True)
    except Exception:
        pass

    if os.path.isfile(PASSWORD_FILE):
        try:
            with open(PASSWORD_FILE, "r", encoding="utf-8") as f:
                pw = f.read().strip()
                if pw:
                    print("[Password] Loaded existing device password.")
                    return pw
        except Exception as e:
            print(f"[Password] Failed to load {PASSWORD_FILE}, using default: {e}")

    # Initialize with default
    try:
        with open(PASSWORD_FILE, "w", encoding="utf-8") as f:
            f.write(DEFAULT_PASSWORD + "\n")
        print(f"[Password] Initialized {PASSWORD_FILE} with default password.")
    except Exception as e:
        print(f"[Password] Failed to write default password: {e}")
    return DEFAULT_PASSWORD


def save_password(new_pw: str):
    """Save new device password to file."""
    import os
    new_pw = (new_pw or "").strip()
    if not new_pw:
        raise ValueError("Password must not be empty")
    try:
        os.makedirs(os.path.dirname(PASSWORD_FILE), exist_ok=True)
        with open(PASSWORD_FILE, "w", encoding="utf-8") as f:
            f.write(new_pw + "\n")
        print("[Password] Updated device password.")
    except Exception as e:
        print(f"[Password] Failed to save new password: {e}")
        raise


def check_password(pw: str) -> bool:
    """Return True if the provided password matches the stored one."""
    stored = load_password()
    return (pw or "").strip() == stored


# ============== Model / camera / recorder config ==============
MODEL_PATH        = 'yolov11s_epoch26.rknn'

CAMERA_SOURCE_ENV = os.getenv("CAMERA_SOURCE", "").strip()


def _parse_camera_source(s: str):
    if not s:
        # Default to Radxa Camera 4K via GStreamer
        return (
            "gst:v4l2src device=/dev/video11 io_utils-mode=4 ! "
            "video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1"
        )
    if s.isdigit():
        return int(s)
    if s.startswith("/dev/video") or s.startswith("/dev/"):
        return s
    try:
        return int(s)
    except Exception:
        return s


CAMERA_SOURCE     = _parse_camera_source(CAMERA_SOURCE_ENV)

CAM_WIDTH   = 1280
CAM_HEIGHT  = 720
CAM_FPS     = 30
SKIP_FRAMES = 2
CONF_THRESHOLD = CONF_THRESHOLD_DEFAULT  # for reference; runtime uses CONF_THRESHOLD_GLOBAL
NMS_IOU     = 0.45
INPUT_SIZE  = 640
QUEUE_MAX   = 3

# ---------- SD card output ----------
SD_ROOT_HINT  = os.getenv("SD_ROOT_HINT", None)
OUTPUT_SUBDIR = "events"

# Time-based saving
PREBUFFER_SEC = 1.0
POSTBUFFER_SEC = 1.0
EVENT_GAP_SEC = 0.25
SAVE_ANNOTATED = True
STITCH_VIDEO   = True
VIDEO_CODEC    = 'mp4v'

# LED timings
STARTUP_LED_SECONDS = float(os.getenv("STARTUP_LED_SECONDS", "30"))
