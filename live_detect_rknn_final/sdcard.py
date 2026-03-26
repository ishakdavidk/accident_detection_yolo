# SD card detection / resolve_sd_root

import os
import shutil
import getpass


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _is_writable(path: str) -> bool:
    try:
        ensure_dir(path)
        testfile = os.path.join(path, ".write_test.tmp")
        with open(testfile, "wb") as f:
            f.write(b"ok")
        os.remove(testfile)
        return True
    except Exception:
        return False


def _is_mount(path: str) -> bool:
    try:
        return os.path.ismount(path)
    except Exception:
        return False


def _disk_ok(path: str, min_free_mb: int = 50) -> bool:
    try:
        total, used, free = shutil.disk_usage(path)
        return (free // (1024 * 1024)) >= min_free_mb
    except Exception:
        return True


def resolve_sd_root(hint: str | None = None) -> str:
    user = getpass.getuser() or os.getenv("USER") or "user"

    def check_candidate(p: str) -> str | None:
        if os.path.isdir(p) and _is_mount(p) and _is_writable(p) and _disk_ok(p):
            return p
        return None

    # 1) If hint is set, require it to work
    if hint:
        c = check_candidate(os.path.expandvars(os.path.expanduser(hint)))
        if c:
            print(f"[SD] Using hinted SD root: {c}")
            return c
        else:
            print(f"[SD-ERROR] Hint path not usable or not a mounted SD card: {hint}")
            raise RuntimeError("SD card not available")

    # 2) Auto-detect mounted SD-like devices
    bases = [f"/media/{user}", f"/run/media/{user}", "/mnt", "/media"]
    prefer_tokens = ("sd", "mmc", "card", "disk", "usb", "flash")

    candidates = []
    for base in bases:
        if not os.path.isdir(base):
            continue
        try:
            for name in sorted(os.listdir(base)):
                full = os.path.join(base, name)
                if os.path.isdir(full) and _is_mount(full):
                    candidates.append(full)
        except Exception:
            pass

    def score(p: str) -> int:
        name = os.path.basename(p).lower()
        return sum(tok in name for tok in prefer_tokens)

    candidates.sort(key=score, reverse=True)
    for c in candidates:
        usable = check_candidate(c)
        if usable:
            print(f"[SD] Auto-detected SD root: {usable}")
            return usable

    print("[SD-ERROR] No mounted, writable SD card found — exiting.")
    raise RuntimeError("SD card not found")
