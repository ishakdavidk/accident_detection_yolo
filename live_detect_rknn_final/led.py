# SysfsGPIOLed

import os
import time
from threading import Thread


class SysfsGPIOLed:
    """
    Control a header LED via legacy sysfs GPIO.
    Env:
      LED_GPIO_PIN           (int, default 138)
      LED_GPIO_ACTIVE_LOW    (0/1, default 0)
    """
    SYSFS_BASE = "/sys/class/gpio"

    def __init__(self, pin=None, active_low=None):
        self.pin = int(os.getenv("LED_GPIO_PIN", pin if pin is not None else 138))
        self.active_low = (os.getenv("LED_GPIO_ACTIVE_LOW", "0") == "1") if active_low is None else bool(active_low)
        self.gpio_dir = f"{self.SYSFS_BASE}/gpio{self.pin}"
        self.fd = None
        self.ok = False
        self._blink_thread = None

        try:
            if not os.path.exists(self.gpio_dir):
                try:
                    with open(f"{self.SYSFS_BASE}/export", "w") as f:
                        f.write(str(self.pin))
                    time.sleep(0.05)
                except Exception as e:
                    print(f"[LED] export warn (non-fatal): {e}")

            try:
                with open(f"{self.gpio_dir}/direction", "w") as f:
                    f.write("out")
            except Exception as e:
                print(f"[LED] direction warn (non-fatal): {e}")

            self.fd = open(f"{self.gpio_dir}/value", "w", buffering=1)
            self._write(False)  # start OFF
            self.ok = True
            print(f"[LED] sysfs gpio ready: pin={self.pin} active_low={self.active_low}")
        except Exception as e:
            print(f"[LED] gpio init failed for pin {self.pin}: {e}")

    def _write(self, on: bool):
        if not self.fd:
            return
        val = (0 if self.active_low else 1) if on else (1 if self.active_low else 0)
        try:
            self.fd.seek(0)
            self.fd.write("1" if val else "0")
            self.fd.flush()
        except Exception as e:
            print(f"[LED] write failed: {e}")

    def set(self, on: bool):
        if self.ok:
            self._write(on)

    def _blink_worker(self, times: int, interval: float):
        try:
            for _ in range(times):
                self._write(True)
                time.sleep(interval)
                self._write(False)
                time.sleep(interval)
        finally:
            self._write(False)
            self._blink_thread = None

    def blink_async_then_off(self, times=3, interval=0.08):
        """Start a non-blocking blink sequence; ends OFF."""
        if not self.ok:
            return
        if self._blink_thread and self._blink_thread.is_alive():
            return
        t = Thread(target=self._blink_worker, args=(times, interval), daemon=True)
        self._blink_thread = t
        t.start()

    def is_blinking(self):
        return self._blink_thread is not None and self._blink_thread.is_alive()

    def close(self):
        try:
            if self.fd:
                self._write(False)
                self.fd.close()
                self.fd = None
        except Exception:
            pass
