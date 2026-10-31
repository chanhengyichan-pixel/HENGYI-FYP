"""
LedHandler — green/red status LEDs on the Raspberry Pi GPIO.

  green : on while the entrance is in the GRANTED hold
  red   : on while the entrance is in the DENIED hold, and for
          red_hold_seconds after a RESTRICTED_ENTRY zone event

gpiozero
--------
gpiozero works on both Raspberry Pi 4 and Pi 5 (RPi.GPIO does NOT work on
the Pi 5 — that is why it is not used). In laptop mode the pin factory is
switched to gpiozero's MockFactory, so the same code runs on the PC with
simulated pins and LED changes are only logged. If gpiozero is not
installed at all (e.g. a fresh laptop venv), a logging-only stand-in is
used — LEDs are never a reason for the system to fail to start.

Wiring (BCM numbering): GPIO -> 330 ohm resistor -> LED long leg (+),
LED short leg (-) -> GND.

Threading
---------
handle() is called from the entrance pipeline thread and
handle_zone_event() from the room pipeline thread; a lock guards the
red-LED timer and pin writes.
"""

import logging
import threading
import time as _time_module

from alerts.base_handler import BaseHandler
from core.session_controller import SessionState
from core.zone_monitor import RESTRICTED_ENTRY

logger = logging.getLogger(__name__)


class _LogLed:
    """Stand-in used when gpiozero is unavailable: remembers and logs state."""

    def __init__(self, pin):
        self.pin = pin
        self.is_lit = False

    def on(self):
        self.is_lit = True

    def off(self):
        self.is_lit = False

    def close(self):
        pass


def _make_leds(green_pin, red_pin, mode):
    try:
        from gpiozero import Device, LED
        if mode != "pi":
            from gpiozero.pins.mock import MockFactory
            Device.pin_factory = MockFactory()
        return LED(green_pin), LED(red_pin)
    except Exception as e:  # ImportError on a laptop without gpiozero, or no GPIO
        if mode == "pi":
            logger.error("LedHandler: GPIO unavailable (%s) — LEDs disabled.", e)
        else:
            logger.info("LedHandler: simulated LEDs (laptop mode).")
        return _LogLed(green_pin), _LogLed(red_pin)


class LedHandler(BaseHandler):
    def __init__(self, config: dict, clock=_time_module.monotonic, leds=None):
        led_cfg = config["leds"]
        self._red_hold_seconds = float(led_cfg["red_hold_seconds"])
        self._clock = clock
        if leds is None:
            leds = _make_leds(int(led_cfg["green_pin"]), int(led_cfg["red_pin"]),
                              config.get("mode", "laptop"))
        self._green, self._red = leds

        self._lock = threading.Lock()
        self._entrance_state = None
        self._restricted_until = 0.0

    def handle(self, session_state, frame_verdict, frame) -> None:
        with self._lock:
            self._entrance_state = session_state
            self._refresh()

    def handle_zone_event(self, event, frame) -> None:
        if event.event_type != RESTRICTED_ENTRY:
            return
        with self._lock:
            self._restricted_until = self._clock() + self._red_hold_seconds
            self._refresh()
        # Turn the red LED off again when the hold ends, even if the
        # entrance pipeline (whose frames normally refresh the LEDs) is
        # disabled.
        timer = threading.Timer(self._red_hold_seconds + 0.05, self._timed_refresh)
        timer.daemon = True
        timer.start()

    def _timed_refresh(self) -> None:
        with self._lock:
            self._refresh()

    def _refresh(self) -> None:
        green = self._entrance_state is SessionState.GRANTED
        red = (self._entrance_state is SessionState.DENIED
               or self._clock() < self._restricted_until)
        self._set(self._green, green, "green")
        self._set(self._red, red, "red")

    @staticmethod
    def _set(led, lit: bool, name: str) -> None:
        if bool(led.is_lit) == lit:
            return
        led.on() if lit else led.off()
        logger.info("LED %s %s", name, "ON" if lit else "OFF")

    def close(self) -> None:
        for led in (self._green, self._red):
            try:
                led.off()
                led.close()
            except Exception:
                pass
