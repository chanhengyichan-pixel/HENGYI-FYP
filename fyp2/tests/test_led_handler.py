"""Unit tests for alerts/led_handler.py — LEDs are injected fakes."""

from core.session_controller import SessionState
from core.zone_monitor import HAZARD_PPE, RESTRICTED_ENTRY, ZoneEvent
from alerts.led_handler import LedHandler, _LogLed

CONFIG = {"mode": "laptop",
          "leds": {"enabled": True, "green_pin": 17, "red_pin": 22,
                   "red_hold_seconds": 5.0}}


def make_handler():
    clock = [0.0]
    green, red = _LogLed(17), _LogLed(22)
    h = LedHandler(CONFIG, clock=lambda: clock[0], leds=(green, red))
    return h, green, red, clock


def event(event_type):
    return ZoneEvent(event_type=event_type, zone="Z", track_id=1)


class TestEntrance:
    def test_granted_lights_green_only(self):
        h, green, red, _ = make_handler()
        h.handle(SessionState.GRANTED, None, None)
        assert green.is_lit and not red.is_lit

    def test_denied_lights_red_only(self):
        h, green, red, _ = make_handler()
        h.handle(SessionState.DENIED, None, None)
        assert red.is_lit and not green.is_lit

    def test_leds_off_after_hold(self):
        h, green, red, _ = make_handler()
        h.handle(SessionState.GRANTED, None, None)
        h.handle(SessionState.COOLDOWN, None, None)
        assert not green.is_lit and not red.is_lit


class TestRestricted:
    def test_restricted_event_lights_red_for_hold(self):
        h, green, red, clock = make_handler()
        h.handle_zone_event(event(RESTRICTED_ENTRY), None)
        assert red.is_lit
        clock[0] = 4.9
        h.handle(SessionState.IDLE, None, None)
        assert red.is_lit
        clock[0] = 5.1
        h.handle(SessionState.IDLE, None, None)
        assert not red.is_lit

    def test_hazard_event_does_not_touch_leds(self):
        h, green, red, _ = make_handler()
        h.handle_zone_event(event(HAZARD_PPE), None)
        assert not red.is_lit and not green.is_lit

    def test_granted_during_restricted_hold_shows_both(self):
        # Entrance and room are independent: green for the door, red for the room.
        h, green, red, _ = make_handler()
        h.handle_zone_event(event(RESTRICTED_ENTRY), None)
        h.handle(SessionState.GRANTED, None, None)
        assert green.is_lit and red.is_lit


class TestConstruction:
    def test_laptop_mode_builds_without_gpio(self):
        h = LedHandler(CONFIG)          # real factory path: MockFactory or _LogLed
        h.handle(SessionState.DENIED, None, None)
        h.close()


# ---------------------------------------------------------------------------
# Separate room LED (leds.room_red_pin)
# ---------------------------------------------------------------------------

def make_room_handler():
    clock = [0.0]
    green, red, room = _LogLed(17), _LogLed(22), _LogLed(27)
    h = LedHandler(CONFIG, clock=lambda: clock[0], leds=(green, red, room))
    return h, green, red, room, clock


class TestRoomLed:
    def test_restricted_lights_room_led_not_entrance(self):
        h, green, red, room, _ = make_room_handler()
        h.handle_zone_event(event(RESTRICTED_ENTRY), None)
        assert room.is_lit and not red.is_lit and not green.is_lit

    def test_hazard_lights_room_led(self):
        h, green, red, room, _ = make_room_handler()
        h.handle_zone_event(event(HAZARD_PPE), None)
        assert room.is_lit and not red.is_lit

    def test_room_led_off_after_hold(self):
        h, green, red, room, clock = make_room_handler()
        h.handle_zone_event(event(RESTRICTED_ENTRY), None)
        clock[0] = 5.1
        h.handle(SessionState.IDLE, None, None)
        assert not room.is_lit

    def test_entrance_denied_does_not_light_room_led(self):
        h, green, red, room, _ = make_room_handler()
        h.handle(SessionState.DENIED, None, None)
        assert red.is_lit and not room.is_lit

    def test_entrance_and_room_independent(self):
        h, green, red, room, _ = make_room_handler()
        h.handle(SessionState.GRANTED, None, None)
        h.handle_zone_event(event(RESTRICTED_ENTRY), None)
        assert green.is_lit and room.is_lit and not red.is_lit

    def test_room_pin_from_config_laptop(self):
        cfg = {"mode": "laptop", "leds": dict(CONFIG["leds"], room_red_pin=27)}
        h = LedHandler(cfg)
        h.handle_zone_event(event(HAZARD_PPE), None)
        assert h._room_red.is_lit
        h.close()
