# Edge AI Industrial Safety Monitoring

PPE (helmet + vest) compliance monitoring with YOLOv8 on a Raspberry Pi, no cloud, no face recognition, no personal data.

- **Entrance camera** — PPE check point. A person stands in the marked square; after ~1.5 s of votes the system decides **ALLOW** (green LED, voice) or **DENY** (red LED, voice).
- **Room camera** — zone monitoring with person tracking (ByteTrack):
  - **Hazard zone**: helmet or vest missing for ~2 s → voice warning, snapshot, `HAZARD_PPE` event.
  - **Restricted zone**: anyone inside for ~1 s → voice, red LED, snapshot, `RESTRICTED_ENTRY` event (highest severity).
  - Same person + same event type: at most once per 30 s.
- **Dashboard** (Flask) — both live streams with zone overlay, live status, event history with filter and snapshots.
- **Storage** — SQLite `output/events.db` (event type, camera, zone, severity, violations, snapshot path).

Everything runs the same on the laptop and on the Pi; only the config differs (`config/config.yaml` vs `config/config_pi.yaml`).

## Project layout

```
main.py                    entry point (python main.py [--config ...])
config/config.yaml         laptop config (webcam / video files, simulated LEDs)
config/config_pi.yaml      Raspberry Pi config (CSI cameras, real GPIO, imgsz 320)
core/camera.py             capture thread; backends: opencv (webcam/video/USB), picamera2 (CSI)
core/detector.py           YOLO detect() and track()
core/decision_engine.py    per-person helmet/vest attribution
core/session_controller.py entrance ALLOW/DENY state machine (time-based window)
core/pipeline.py           entrance pipeline
core/zones.py              zone polygons, foot-point test
core/zone_monitor.py       hazard/restricted timers, grace, cooldown
core/room_pipeline.py      room pipeline (track -> PPE -> zones -> alerts)
alerts/                    screenshot, voice (aplay), LEDs (gpiozero), event log
storage/event_store.py     SQLite events (+ automatic migration of old databases)
dashboard/                 Flask app + templates
scripts/zone_picker.py     click zone polygons on a real room frame
scripts/make_voice_prompts.sh  generate the 4 voice prompts (espeak-ng)
verify_camera.py           grab one frame from a camera (works over SSH)
```

## Laptop setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash scripts/make_voice_prompts.sh      # needs: sudo apt install espeak-ng
python -m pytest -q                      # all tests should pass
python main.py                           # dashboard on http://localhost:5000
```

The model weights (`models/exp01_roboflow/weights/best.pt`) are not in git (`*.pt` is ignored) — copy them manually to every machine.

### Testing the room camera without a second camera

Record a short video of yourself walking into taped zones with and without helmet/vest, save it as `videos/room_test.mp4`, then in `config/config.yaml` set `cameras.room.enabled: true`. Video files loop and play at real-time speed, so the 1 s / 2 s rules behave as they would live. Draw the zones on a frame of that video:

```bash
python scripts/zone_picker.py videos/room_test.mp4
```

and paste the printed polygons into `zones.areas`.

## Raspberry Pi setup (Pi 4 or Pi 5, Raspberry Pi OS 64-bit Bookworm)

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-gpiozero espeak-ng git
git clone <your repo> && cd <repo>/fyp2
# --system-site-packages: lets the venv see apt's picamera2 and gpiozero
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
# copy best.pt from the laptop, e.g.:
#   scp models/exp01_roboflow/weights/best.pt pi@<pi-ip>:~/<repo>/fyp2/models/exp01_roboflow/weights/
# and the pose model used by the wear check (downloaded on the laptop the first run):
#   scp yolov8n-pose.pt pi@<pi-ip>:~/<repo>/fyp2/
bash scripts/make_voice_prompts.sh
```

Do **not** copy the laptop's `venv/` to the Pi (x86 vs ARM packages).

**Cameras.** `rpicam-hello --list-cameras` should list them. Pi 5 has two CSI ports (entrance = 0, room = 1). Pi 4 has only one CSI port — use a USB webcam for the room camera (`backend: opencv`, `source: 0`). Check each one with `python verify_camera.py entrance` / `python verify_camera.py room`.

**USB speaker as default audio device** (Pi 5 has no 3.5 mm jack):

```bash
aplay -l                                  # note the USB card number, e.g. card 2
printf 'defaults.pcm.card 2\ndefaults.ctl.card 2\n' > ~/.asoundrc
aplay assets/audio/granted.wav            # should play on the speaker
```

**LEDs** (BCM numbering): GPIO17 → 330 Ω → green LED (+), LED (−) → GND; GPIO22 → 330 Ω → red LED (+), LED (−) → GND.

**Faster model (NCNN).** On the Pi, export once and point the config at the folder:

```bash
yolo export model=models/exp01_roboflow/weights/best.pt format=ncnn imgsz=320
# config_pi.yaml -> detector.weights: "models/exp01_roboflow/weights/best_ncnn_model"
```

`detector.imgsz` must match the export size. The pose model can be exported the same way (`yolo export model=yolov8n-pose.pt format=ncnn imgsz=320`, then `decision_engine.pose_check.weights: "yolov8n-pose_ncnn_model"`). It only runs on frames where a helmet or vest was already accepted. Then:

```bash
python main.py --config config/config_pi.yaml
```

Open `http://<pi-ip>:5000` from the laptop. Watch the `Pipeline FPS` / `Room pipeline FPS` log lines: if each stays at 2 FPS or more, the CPU is enough; if not, try `imgsz: 256`, or consider the AI HAT+ (Pi 5 only).

## Calibration notes

- `decision_engine` head/torso ratios are estimates; tune them on real footage from the mounted camera (`debug: true` + `debug_overlay: true` temporarily).
- `decision_engine.pose_check` rejects PPE that is held instead of worn (hand on the vest at chest height, vest not over the shoulders, helmet beside or in front of the face). At the entrance, stand with **arms down**. The log prints `Pose wear check: not worn — ...` with the rule that fired.
- `zones.grace_seconds` must be longer than the time between processed frames (0.5 s at 2 FPS).
- Zones that reach the bottom of the image should extend past it (`zone_picker.py` does this) — see the comment in `config.yaml`.
