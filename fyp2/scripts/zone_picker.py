"""
Click out zone polygons on a real room-camera frame and print them in the
format config.yaml expects.

Usage (from the fyp2/ directory, on a machine with a display):
    python scripts/zone_picker.py output/camera_test_room.jpg
    python scripts/zone_picker.py videos/room_test.mp4      # first frame

Get the image on the Pi with:  python verify_camera.py room

Controls:
    left click  add a corner
    n           finish the current polygon, start the next one
    u           undo the last corner
    q / Esc     quit and print every polygon

The image must be the SAME resolution as the room camera in config.yaml
(width x height), because polygons are stored in pixel coordinates.

Clicks within 10 px of the image border are snapped to 20 px OUTSIDE the
border. A person close to the camera has their box cut off at the bottom
of the image, so their foot point sits exactly on the last pixel row; a
zone that should reach the bottom of the picture must extend past it.
"""

import sys

import cv2

COLORS = [(0, 170, 245), (60, 60, 230), (0, 200, 0), (255, 140, 0)]
SNAP_PX = 10
OUTSIDE_PX = 20


def snap(v, size):
    if v < SNAP_PX:
        return -OUTSIDE_PX
    if v > size - 1 - SNAP_PX:
        return size - 1 + OUTSIDE_PX
    return v


def load_frame(path):
    frame = cv2.imread(path)
    if frame is None:
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            sys.exit(f"Could not read an image or video frame from {path}")
    return frame


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    base = load_frame(sys.argv[1])
    polygons = [[]]

    h, w = base.shape[:2]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            polygons[-1].append([snap(x, w), snap(y, h)])

    cv2.namedWindow("zone picker")
    cv2.setMouseCallback("zone picker", on_mouse)
    print(f"Frame size: {base.shape[1]}x{base.shape[0]}")

    while True:
        frame = base.copy()
        for i, poly in enumerate(polygons):
            color = COLORS[i % len(COLORS)]
            for p in poly:
                cv2.circle(frame, (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1)),
                           4, color, -1)
            if len(poly) > 1:
                pts = [tuple(p) for p in poly]
                for a, b in zip(pts, pts[1:] + ([pts[0]] if i < len(polygons) - 1 else [])):
                    cv2.line(frame, a, b, color, 2)
        cv2.imshow("zone picker", frame)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("n") and len(polygons[-1]) >= 3:
            polygons.append([])
        if key == ord("u") and polygons[-1]:
            polygons[-1].pop()

    cv2.destroyAllWindows()
    print("\nPaste into config.yaml under zones.areas (set name and type):")
    for i, poly in enumerate(p for p in polygons if len(p) >= 3):
        print(f'    - name: "Zone {i + 1}"')
        print("      type: hazard        # or restricted")
        print(f"      polygon: {poly}")


if __name__ == "__main__":
    main()
