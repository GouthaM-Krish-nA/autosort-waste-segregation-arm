import json
import os
import time
import threading
import keras
import collections
from datetime import datetime

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS

# =====================================================================
# CONFIGURATION - the only part you normally need to edit
# =====================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.normpath(os.path.join(HERE, "..", "frontend"))
MODEL_FILE = os.path.normpath(os.path.join(HERE, "..", "ml", "waste_classifier.h5"))

CATEGORIES = ["plastic", "metal", "paper"]   # same order as train_classifier.py
IMG_SIZE = (160, 160)                         # same size as training
CAMERA_INDEX = 0

SIMULATE_HARDWARE = False     # True = no arm needed, commands are only logged. Set False for the real arm.
ARM_IP = os.environ.get("ARM_IP", "192.168.4.1")   # ESP32 hotspot address; change for STA mode
ARM_URL = "http://" + ARM_IP
ARM_MOVE_TIMEOUT = 240       # seconds to wait for a full pick-and-place (slow speed = over a minute; the ESP32 answers when done)
SIMULATED_MOVE_SECONDS = 6   # how long a pretend pick-and-place takes

VAL_ACCURACY = 91.7          # from your training run
VAL_IMAGES = 120             # 40 validation images x 3 classes

# "Is something lying on the pick spot?" (compares the live frame with an empty-spot photo)
CHANGE_THRESHOLD = 30        # pixel change (0-255) that counts as "different"
MIN_OBJECT_AREA = 0.015      # object must cover at least 1.5% of the frame
MAX_OBJECT_AREA = 0.60       # bigger than this = lighting change or a hand, ignore it
SETTLE_SECONDS = 0.6         # object must stay still this long before we classify it

BIN_NAMES = {"Plastic": "Bin A", "Paper": "Bin B", "Metal": "Bin C"}

# Words the dashboard may send. The arm link turns each one into an ESP32 call.
ARM_COMMANDS = {"PLASTIC", "METAL", "PAPER", "HOME", "GRIPPER_OPEN", "GRIPPER_CLOSE"}
BIN_OF_WORD = {"PAPER": "north", "METAL": "east", "PLASTIC": "west"}   # same bins as the sketch

# ---- full sorting cycle: GRAB -> LIFT -> SHOW CAM -> classify -> DROP ----
# The webcam looks at the SHOW CAM pose (the side opposite the pickup spot).
CYCLE_MIN_CONFIDENCE = 60.0          # below this the item is NOT dropped: you choose the bucket
SHOW_SETTLE_SECONDS = 1.0            # wait after the arm reaches SHOW CAM before looking
SHOW_VOTE_FRAMES = 5                 # frames averaged for one decision
SHOW_VOTE_GAP = 0.15                 # seconds between those frames
SHOW_ROI = (0.20, 0.10, 0.80, 0.85)  # where the held item appears: x1, y1, x2, y2 as fractions of the frame
CYCLE_COOLDOWN_SECONDS = 15          # pause between cycles so the gripper servo can cool down
SIMULATED_STEP_SECONDS = 2           # each arm step takes this long in simulation

# ---- manual joint jogging --------------------------------------------------
# The dashboard's arm drawing uses its own angle scale; the servos use 0-180.
# servo_angle = servo_home + sign * (dashboard_angle - dashboard_home)
# If a joint moves the WRONG WAY on the real arm, change its sign from +1 to -1.
#                 channel, servo_home, dashboard_home, sign
JOINT_MAP = {
    "base":       (0,  85,   0, +1),
    "shoulder":   (1,  66,  62, +1),
    "elbow":      (2,  90,  98, +1),
    "wristPitch": (3,  90, -22, +1),
    "wristRoll":  (4, 108,   0, +1),
}
GRIPPER_CH = 5
GRIP_OPEN, GRIP_CLOSE = 180, 100      # same calibrated values as the sketch
GRIPPER_DASH_MAX = 90                 # dashboard gripper slider goes 0-90


def joint_to_servo(joint, dash_angle):
    """Dashboard joint id + angle -> (servo channel, servo angle 0-180), or (None, None)."""
    if joint == "gripper":
        a = GRIP_CLOSE + (dash_angle / float(GRIPPER_DASH_MAX)) * (GRIP_OPEN - GRIP_CLOSE)
        return GRIPPER_CH, int(round(max(0, min(180, a))))
    if joint in JOINT_MAP:
        ch, servo_home, dash_home, sign = JOINT_MAP[joint]
        a = servo_home + sign * (dash_angle - dash_home)
        return ch, int(round(max(0, min(180, a))))
    return None, None


# =====================================================================
# CAMERA - one background thread owns the webcam, everyone else just
# asks for "the latest frame" (so the live video and the classifier
# never fight over the device)
# =====================================================================
class Camera:
    def __init__(self, index):
        use_dshow = os.name == "nt"   # DirectShow opens webcams much faster on Windows
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if use_dshow else cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.opened = bool(self.cap.isOpened())
        if not self.opened:
            print("WARNING: could not open the webcam (index %s)." % index)
        self.frame = None
        self.fps = 0.0
        self._lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        count, t0 = 0, time.time()
        while True:
            if not self.opened:
                time.sleep(0.5)
                continue
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self.frame = frame
            count += 1
            now = time.time()
            if now - t0 >= 1.0:
                self.fps = count / (now - t0)
                count, t0 = 0, now

    def read(self):
        with self._lock:
            return None if self.frame is None else self.frame.copy()


# =====================================================================
# PRESENCE CHECK - "is there an object on the pick spot?"
# Classic background subtraction: keep one photo of the EMPTY spot and
# measure how much the live frame differs from it.
# =====================================================================
class Presence:
    def __init__(self):
        self.ref = None

    @staticmethod
    def _prep(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (21, 21), 0)

    def set_reference(self, frame):
        self.ref = self._prep(frame)

    def check(self, frame):
        """Returns (present, bbox_in_percent_or_None)."""
        if frame is None or self.ref is None:
            return False, None
        gray = self._prep(frame)
        if gray.shape != self.ref.shape:
            return False, None
        diff = cv2.absdiff(self.ref, gray)
        mask = cv2.threshold(diff, CHANGE_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, None
        biggest = max(contours, key=cv2.contourArea)
        h, w = gray.shape
        fraction = cv2.contourArea(biggest) / float(w * h)
        if fraction < MIN_OBJECT_AREA or fraction > MAX_OBJECT_AREA:
            return False, None
        x, y, bw, bh = cv2.boundingRect(biggest)
        return True, {"x": 100.0 * x / w, "y": 100.0 * y / h,
                      "w": 100.0 * bw / w, "h": 100.0 * bh / h}


def crop_to_bbox(frame, bbox, margin=0.25):
    """Square crop around the object, so it fills the image like the training photos did."""
    h, w = frame.shape[:2]
    cx = (bbox["x"] + bbox["w"] / 2) / 100.0 * w
    cy = (bbox["y"] + bbox["h"] / 2) / 100.0 * h
    side = max(bbox["w"] / 100.0 * w, bbox["h"] / 100.0 * h) * (1 + margin)
    x0, x1 = int(max(0, cx - side / 2)), int(min(w, cx + side / 2))
    y0, y1 = int(max(0, cy - side / 2)), int(min(h, cy + side / 2))
    crop = frame[y0:y1, x0:x1]
    return crop if crop.size else frame


# =====================================================================
# CLASSIFIER
# =====================================================================
def load_classifier():
    if not os.path.exists(MODEL_FILE):
        raise SystemExit(
            "\nModel file not found:\n  %s\n"
            "Copy waste_classifier.h5 into the 'ml' folder and start again.\n" % MODEL_FILE)
    print("Loading classifier (first start takes a few seconds)...")
    return keras.models.load_model(MODEL_FILE)


model_lock = threading.Lock()


def classify(image_bgr):
    """Returns (category, confidence_percent, latency_ms)."""
    img = cv2.cvtColor(cv2.resize(image_bgr, IMG_SIZE), cv2.COLOR_BGR2RGB)
    batch = np.expand_dims(img.astype("float32"), axis=0)   # rescaling is built into the model
    t0 = time.time()
    with model_lock:
        preds = model.predict(batch, verbose=0)[0]
    latency = (time.time() - t0) * 1000.0
    idx = int(np.argmax(preds))
    return CATEGORIES[idx], float(preds[idx]) * 100.0, latency


def predict_probs(image_bgr):
    """Raw softmax vector (one probability per category) for one image."""
    img = cv2.cvtColor(cv2.resize(image_bgr, IMG_SIZE), cv2.COLOR_BGR2RGB)
    batch = np.expand_dims(img.astype("float32"), axis=0)
    with model_lock:
        return model.predict(batch, verbose=0)[0]


def crop_fraction(frame, roi):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi
    crop = frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
    return crop if crop.size else frame


# =====================================================================
# ARM LINK - talks to the ESP32 over WiFi (HTTP). Same job as the old
# serial link: send one word, wait until the arm says it is done.
# =====================================================================
class ArmLink:
    def __init__(self):
        self.simulated = SIMULATE_HARDWARE
        self.error = None
        self.last_reply = "-"
        self.online = False
        self._lock = threading.Lock()          # only one command at a time
        if not self.simulated:
            if self.ping():
                self.online = True
                self.sync_gripper()
            else:
                self.error = "no answer from %s" % ARM_URL
                self.simulated = True
                print("WARNING: the arm did not answer at %s. Falling back to simulation." % ARM_URL)
                print("         Are you on the ARM-Control WiFi (or is ARM_IP right)?")

    @staticmethod
    def _get(path, timeout):
        """Returns (http_status_or_None, reply_text)."""
        quiet = path.startswith("/state") or path.startswith("/angles")
        try:
            r = requests.get(ARM_URL + path, timeout=timeout)
            if not quiet:
                print("[arm] %s -> %s %s" % (path, r.status_code, r.text.strip()[:60]))
            return r.status_code, r.text.strip()
        except requests.exceptions.RequestException as exc:
            if not quiet:
                print("[arm] %s -> %s" % (path, type(exc).__name__))
            return None, type(exc).__name__

    def sync_gripper(self):
        """Use the gripper open/close values saved on the ESP32 (the ones set on the arm's own page)."""
        global GRIP_OPEN, GRIP_CLOSE
        code, text = self._get("/state", 3)
        if code != 200:
            return
        try:
            data = json.loads(text)
            GRIP_OPEN, GRIP_CLOSE = int(data["gopen"]), int(data["gclose"])
            print("Gripper values read from the arm: open=%d close=%d" % (GRIP_OPEN, GRIP_CLOSE))
        except (ValueError, KeyError, TypeError):
            print("NOTE: could not read gripper values from the arm; using %d / %d." % (GRIP_OPEN, GRIP_CLOSE))

    def ping(self):
        # NOTE: the ESP32 cannot answer while it is mid-move, so only ping when idle.
        code, _ = self._get("/state", 3)
        return code == 200

    def send(self, command, timeout=ARM_MOVE_TIMEOUT):
        """command: PLASTIC / METAL / PAPER / HOME / GRIPPER_OPEN / GRIPPER_CLOSE.
        Returns "DONE", "ERR" or "TIMEOUT" like the old serial link."""
        if self.simulated:
            time.sleep(SIMULATED_MOVE_SECONDS)
            self.last_reply = "DONE (simulated)"
            return "DONE"

        if command in BIN_OF_WORD:
            path = "/sort?bin=" + BIN_OF_WORD[command]      # pick from the pick spot, drop in that bin
        elif command == "HOME":
            path = "/pose/go?k=home"
        elif command == "GRIPPER_OPEN":
            path = "/servo?ch=%d&angle=%d" % (GRIPPER_CH, GRIP_OPEN)
        elif command == "GRIPPER_CLOSE":
            path = "/servo?ch=%d&angle=%d" % (GRIPPER_CH, GRIP_CLOSE)
        else:
            return "ERR"

        with self._lock:
            code, text = self._get(path, timeout)
        if code == 200:
            self.online = True
            self.last_reply = "DONE"
            return "DONE"
        if code is None:
            self.online = False
            self.last_reply = "TIMEOUT (%s)" % text
            return "TIMEOUT"
        self.last_reply = "ERR %s %s" % (code, text)      # e.g. 409 BUSY
        return "ERR"

    def call(self, path, timeout=ARM_MOVE_TIMEOUT):
        """Run ONE ESP32 route (grab, lift, show, drop...). Returns (status_code_or_None, text)."""
        with self._lock:
            code, text = self._get(path, timeout)
        if code == 200:
            self.online = True
            self.last_reply = "DONE " + path
        elif code is None:
            self.online = False
            self.last_reply = "TIMEOUT (%s)" % text
        else:
            self.last_reply = "ERR %s %s" % (code, text)
        return code, text

    def jog(self, channel, angle):
        """Move ONE servo (manual sliders). Returns "DONE", "BUSY" or "ERR"."""
        if self.simulated:
            return "DONE"
        if not self._lock.acquire(False):
            return "BUSY"
        try:
            code, text = self._get("/servo?ch=%d&angle=%d" % (channel, angle), 15)
        finally:
            self._lock.release()
        if code == 200:
            return "DONE"
        return "BUSY" if code == 409 else "ERR"


# =====================================================================
# APP + SHARED STATE
# =====================================================================
app = Flask(__name__, static_folder=None)
CORS(app)

STARTED = time.time()
stats = {"requests": 0, "detections": 0}
state = {
    "armed": True,        # ready to report a NEW item (re-armed whenever the spot is seen empty)
    "busy": False,        # True while the arm is carrying out a command
    "estop": False,
    "mode": "AUTOMATIC",
    "frame_no": 1000,
    "last_event": None,
    "cooldown_until": 0.0,
    "settings": {},
}
events = collections.deque(maxlen=500)   # in-memory log (SQLite would replace this later)

camera = Camera(CAMERA_INDEX)
presence = Presence()
model = load_classifier()
arm = ArmLink()


def _take_first_reference():
    time.sleep(2.5)                      # let the webcam's auto-exposure settle
    frame = camera.read()
    if frame is not None:
        presence.set_reference(frame)
        print("Empty-spot reference captured. (Use 'Set empty background' in the dashboard to redo it.)")


threading.Thread(target=_take_first_reference, daemon=True).start()


@app.before_request
def _count_requests():
    if request.path.startswith("/api/") and request.path != "/api/camera/stream":
        stats["requests"] += 1


# ---------------------------------------------------------------- pages
@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<path:name>")
def frontend_files(name):
    return send_from_directory(FRONTEND_DIR, name)


# ---------------------------------------------------------------- camera
def _mjpeg():
    while True:
        frame = camera.read()
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(0.07)


@app.get("/api/camera/stream")
def camera_stream():
    return Response(_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/camera/reset_background")
def reset_background():
    frame = camera.read()
    if frame is None:
        return jsonify({"ok": False, "error": "camera not ready"})
    presence.set_reference(frame)
    state["armed"] = True
    return jsonify({"ok": True})


# ---------------------------------------------------------------- detection
@app.get("/api/detection/latest")
def latest_detection():
    """
    The dashboard polls this. It answers {"detected": false} unless a NEW item
    has been placed on the pick spot, held still, and classified.
    ?force=1 classifies whatever the camera sees right now (used by the manual button).
    """
    force = request.args.get("force") == "1"
    frame = camera.read()
    if frame is None:
        return jsonify({"detected": False, "message": "camera not ready"})

    present, bbox = presence.check(frame)

    if not force:
        if state["busy"] or state["estop"]:
            return jsonify({"detected": False, "message": "arm busy"})
        if not present:
            state["armed"] = True
            return jsonify({"detected": False, "message": "pick spot empty"})
        if not state["armed"]:
            return jsonify({"detected": False, "message": "waiting for the item to be removed"})

        # Make sure the object has stopped moving (a hand placing it would still be moving)
        time.sleep(SETTLE_SECONDS)
        frame2 = camera.read()
        present2, bbox2 = presence.check(frame2)
        if not present2:
            return jsonify({"detected": False, "message": "settling"})
        shift = abs((bbox["x"] + bbox["w"] / 2) - (bbox2["x"] + bbox2["w"] / 2)) + \
                abs((bbox["y"] + bbox["h"] / 2) - (bbox2["y"] + bbox2["h"] / 2))
        if shift > 6:
            return jsonify({"detected": False, "message": "object still moving"})
        frame, bbox = frame2, bbox2
        state["armed"] = False

    image = crop_to_bbox(frame, bbox) if bbox else frame
    category_raw, confidence, latency = classify(image)
    category = category_raw.capitalize()

    state["frame_no"] += 1
    stats["detections"] += 1
    item = {
        "detected": True,
        "object": "%s item" % category,
        "category": category,
        "confidence": round(confidence, 1),
        "bin": BIN_NAMES[category],
        "latency": int(round(latency)),
        "frame": "F%05d" % state["frame_no"],
        "bbox": bbox or {"x": 25.0, "y": 25.0, "w": 50.0, "h": 50.0},
        "timestamp": datetime.now().isoformat(),
    }
    event = {"ts": datetime.now(), "category": category,
             "confidence": item["confidence"], "status": "Detected"}
    events.append(event)
    state["last_event"] = event
    return jsonify(item)


# ---------------------------------------------------------------- full sorting cycle
# One button runs: GRAB -> LIFT -> SHOW CAM -> classify -> DROP into the matching bucket.
class CycleAbort(Exception):
    pass


cycle = {"running": False, "stage": "IDLE", "message": "Ready", "detection": None}


def _cycle_set(stage, message):
    cycle["stage"] = stage
    cycle["message"] = message
    print("[cycle] %s - %s" % (stage, message))


def _arm_step(path, label):
    """One step on the real arm (or a pause in simulation). Raises CycleAbort on any problem."""
    if state["estop"]:
        raise CycleAbort("Emergency stop engaged - cycle stopped")
    if arm.simulated:
        time.sleep(SIMULATED_STEP_SECONDS)
        return
    code, text = arm.call(path)
    if code != 200:
        raise CycleAbort("%s failed (%s %s)" % (label, code, text))


last_view = {"jpeg": None}      # the crop the classifier last looked at (for the dashboard preview)


def classify_held():
    """Average the model output over a few frames of the item held at the camera.
    Returns (category, confidence_percent, latency_ms, probabilities) or None if there is no frame."""
    acc = np.zeros(len(CATEGORIES), dtype="float32")
    used = 0
    crop = None
    t0 = time.time()
    for _ in range(SHOW_VOTE_FRAMES):
        frame = camera.read()
        if frame is not None:
            crop = crop_fraction(frame, SHOW_ROI)
            acc += predict_probs(crop)
            used += 1
        time.sleep(SHOW_VOTE_GAP)
    if used == 0:
        return None
    if crop is not None:
        ok, buf = cv2.imencode(".jpg", crop)
        if ok:
            last_view["jpeg"] = buf.tobytes()
    acc /= used
    idx = int(np.argmax(acc))
    return CATEGORIES[idx], float(acc[idx]) * 100.0, (time.time() - t0) * 1000.0 / used, acc


def _make_item(category_raw, confidence, latency, probs):
    """The detection dict the dashboard displays (includes all three probabilities)."""
    category = category_raw.capitalize()
    state["frame_no"] += 1
    x1, y1, x2, y2 = SHOW_ROI
    return {
        "detected": True,
        "object": "%s item" % category,
        "category": category,
        "confidence": round(confidence, 1),
        "bin": BIN_NAMES[category],
        "latency": int(round(latency)),
        "frame": "F%05d" % state["frame_no"],
        "bbox": {"x": 100.0 * x1, "y": 100.0 * y1, "w": 100.0 * (x2 - x1), "h": 100.0 * (y2 - y1)},
        "timestamp": datetime.now().isoformat(),
        "probs": {c: round(float(p) * 100.0, 1) for c, p in zip(CATEGORIES, probs)},
    }


def _run_cycle():
    event = None
    try:
        _cycle_set("GRABBING", "Grabbing the item...")
        _arm_step("/grab", "Grab")
        _cycle_set("LIFTING", "Lifting the item...")
        _arm_step("/lift", "Lift")
        _cycle_set("SHOWING", "Showing the item to the camera...")
        _arm_step("/show", "Show cam")
        time.sleep(SHOW_SETTLE_SECONDS)

        _cycle_set("CLASSIFYING", "Classifying...")
        result = classify_held()
        if result is None:
            raise CycleAbort("No camera frame - cannot classify")
        category_raw, confidence, latency, probs = result
        stats["detections"] += 1
        item = _make_item(category_raw, confidence, latency, probs)
        category = item["category"]
        print("[cycle] probabilities: %s" % item["probs"])
        cycle["detection"] = item
        event = {"ts": datetime.now(), "category": category,
                 "confidence": item["confidence"], "status": "Detected"}
        events.append(event)
        state["last_event"] = event

        if confidence < CYCLE_MIN_CONFIDENCE:
            event["status"] = "Failed"
            _cycle_set("NEEDS_MANUAL",
                       "Low confidence (%.1f%%) - the item is held at the camera. Choose a bucket or release it."
                       % confidence)
            return

        _cycle_set("DROPPING", "%s (%.1f%%) -> dropping into the %s bucket..."
                   % (category, confidence, category.lower()))
        _arm_step("/drop?bin=" + category.lower(), "Drop")
        event["status"] = "Sorted"
        state["cooldown_until"] = time.time() + CYCLE_COOLDOWN_SECONDS
        _cycle_set("DONE", "%s sorted into the %s bucket." % (category, category.lower()))
    except CycleAbort as exc:
        if event and event["status"] == "Detected":
            event["status"] = "Failed"
        _cycle_set("ERROR", str(exc))
    except Exception as exc:                      # never leave the cycle stuck "running"
        if event and event["status"] == "Detected":
            event["status"] = "Failed"
        _cycle_set("ERROR", "%s: %s" % (type(exc).__name__, exc))
    finally:
        cycle["running"] = False
        state["busy"] = False


def _run_manual_drop(which):
    """After NEEDS_MANUAL: drop into the chosen bucket, or just release the item."""
    event = state.get("last_event")
    try:
        if which == "release":
            cycle["detection"] = None
            _cycle_set("DROPPING", "Releasing the item...")
            _arm_step("/release", "Release")
            _cycle_set("DONE", "Item released.")
        else:
            _cycle_set("DROPPING", "Dropping into the %s bucket..." % which)
            _arm_step("/drop?bin=" + which, "Drop")
            if event and event["status"] in ("Detected", "Failed"):
                event["status"] = "Sorted"
            state["cooldown_until"] = time.time() + CYCLE_COOLDOWN_SECONDS
            _cycle_set("DONE", "Dropped into the %s bucket." % which)
    except CycleAbort as exc:
        _cycle_set("ERROR", str(exc))
    finally:
        cycle["running"] = False
        state["busy"] = False


@app.post("/api/robot/cycle")
def cycle_start():
    if state["estop"]:
        return jsonify({"ok": False, "error": "emergency stop is engaged"})
    if state["busy"] or cycle["running"]:
        return jsonify({"ok": False, "error": "arm is busy"})
    wait = state["cooldown_until"] - time.time()
    if wait > 0:
        return jsonify({"ok": False, "error": "cooling down - wait %d s" % int(wait + 1)})
    state["busy"] = True
    cycle["running"] = True
    cycle["detection"] = None
    threading.Thread(target=_run_cycle, daemon=True).start()
    return jsonify({"ok": True, "started": True})


@app.get("/api/robot/cycle")
def cycle_status():
    out = dict(cycle)
    out["cooldown"] = max(0, int(state["cooldown_until"] - time.time() + 0.99))
    return jsonify(out)


STEP_PATHS = {
    "grab":         ("/grab",              "GRABBING", "Grabbing the item..."),
    "lift":         ("/lift",              "LIFTING",  "Lifting the item..."),
    "show":         ("/show",              "SHOWING",  "Moving to the camera..."),
    "pick":         ("/pick",              "GRABBING", "Picking up (grab + lift)..."),
    "release":      ("/release",           "DROPPING", "Releasing the item..."),
    "home":         ("/pose/go?k=home",    "MOVING",   "Going home..."),
    "drop_paper":   ("/drop?bin=paper",    "DROPPING", "Dropping into the paper bucket..."),
    "drop_metal":   ("/drop?bin=metal",    "DROPPING", "Dropping into the metal bucket..."),
    "drop_plastic": ("/drop?bin=plastic",  "DROPPING", "Dropping into the plastic bucket..."),
}


def _run_single_step(key):
    path, stage, message = STEP_PATHS[key]
    try:
        _cycle_set(stage, message)
        _arm_step(path, key)
        _cycle_set("DONE", "%s done." % key)
    except CycleAbort as exc:
        _cycle_set("ERROR", str(exc))
    finally:
        cycle["running"] = False
        state["busy"] = False


@app.post("/api/robot/step")
def robot_step():
    """One manual arm step: grab, lift, show, pick, release, home, drop_paper/metal/plastic."""
    key = str((request.get_json(silent=True) or {}).get("step", "")).lower()
    if key not in STEP_PATHS:
        return jsonify({"ok": False, "error": "unknown step"}), 400
    if state["estop"]:
        return jsonify({"ok": False, "error": "emergency stop is engaged"})
    if state["busy"] or cycle["running"]:
        return jsonify({"ok": False, "error": "arm is busy"})
    state["busy"] = True
    cycle["running"] = True
    cycle["detection"] = None
    threading.Thread(target=_run_single_step, args=(key,), daemon=True).start()
    return jsonify({"ok": True, "started": True})


@app.post("/api/robot/classify")
def classify_now():
    """Classify what the camera sees right now, with the SAME crop the cycle uses. The arm does not move."""
    result = classify_held()
    if result is None:
        return jsonify({"detected": False, "message": "camera not ready"})
    stats["detections"] += 1
    return jsonify(_make_item(*result))


@app.get("/api/camera/last_crop")
def last_crop():
    if last_view["jpeg"] is None:
        return Response(status=404)
    return Response(last_view["jpeg"], mimetype="image/jpeg")


@app.post("/api/robot/drop")
def manual_drop():
    which = str((request.get_json(silent=True) or {}).get("bin", "")).lower()
    if which not in ("paper", "metal", "plastic"):
        return jsonify({"ok": False, "error": "bad bin"}), 400
    if state["estop"]:
        return jsonify({"ok": False, "error": "emergency stop is engaged"})
    if state["busy"] or cycle["running"]:
        return jsonify({"ok": False, "error": "arm is busy"})
    state["busy"] = True
    cycle["running"] = True
    threading.Thread(target=_run_manual_drop, args=(which,), daemon=True).start()
    return jsonify({"ok": True, "started": True})


@app.post("/api/robot/release")
def manual_release():
    if state["estop"]:
        return jsonify({"ok": False, "error": "emergency stop is engaged"})
    if state["busy"] or cycle["running"]:
        return jsonify({"ok": False, "error": "arm is busy"})
    state["busy"] = True
    cycle["running"] = True
    threading.Thread(target=_run_manual_drop, args=("release",), daemon=True).start()
    return jsonify({"ok": True, "started": True})


# ---------------------------------------------------------------- robot
def _run_arm_command(cmd):
    event = state.get("last_event")
    try:
        reply = arm.send(cmd)
    finally:
        state["busy"] = False
    if event and event["status"] == "Detected" and event["category"].upper() == cmd:
        event["status"] = "Sorted" if reply == "DONE" else "Failed"


@app.post("/api/robot/command")
def robot_command():
    cmd = str((request.get_json(silent=True) or {}).get("command", "")).strip()
    word = cmd.upper()
    if word not in ARM_COMMANDS:
        # pick / drop / stop are dashboard-only: the real arm's "pick" always continues to the
        # camera pose and "drop" needs a bin, so use the automatic cycle (PLASTIC / METAL / PAPER).
        return jsonify({"ok": True, "forwarded": False, "command": cmd})
    if state["estop"]:
        return jsonify({"ok": False, "forwarded": False, "error": "emergency stop is engaged"})
    if state["busy"]:
        return jsonify({"ok": False, "forwarded": False, "error": "arm is busy"})
    state["busy"] = True
    threading.Thread(target=_run_arm_command, args=(word,), daemon=True).start()
    return jsonify({"ok": True, "forwarded": True, "command": word})


@app.post("/api/robot/joint")
def robot_joint():
    """Manual jogging: one slider -> one servo on the real arm."""
    data = request.get_json(silent=True) or {}
    joint = data.get("joint")
    try:
        angle = float(data.get("angle"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "forwarded": False, "error": "bad angle"}), 400

    if state["estop"]:
        return jsonify({"ok": False, "forwarded": False, "error": "emergency stop is engaged"})
    if state["mode"] != "MANUAL":
        return jsonify({"ok": True, "forwarded": False, "reason": "not in manual mode"})
    if state["busy"]:
        return jsonify({"ok": False, "forwarded": False, "error": "arm is busy"})

    channel, servo_angle = joint_to_servo(joint, angle)
    if channel is None:
        return jsonify({"ok": False, "forwarded": False, "error": "unknown joint"}), 400
    reply = arm.jog(channel, servo_angle)
    return jsonify({"ok": reply == "DONE", "forwarded": True, "joint": joint,
                    "channel": channel, "servo_angle": servo_angle, "reply": reply})


@app.get("/api/robot/joints")
def robot_joints():
    return jsonify({})


@app.post("/api/robot/mode")
def robot_mode():
    mode = (request.get_json(silent=True) or {}).get("mode", "AUTOMATIC")
    state["mode"] = mode
    return jsonify({"ok": True, "mode": mode})


@app.post("/api/robot/estop")
def robot_estop():
    # NOTE: this blocks NEW commands. It cannot interrupt a move the arm is already
    # making (the sketch's moves are blocking) - for a real emergency use the power switch.
    engaged = bool((request.get_json(silent=True) or {}).get("engaged", True))
    state["estop"] = engaged
    return jsonify({"ok": True, "estop": engaged})


@app.post("/api/settings")
def save_settings():
    state["settings"] = request.get_json(silent=True) or {}
    return jsonify({"ok": True})


# ---------------------------------------------------------------- status / data
def _uptime():
    s = int(time.time() - STARTED)
    return "%02d:%02d:%02d" % (s // 3600, (s // 60) % 60, s % 60)


def _subsystems():
    frame = camera.read()
    h, w = (frame.shape[:2] if frame is not None else (0, 0))
    cam_ok = camera.opened and frame is not None
    if arm.error:
        arm_state, arm_link = "bad", "Simulated (arm not reachable)"
    elif arm.simulated:
        arm_state, arm_link = "warn", "Simulated (no hardware)"
    else:
        if not state["busy"]:               # the ESP32 cannot answer while it is moving
            arm.online = arm.ping()
        arm_state = "ok" if arm.online else "bad"
        arm_link = "WiFi (HTTP)" if arm.online else "WiFi - no answer"
    return [
        {"name": "Camera", "state": "ok" if cam_ok else "bad", "detail": "Laptop webcam, live stream + presence check",
         "kv": [["Device", "index %d" % CAMERA_INDEX], ["Resolution", "%d x %d" % (w, h)],
                ["Frame rate", "%.0f fps" % camera.fps],
                ["Empty-spot reference", "set" if presence.ref is not None else "not set"]]},
        {"name": "ML model", "state": "ok", "detail": "MobileNetV2 transfer-learning classifier",
         "kv": [["Model file", os.path.basename(MODEL_FILE)], ["Input size", "%d x %d" % IMG_SIZE],
                ["Classes", ", ".join(CATEGORIES)], ["Validation accuracy", "%.1f %%" % VAL_ACCURACY]]},
        {"name": "Arm controller", "state": arm_state, "detail": "ESP32 + PCA9685 running the arm firmware",
         "kv": [["Link", arm_link], ["Address", ARM_IP],
                ["Last reply", arm.last_reply], ["Busy", "yes" if state["busy"] else "no"]]},
        {"name": "Flask backend", "state": "ok", "detail": "REST API, camera stream, arm bridge",
         "kv": [["Uptime", _uptime()], ["API requests", str(stats["requests"])],
                ["Detections", str(stats["detections"])], ["Mode", state["mode"]]]},
    ]


@app.get("/api/status")
def status():
    return jsonify({"connected": True, "mode": state["mode"], "busy": state["busy"],
                    "cycle": cycle["stage"], "subsystems": _subsystems()})


@app.get("/api/history")
def history():
    rows = [{"timestamp": e["ts"].isoformat(), "category": e["category"],
             "confidence": e["confidence"], "status": e["status"]} for e in reversed(events)]
    return jsonify(rows)


@app.get("/api/analytics")
def analytics():
    counts = {"Plastic": 0, "Paper": 0, "Metal": 0}
    outcome = {k: {"ok": 0, "fail": 0} for k in counts}
    ok = fail = 0
    for e in events:
        counts[e["category"]] += 1
        if e["status"] == "Sorted":
            ok += 1
            outcome[e["category"]]["ok"] += 1
        elif e["status"] == "Failed":
            fail += 1
            outcome[e["category"]]["fail"] += 1
    now = datetime.now()
    hourly = []
    for back in range(10, -1, -1):
        hour = (now.hour - back) % 24
        n = sum(1 for e in events if e["ts"].date() == now.date() and e["ts"].hour == hour)
        hourly.append({"label": "%02d" % hour, "value": n})
    return jsonify({
        "counts": counts, "ok": ok, "fail": fail, "total": len(events),
        "accuracy": VAL_ACCURACY,
        "perClass": [],     # fill in later from an evaluation script (confusion matrix)
        "outcome": [{"label": k, "ok": v["ok"], "fail": v["fail"]}
                    for k, v in outcome.items() if v["ok"] + v["fail"] > 0],
        "hourly": hourly,
    })


# =====================================================================
if __name__ == "__main__":
    print("\nSegreBot is starting. Open  http://127.0.0.1:5000  in your browser.")
    print("Hardware: %s\n" % ("SIMULATED" if arm.simulated else "arm at " + ARM_URL))
    # debug/reloader must stay off - the reloader would start the webcam and the model twice.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)