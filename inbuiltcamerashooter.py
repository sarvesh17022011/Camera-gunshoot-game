
import os
import time
import threading

import cv2
import numpy as np
import mediapipe as mp

from flask import Flask, jsonify, Response, send_from_directory
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =====================================================
# SETTINGS
# =====================================================

PORT = 5000
CAMERA_INDEX = 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

PROCESS_WIDTH = 480
PREVIEW_WIDTH = 640

SMOOTHING = 0.40

SENSITIVITY_X = 1.75
SENSITIVITY_Y = 1.30

GESTURE_STABILITY = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")


# =====================================================
# FLASK
# =====================================================

app = Flask(__name__)


# =====================================================
# LAPTOP CAMERA
# =====================================================

class LaptopCamera:

    def __init__(self):

        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.sequence = 0

        self.cap = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_DSHOW
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT
        )

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                "Could not open laptop camera. "
                "Check Windows camera permissions."
            )

        threading.Thread(
            target=self.read_camera,
            daemon=True
        ).start()

    def read_camera(self):

        while self.running:

            success, frame = self.cap.read()

            if not success:
                time.sleep(0.01)
                continue

            # Mirror camera
            frame = cv2.flip(frame, 1)

            with self.lock:

                self.frame = frame
                self.sequence += 1

    def get_frame(self):

        with self.lock:

            if self.frame is None:
                return None

            return self.frame.copy()

    def release(self):

        self.running = False
        self.cap.release()


camera = LaptopCamera()


# =====================================================
# MEDIAPIPE SETUP
# =====================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(

    base_options=base_options,

    running_mode=vision.RunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.6,

    min_hand_presence_confidence=0.6,

    min_tracking_confidence=0.6
)

detector = vision.HandLandmarker.create_from_options(
    options
)


# =====================================================
# SHARED DATA
# =====================================================

latest_data = {

    "hand_detected": False,

    "x": 0.5,
    "y": 0.5,

    "gesture": "NO HAND",

    "pinch": False
}

data_lock = threading.Lock()

latest_landmarks = None
landmarks_lock = threading.Lock()


# =====================================================
# GESTURE DETECTION
# =====================================================

def detect_gesture(landmarks):
    """
    Detect:
    - FIST
    - TWO FINGERS
    - OPEN PALM

    Uses finger tip vs PIP joint positions.
    """

    # Index finger
    index_up = (
        landmarks[8].y <
        landmarks[6].y
    )

    # Middle finger
    middle_up = (
        landmarks[12].y <
        landmarks[10].y
    )

    # Ring finger
    ring_up = (
        landmarks[16].y <
        landmarks[14].y
    )

    # Pinky finger
    pinky_up = (
        landmarks[20].y <
        landmarks[18].y
    )


    # Count raised fingers
    raised_fingers = sum([
        index_up,
        middle_up,
        ring_up,
        pinky_up
    ])


    # =====================================
    # FIST
    # =====================================

    if raised_fingers == 0:
        return "FIST"


    # =====================================
    # TWO FINGERS
    # Index + Middle only
    # =====================================

    if (
        index_up
        and middle_up
        and not ring_up
        and not pinky_up
    ):
        return "TWO FINGERS"


    # =====================================
    # OPEN PALM
    # =====================================

    if raised_fingers >= 3:
        return "OPEN PALM"


    # =====================================
    # PARTIAL / UNKNOWN
    # Treat as open palm so it doesn't
    # accidentally trigger shooting.
    # =====================================

    return "OPEN PALM"


# =====================================================
# HAND TRACKING
# =====================================================

def process_hand():

    global latest_landmarks

    last_sequence = -1

    timestamp = 0

    smooth_x = 0.5
    smooth_y = 0.5

    previous_candidate = None
    candidate_count = 0

    stable_gesture = "NO HAND"

    while True:

        # Get newest camera frame
        with camera.lock:

            if camera.frame is None:

                frame = None
                sequence = -1

            else:

                frame = camera.frame.copy()
                sequence = camera.sequence

        if frame is None or sequence == last_sequence:

            time.sleep(0.003)
            continue

        last_sequence = sequence

        # Resize for faster MediaPipe processing
        height, width = frame.shape[:2]

        new_height = int(
            height * PROCESS_WIDTH / width
        )

        small = cv2.resize(
            frame,
            (PROCESS_WIDTH, new_height),
            interpolation=cv2.INTER_AREA
        )

        rgb = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = max(
            timestamp + 1,
            int(time.monotonic() * 1000)
        )

        result = detector.detect_for_video(
            mp_image,
            timestamp
        )

        # No hand detected
        if not result.hand_landmarks:

            previous_candidate = None
            candidate_count = 0
            stable_gesture = "NO HAND"

            with landmarks_lock:

                latest_landmarks = None

            with data_lock:

                latest_data["hand_detected"] = False
                latest_data["gesture"] = "NO HAND"
                latest_data["pinch"] = False

            continue

        # Hand detected
        landmarks = result.hand_landmarks[0]

        # Save landmarks for camera overlay
        with landmarks_lock:

            latest_landmarks = [
                (point.x, point.y)
                for point in landmarks
            ]

        # Palm centre using wrist and palm joints
        palm_points = [

            landmarks[0],
            landmarks[5],
            landmarks[9],
            landmarks[13],
            landmarks[17]

        ]

        center_x = (
    sum(p.x for p in palm_points) / 5
)

        center_y = (
            sum(p.y for p in palm_points) / 5
        )

        # Increase cursor sensitivity
        raw_x = (
            0.5 + (center_x - 0.5) * SENSITIVITY_X
        )

        raw_y = (
            0.5 + (center_y - 0.5) * SENSITIVITY_Y
        )

        raw_x = max(
            0.0,
            min(1.0, raw_x)
        )

        raw_y = max(
            0.0,
            min(1.0, raw_y)
        )

        # Smooth movement
        smooth_x = (
            SMOOTHING * raw_x
            + (1 - SMOOTHING) * smooth_x
        )

        smooth_y = (
            SMOOTHING * raw_y
            + (1 - SMOOTHING) * smooth_y
        )

        # Detect gesture
        candidate = detect_gesture(landmarks)

        # Gesture stability
        if candidate == previous_candidate:

            candidate_count += 1

        else:

            previous_candidate = candidate
            candidate_count = 1

        if candidate_count >= GESTURE_STABILITY:

            stable_gesture = candidate

        # Publish data
        with data_lock:

            latest_data["hand_detected"] = True

            latest_data["x"] = smooth_x
            latest_data["y"] = smooth_y

            latest_data["gesture"] = stable_gesture

            # Existing HTML uses pinch as the action signal.
            # A fist now replaces the old pinch gesture.
            latest_data["pinch"] = (
                stable_gesture == "FIST"
            )


# Start tracking thread
threading.Thread(
    target=process_hand,
    daemon=True
).start()


# =====================================================
# HAND DATA API
# =====================================================

@app.route("/hand")
def hand():

    with data_lock:

        return jsonify(
            latest_data.copy()
        )


# =====================================================
# CAMERA PREVIEW WITH HAND SKELETON
# =====================================================

@app.route("/camera")
def camera_feed():

    def generate():

        connections = [

            (0, 1), (1, 2), (2, 3), (3, 4),

            (0, 5), (5, 6), (6, 7), (7, 8),

            (5, 9), (9, 10), (10, 11), (11, 12),

            (9, 13), (13, 14), (14, 15), (15, 16),

            (13, 17), (17, 18), (18, 19), (19, 20),

            (0, 17)

        ]

        while True:

            frame = camera.get_frame()

            if frame is None:

                time.sleep(0.01)
                continue

            # Resize preview
            height, width = frame.shape[:2]

            preview_height = int(
                height * PREVIEW_WIDTH / width
            )

            preview = cv2.resize(
                frame,
                (PREVIEW_WIDTH, preview_height),
                interpolation=cv2.INTER_LINEAR
            )

            # Get latest hand landmarks
            with landmarks_lock:

                points = latest_landmarks

            # Draw skeleton
            if points is not None:

                for start, end in connections:

                    x1 = int(
                        points[start][0] * PREVIEW_WIDTH
                    )

                    y1 = int(
                        points[start][1] * preview_height
                    )

                    x2 = int(
                        points[end][0] * PREVIEW_WIDTH
                    )

                    y2 = int(
                        points[end][1] * preview_height
                    )

                    cv2.line(
                        preview,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 100),
                        2
                    )

                # Draw joint dots
                for x, y in points:

                    px = int(
                        x * PREVIEW_WIDTH
                    )

                    py = int(
                        y * preview_height
                    )

                    cv2.circle(
                        preview,
                        (px, py),
                        4,
                        (0, 255, 255),
                        -1
                    )

            # Camera label
            cv2.putText(
                preview,
                "LAPTOP CAMERA",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            success, encoded = cv2.imencode(
                ".jpg",
                preview,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    65
                ]
            )

            if not success:
                continue

            yield (

                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"

                + encoded.tobytes()

                + b"\r\n"

            )

            time.sleep(0.033)

    return Response(

        generate(),

        mimetype=(
            "multipart/x-mixed-replace; boundary=frame"
        )

    )


# =====================================================
# MAIN PAGE
# =====================================================

@app.route("/")
def main_page():

    return send_from_directory(
        BASE_DIR,
        "mainpage.html"
    )


# =====================================================
# GAME PAGE
# =====================================================

@app.route("/game.html")
def game_page():

    return send_from_directory(
        BASE_DIR,
        "game.html"
    )


# =====================================================
# STATUS
# =====================================================

@app.route("/status")
def status():

    return jsonify({

        "camera": camera.get_frame() is not None,

        "server": True

    })


# =====================================================
# START SERVER
# =====================================================

if __name__ == "__main__":

    print("\n================================")
    print("       TARGET GESTURE GAME")
    print("================================")

    print("Camera: Laptop webcam")

    print("Main page: http://localhost:5000")

    print("Game: http://localhost:5000/game.html")

    print("Camera preview: http://localhost:5000/camera")

    print("================================\n")

    try:

        app.run(
            host="0.0.0.0",
            port=PORT,
            threaded=True,
            debug=False,
            use_reloader=False
        )

    finally:

        camera.release()
