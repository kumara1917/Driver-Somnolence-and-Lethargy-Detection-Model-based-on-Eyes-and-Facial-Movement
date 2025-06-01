from ultralytics import YOLO
import cv2
import mediapipe as mp
from scipy.spatial import distance
from playsound import playsound
import threading
import time
import numpy as np
import math

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def sound_alarm():
    playsound("D:/beep-01a.mp3")

EAR_THRESHOLD = 0.25
CLOSED_EYE_DURATION = 3
HEAD_TILT_THRESHOLD = 45

model = YOLO("yolov8n-face.pt")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1
CHIN = 152

cv2.namedWindow("EyeF Model", cv2.WINDOW_NORMAL)
intro = np.zeros((400, 800, 3), dtype=np.uint8)
cv2.putText(intro, "EyeF Model - Drowsiness Detection Model", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
cv2.imshow("EyeF Model", intro)
cv2.waitKey(3000)

cap = cv2.VideoCapture(0)
alarm_on = False
eye_closed_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    results = model(frame, verbose=False)
    dets = results[0].boxes.xyxy.cpu().numpy()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_mp = face_mesh.process(rgb_frame)

    if results_mp.multi_face_landmarks:
        landmarks = results_mp.multi_face_landmarks[0]
        left_eye = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in LEFT_EYE]
        right_eye = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in RIGHT_EYE]
        nose = landmarks.landmark[NOSE_TIP]
        chin = landmarks.landmark[CHIN]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 0), 1)

        nx, ny = nose.x * w, nose.y * h
        cx, cy = chin.x * w, chin.y * h
        dx, dy = cx - nx, cy - ny
        angle = abs(math.degrees(math.atan2(dy, dx)))

        alert = False
        if ear < EAR_THRESHOLD:
            if eye_closed_time is None:
                eye_closed_time = time.time()
            elif time.time() - eye_closed_time >= CLOSED_EYE_DURATION:
                alert = True
        else:
            eye_closed_time = None

        if angle > HEAD_TILT_THRESHOLD:
            alert = True

        if alert:
            if not alarm_on:
                alarm_on = True
                threading.Thread(target=sound_alarm, daemon=True).start()
            cv2.putText(frame, "ALERT! DROWSY or HEAD TILT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            alarm_on = False

    for det in dets:
        x1, y1, x2, y2 = map(int, det)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("EyeF Model", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
