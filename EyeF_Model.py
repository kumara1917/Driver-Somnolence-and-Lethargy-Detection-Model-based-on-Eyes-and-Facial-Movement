import cv2
import mediapipe as mp
from scipy.spatial import distance
from playsound import playsound
import threading
import time
import numpy as np

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def sound_alarm():
    playsound("D:/beep-01a.mp3")

def intro_screen():
    intro = np.zeros((400, 800, 3), dtype=np.uint8)
    
    cv2.putText(intro, "EyeF Model - Drowsiness Detection Model", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("EyeF Intro", intro)
    cv2.waitKey(3000)
    cv2.destroyWindow("EyeF Intro")

EAR_THRESHOLD = 0.25
CLOSED_EYE_DURATION = 2

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TOP = 1
CHIN = 152

intro_screen()
cap = cv2.VideoCapture(0)
alarm_on = False
eye_closed_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        left_eye = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in LEFT_EYE]
        right_eye = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in RIGHT_EYE]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        nose_y = landmarks.landmark[NOSE_TOP].y * h
        chin_y = landmarks.landmark[CHIN].y * h
        head_angle = np.abs(nose_y - chin_y)

        cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 0), 1)

        if ear < EAR_THRESHOLD or head_angle < 15:
            if eye_closed_time is None:
                eye_closed_time = time.time()
            elif time.time() - eye_closed_time >= CLOSED_EYE_DURATION:
                if not alarm_on:
                    alarm_on = True
                    threading.Thread(target=sound_alarm, daemon=True).start()
                cv2.putText(frame, "ALERT! DROWSINESS DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            eye_closed_time = None
            alarm_on = False

        cv2.putText(frame, "Running EyeF Model [YOLO + Perclos + EAR]", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    cv2.imshow("EyeF Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
