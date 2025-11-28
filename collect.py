import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 三类动作（可扩展）
GESTURES = ["rock", "paper", "scissors"]

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

all_data = []
all_labels = []

# 初始化MediaPipe
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def record_gesture(label):
    print(f"\n准备录制手势: {label}")
    print("按 's' 开始录制，'q' 退出该手势")

    cap = cv2.VideoCapture(0)
    recording = False
    collected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Gesture: {label} ({collected})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Collecting Data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            recording = True
            print("开始录制...")
        elif key == ord('q'):
            break

        if recording and result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            data = []
            for lm in landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])
            all_data.append(data)
            all_labels.append(label)
            collected += 1

    cap.release()
    cv2.destroyAllWindows()

for gesture in GESTURES:
    record_gesture(gesture)

# 保存数据
np.save(os.path.join(DATA_DIR, "dataset.npy"), np.array(all_data))
np.save(os.path.join(DATA_DIR, "labels.npy"), np.array(all_labels))
print(f"✅ 数据保存完成，共 {len(all_data)} 条样本")
