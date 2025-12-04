import cv2
import mediapipe as mp
import numpy as np
import os

# 文件用途：数据采集入口，录制手势关键点数据
# 最后修改：2025-12-04
# 主要功能：
# - 交互采集不同手势的 21 点 (x,y,z) 坐标
# - 打包为 63 维向量保存到 .npy 文件
# 重要函数：record_gesture(label)
# 使用说明：采集完成后运行 train.py 训练生成 rps_mlp.pth。
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 三类动作（可扩展）
GESTURES = ["rock", "paper", "scissors"]

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
# 解释：如果不存在 data 文件夹就创建一个，用来保存采集结果。

all_data = []
all_labels = []

# 初始化MediaPipe
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def record_gesture(label):
    """录制指定标签的若干帧关键点并缓存到内存列表"""
    print(f"\n准备录制手势: {label}")
    print("按 's' 开始录制，'q' 退出该手势")

    cap = cv2.VideoCapture(0)
    # 解释：打开默认摄像头（编号 0）。
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
                # 解释：把检测到的手部关键点和连接线画到画面上，便于可视化。

        cv2.putText(frame, f"Gesture: {label} ({collected})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Collecting Data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            recording = True
            print("开始录制...")
        elif key == ord('q'):
            break
        # 解释：按 's' 开始采集，按 'q' 退出该手势的采集循环。

        if recording and result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            data = []
            for lm in landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])
            all_data.append(data)
            all_labels.append(label)
            collected += 1
            # 解释：把一帧的 63 个数字存入 all_data，并记录对应的手势标签。

    cap.release()
    cv2.destroyAllWindows()

for gesture in GESTURES:
    record_gesture(gesture)

# 保存数据：生成特征与标签的 numpy 文件，供训练脚本使用
np.save(os.path.join(DATA_DIR, "dataset.npy"), np.array(all_data))
np.save(os.path.join(DATA_DIR, "labels.npy"), np.array(all_labels))
# 解释：保存成 .npy（二进制的 numpy 格式），训练脚本能快速读取。
print(f"✅ 数据保存完成，共 {len(all_data)} 条样本")
