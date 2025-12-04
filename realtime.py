import cv2
import mediapipe as mp
import numpy as np
import torch
import time
import random
from common.models import load_gesture_mlp
from common.strategy import get_winning_move
from common.vision import init_hands, landmarks_to_vector, infer_label

# 文件用途：对战入口，仅识别与克制策略展示
# 最后修改：2025-12-04
# 主要功能：
# - 倒计时后采一帧手势
# - AI 输出克制手势并记分
# - 信息叠加显示与 FPS 统计
# 重要组件：predict、ai_counter_move、状态机循环
# 使用说明：只测策略运行本文件；仿真请运行 main.py。
# ---------- 参数 ----------
MODEL_PATH = "rps_mlp.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN = 128
COUNTDOWN_TIME = 3   # 秒
# --------------------------

# ---------- 加载模型（统一接口） ----------
model, classes = load_gesture_mlp(MODEL_PATH, DEVICE)
print("✅ 已加载模型，类别:", classes)

# ---------- Mediapipe ----------
hands, mp_drawing = init_hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)
# 解释：
# - static_image_mode=False：视频流模式（不是单张图片），适合实时场景。
# - min_tracking_confidence：跟踪阈值，越高越稳但可能漏检，越低更敏感但易误检。

# ---------- 判定逻辑 ----------
def ai_counter_move(player_move: str):
    """返回能克制玩家的手势；未知时随机一类以保持互动

    核心逻辑：
    - rock(石头) → paper(布)
    - paper(布) → scissors(剪刀)
    - scissors(剪刀) → rock(石头)
    """
    """AI 出能赢玩家的手势"""
    move = get_winning_move(player_move)
    return move if move != "waiting" else random.choice(classes)

def get_winner(player, ai):
    """根据双方手势判定胜负：相同为平局，其他按规则比对"""
    if player == ai:
        return "Draw"
    elif (player == "rock" and ai == "scissors") or \
         (player == "paper" and ai == "rock") or \
         (player == "scissors" and ai == "paper"):
        return "You"
    else:
        return "AI"

# ---------- 预测 ----------
def predict(hand_landmarks):
    coords = landmarks_to_vector(hand_landmarks)
    label, conf = infer_label(model, classes, coords, DEVICE, use_softmax=True)
    return label, conf

# ---------- 主程序 ----------
cap = cv2.VideoCapture(0)
prev_time = 0

score = {"You": 0, "AI": 0, "Draw": 0}
player_move = "None"
ai_move = "None"
winner = "None"

last_round_time = time.time()
state = "COUNTDOWN"  # ["COUNTDOWN", "SHOW"]
# 解释：程序初始在“倒计时”状态，计时到达后采一帧并切换到“展示”状态。

print("🎮 石头剪刀布实时对战开始！(按 Q 退出)")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    curr_time = time.time()

    # FPS
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    # 计时逻辑
    elapsed = curr_time - last_round_time

    if state == "COUNTDOWN":
        remaining = COUNTDOWN_TIME - int(elapsed)
        cv2.putText(
            frame,
            f"Get ready in {remaining if remaining>0 else 0}s",
            (180, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 255),
            4,
        )
        if elapsed >= COUNTDOWN_TIME:
            # 核心逻辑：到时间，预测一帧（只取当前帧，避免玩家变化期间“作弊”）
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    player_move, conf = predict(hand_landmarks)
                    ai_move = ai_counter_move(player_move)
                    winner = get_winner(player_move, ai_move)
                    score[winner] += 1
            else:
                player_move, ai_move, winner = "None", "None", "Draw"
                score[winner] += 1
            state = "SHOW"
            last_round_time = curr_time

    elif state == "SHOW":
        # 显示结果 2 秒（固定显示周期，之后再次进入 COUNTDOWN）
        cv2.putText(
            frame,
            f"You: {player_move}",
            (60, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 0),
            3,
        )
        cv2.putText(
            frame,
            f"AI: {ai_move}",
            (60, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )
        color = (0,255,0) if winner=="You" else (0,0,255) if winner=="AI" else (200,200,200)
        cv2.putText(
            frame,
            f"Winner: {winner}",
            (150, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            color,
            4,
        )
        if elapsed > 2:
            state = "COUNTDOWN"
            last_round_time = curr_time

    # 显示分数
    cv2.putText(
        frame,
        f"You {score['You']} - {score['AI']} AI",
        (150, 420),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Draws: {score['Draw']}",
        (10, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (180, 180, 180),
        2,
    )

    # FPS
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        2,
    )

    cv2.imshow("Rock Paper Scissors AI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
print("👋 游戏结束！")
