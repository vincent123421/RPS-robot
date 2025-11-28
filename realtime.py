import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import time
import random

# ---------- 参数 ----------
MODEL_PATH = "rps_mlp.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN = 128
COUNTDOWN_TIME = 3   # 秒
# --------------------------

# ---------- 定义模型 ----------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x):
        return self.net(x)

# ---------- 加载模型 ----------
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
classes = list(ckpt["classes"])
model = MLP(63, HIDDEN, len(classes)).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("✅ 已加载模型，类别:", classes)

# ---------- Mediapipe ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ---------- 判定逻辑 ----------
def ai_counter_move(player_move: str):
    """AI 出能赢玩家的手势"""
    if player_move == "rock":
        return "paper"
    elif player_move == "paper":
        return "scissors"
    elif player_move == "scissors":
        return "rock"
    else:
        return random.choice(classes)

def get_winner(player, ai):
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
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    x = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        pred = torch.softmax(out, dim=1)
        idx = pred.argmax(1).item()
        conf = pred[0, idx].item()
    return classes[idx], conf

# ---------- 主程序 ----------
cap = cv2.VideoCapture(0)
prev_time = 0

score = {"You": 0, "AI": 0, "Draw": 0}
player_move = "None"
ai_move = "None"
winner = "None"

last_round_time = time.time()
state = "COUNTDOWN"  # ["COUNTDOWN", "SHOW"]

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
        cv2.putText(frame, f"Get ready in {remaining if remaining>0 else 0}s",
                    (180, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 4)
        if elapsed >= COUNTDOWN_TIME:
            # 到时间，预测一帧
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
        # 显示结果 2 秒
        cv2.putText(frame, f"You: {player_move}", (60, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 3)
        cv2.putText(frame, f"AI: {ai_move}", (60, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        color = (0,255,0) if winner=="You" else (0,0,255) if winner=="AI" else (200,200,200)
        cv2.putText(frame, f"Winner: {winner}", (150, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)
        if elapsed > 2:
            state = "COUNTDOWN"
            last_round_time = curr_time

    # 显示分数
    cv2.putText(frame, f"You {score['You']} - {score['AI']} AI",
                (150, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, f"Draws: {score['Draw']}",
                (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,180,180), 2)

    # FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

    cv2.imshow("Rock Paper Scissors AI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
print("👋 游戏结束！")
