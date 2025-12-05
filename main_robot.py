import cv2
import mediapipe as mp
import torch
import numpy as np
from collections import deque
import time
import threading
import sys

sys.path.insert(0, "/home/jerry/pnd_sdk_python/")
# export PYTHONPATH=$PYTHONPATH:/home/jerry/pnd_sdk_python/

from pndbotics_sdk_py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from pndbotics_sdk_py.idl.default import adam_u_msg_dds__LowCmd_
from pndbotics_sdk_py.idl.default import adam_u_msg_dds__LowState_
from pndbotics_sdk_py.idl.adam_u.msg.dds_ import LowCmd_
from pndbotics_sdk_py.idl.adam_u.msg.dds_ import LowState_
from pndbotics_sdk_py.idl.adam_u.msg.dds_ import HandCmd_
from pndbotics_sdk_py.idl.default import adam_u_msg_dds__HandCmd_

from pst_gesture import (
    ADAM_U_NUM_MOTOR,
    KP_CONFIG,
    KD_CONFIG,
    HOME,
    EXE,
    PAPER,
    ROCK,
    SCISSORS,
)

# ==========================================
# 1. 极速配置
# ==========================================
MODEL_PATH = "rps_mlp.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ⚡ 降低图像分辨率以提高处理速度
CAM_WIDTH, CAM_HEIGHT = 640, 480


# ==========================================
# 2. 多线程摄像头类 (核心优化)
# ==========================================
class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        # 尝试设置 FPS 为 60 (取决于摄像头是否支持)
        self.capture.set(cv2.CAP_PROP_FPS, 60)

        self.status, self.frame = self.capture.read()
        self.stop_event = False
        self.lock = threading.Lock()

        # 启动后台线程
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stop_event:
            status, frame = self.capture.read()
            if status:
                with self.lock:
                    self.frame = frame
                    self.status = status
            else:
                self.stop_event = True

    def get_frame(self):
        with self.lock:
            return self.status, self.frame.copy()  # 返回副本以防竞争

    def release(self):
        self.stop_event = True
        self.thread.join()
        self.capture.release()


# ==========================================
# 3. 逻辑与模型
# ==========================================
class RPS_MLP(torch.nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def get_winning_move(user_move):
    mapping = {"rock": "paper", "paper": "scissors", "scissors": "rock"}
    return mapping.get(user_move, "waiting")


# ==========================================
# 4. 主程序
# ==========================================
def run_simulation():
    # Load Model
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    net = RPS_MLP(num_classes=len(ckpt["classes"]))
    net.load_state_dict(ckpt["model_state"])
    net.to(DEVICE)
    net.eval()
    classes = list(ckpt["classes"])

    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(1, sys.argv[1])

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    cmd = adam_u_msg_dds__LowCmd_()

    hand_pub = ChannelPublisher("rt/handcmd", HandCmd_)
    hand_pub.Init()
    hand_cmd = adam_u_msg_dds__HandCmd_()

    dt = 0.002
    runing_time = 0.0

    # Init MediaPipe (CPU Mode, but optimized)
    mp_hands = mp.solutions.hands
    # ⚠️ 降低置信度阈值：让它更早地检测到手，哪怕有点模糊
    hands = mp_hands.Hands(
        max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.3
    )

    # Init Threaded Camera
    print("🎥 启动多线程摄像头采集...")
    cam = ThreadedCamera(0)
    time.sleep(1.0)  # 等待摄像头预热

    while runing_time < 3.0:
        step_start = time.perf_counter()
        runing_time += dt

        phase = np.tanh(runing_time / 1.2)
        for i in range(ADAM_U_NUM_MOTOR):
            cmd.motor_cmd[i].q = phase * EXE[i] + (1 - phase) * HOME[i]
            # 使用配置的 Kp 和 Kd 值
            cmd.motor_cmd[i].kp = KP_CONFIG[i]
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = KD_CONFIG[i]
            cmd.motor_cmd[i].tau = 0.0
        for i in range(12):
            hand_cmd.position[i] = ROCK[i]

        pub.Write(cmd)
        hand_pub.Write(hand_cmd)

        time_until_next_step = dt - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    current_robot_move = "waiting"
    winning_move = "waiting"
    # FPS Calculation
    fps_start = time.time()
    frames = 0
    inference_time = 0

    last_move_time = time.time()

    print("🚀 极速模式已启动 - 请测试")

    while cam.status:
        loop_start = time.time()

        # 1. 获取最新帧 (非阻塞)
        ret, frame = cam.get_frame()
        if not ret:
            break

        # 2. 图像处理
        frame_flip = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2RGB)

        t0 = time.time()
        result = hands.process(rgb)
        inference_time = (time.time() - t0) * 1000  # ms

        user_move = "waiting"
        if result.multi_hand_landmarks:
            landmarks = []
            for lm in result.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # MLP Inference
            x_tensor = (
                torch.tensor(landmarks, dtype=torch.float32).to(DEVICE).unsqueeze(0)
            )
            with torch.no_grad():
                preds = net(x_tensor)
                # ⚠️ 激进策略：只要概率分布有一点倾向，马上行动，不要等 softmax 完全确定
                user_move = classes[torch.argmax(preds, dim=1).item()]

        # 3. 立即触发动作
        if time.time() - last_move_time > 0.25:
            last_move_time = time.time()
            winning_move = get_winning_move(user_move)
        if winning_move != current_robot_move:
            current_robot_move = winning_move
            if current_robot_move == "rock":
                target_gesture = ROCK
            elif current_robot_move == "paper":
                target_gesture = PAPER
            elif current_robot_move == "scissors":
                target_gesture = SCISSORS
            else:
                target_gesture = ROCK
            for i in range(12):
                hand_cmd.position[i] = target_gesture[i]
            pub.Write(cmd)
            hand_pub.Write(hand_cmd)
            print("msg sent:", hand_cmd.position)

        # 4. 信息显示
        frames += 1
        if time.time() - fps_start >= 1.0:
            fps = frames / (time.time() - fps_start)
            print(f"FPS: {fps:.1f} | Inference: {inference_time:.1f}ms")
            frames = 0
            fps_start = time.time()

        cv2.putText(
            frame_flip,
            f"User: {user_move}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame_flip,
            f"Robot: {current_robot_move}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame_flip,
            f"Inference: {inference_time:.1f}ms",
            (10, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 0),
            1,
        )

        cv2.imshow("Fast Sim", frame_flip)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    runing_time = 0.0
    while runing_time < 3.0:
        step_start = time.perf_counter()
        runing_time += dt

        phase = np.tanh(runing_time / 1.2)
        for i in range(ADAM_U_NUM_MOTOR):
            cmd.motor_cmd[i].q = phase * HOME[i] + (1 - phase) * EXE[i]
            # 使用配置的 Kp 和 Kd 值
            cmd.motor_cmd[i].kp = KP_CONFIG[i]
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = KD_CONFIG[i]
            cmd.motor_cmd[i].tau = 0.0
        for i in range(12):
            hand_cmd.position[i] = ROCK[i]

        pub.Write(cmd)
        hand_pub.Write(hand_cmd)

        time_until_next_step = dt - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


if __name__ == "__main__":
    run_simulation()
