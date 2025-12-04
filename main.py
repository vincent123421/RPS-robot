import cv2
import mediapipe as mp
import torch
import numpy as np
import mujoco
import mujoco.viewer
from collections import deque
import time
import threading
from common.models import load_gesture_mlp
from common.strategy import get_winning_move

# 名词解释（小白友好）：
# - MediaPipe：谷歌的视觉 AI 库，能从摄像头画面中提取“手部关键点”。
# - 关键点(Landmark)：手指关节等位置的坐标点，每点含 x/y/z，合计 21×3=63 维。
# - 张量(Tensor)：PyTorch 的多维数组格式，适合做神经网络计算。
# - MLP（多层感知机）：基础神经网络，由“线性层 + 激活层”堆叠，用于分类手势。
# - MuJoCo：物理仿真引擎，模拟机械结构的运动与碰撞；本项目用它来“动手指”。
# - qpos / qvel：MuJoCo 中关节的“位置数组/速度数组”，分别表示角度和角速度。
# - dt(m.opt.timestep)：每次物理仿真的“时间步长”，例如 0.002 秒/步。
# - viewer：MuJoCo 自带的渲染窗口，用来可视化模型的当前姿态。
# 文件用途：仿真入口，将手势映射到机械臂动作
# 最后修改：2025-12-04
# 主要功能：
# - 读取模型文件 rps_mlp.pth
# - 读取仿真模型 adam_u.xml
# - 摄像头采集与手势识别
# - 必胜策略映射与物理仿真渲染
# 重要组件：ThreadedCamera、RealisticServoController、run_simulation
# 使用说明：仿真需求运行 main.py；仅测策略运行 realtime.py。
# ==========================================
# 1. 极速配置
# ==========================================
XML_PATH = "adam_u.xml"
MODEL_PATH = "rps_mlp.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ⚡ SERVO_SPEED：我们为“虚拟舵机”设定的最大角速度(弧度/秒)，数值越大手指越快。
SERVO_SPEED = 20.0 
# ⚡ 相机分辨率：越低处理越快，640x480 是常见的折中方案。
CAM_WIDTH, CAM_HEIGHT = 640, 480

# 关节与动作配置：
# - BEND_VAL / STRAIGHT_VAL：手指“弯曲/伸直”时的目标角度（此处用 1.0/0.0 简化）。
BEND_VAL = 1.0  
STRAIGHT_VAL = 0.0
FINGER_JOINTS = {
    "thumb":  ["R_thumb_MCP_joint1", "R_thumb_MCP_joint2", "R_thumb_PIP_joint", "R_thumb_DIP_joint"],
    "index":  ["R_index_MCP_joint", "R_index_DIP_joint"],
    "middle": ["R_middle_MCP_joint", "R_middle_DIP_joint"],
    "ring":   ["R_ring_MCP_joint", "R_ring_DIP_joint"],
    "pinky":  ["R_pinky_MCP_joint", "R_pinky_DIP_joint"]
}
# 解释：
# - MCP / PIP / DIP：解剖学名称，分别是掌指关节/近指间关节/远指间关节；拇指结构略有不同。
ARM_POSE = {"shoulderPitch_Right": -0.5, "elbow_Right": -1.0, "wristPitch_Right": 0.0}
# 解释：
# - shoulderPitch/elbow/wristPitch：右臂的三个主要关节名；负值代表某方向弯曲（单位：弧度）。
GESTURES = {
    "rock":     {"thumb": 1, "index": 1, "middle": 1, "ring": 1, "pinky": 1},
    "paper":    {"thumb": 0, "index": 0, "middle": 0, "ring": 0, "pinky": 0},
    "scissors": {"thumb": 1, "index": 0, "middle": 0, "ring": 1, "pinky": 1},
    "waiting":  {"thumb": 0, "index": 0, "middle": 0, "ring": 0, "pinky": 0} 
}

# ==========================================
# 2. 多线程摄像头类 (核心优化)
# ==========================================
class ThreadedCamera:
    """多线程摄像头读取器：后台拉流，前台拿副本，降低读取阻塞与撕裂"""
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
        """后台线程循环：持续读取最新帧并更新状态"""
        while not self.stop_event:
            status, frame = self.capture.read()
            if status:
                with self.lock:
                    self.frame = frame
                    self.status = status
            else:
                self.stop_event = True

    def get_frame(self):
        """返回一帧的安全副本，避免上层修改底层缓冲导致竞争"""
        with self.lock:
            return self.status, self.frame.copy() # 返回副本以防竞争

    def release(self):
        """停止后台线程并释放摄像头资源"""
        self.stop_event = True
        self.thread.join()
        self.capture.release()

# ==========================================
# 3. 逻辑与模型（统一从 common 导入）
# ==========================================

# ==========================================
# 4. 仿真控制器
# ==========================================
class RealisticServoController:
    """MuJoCo 关节控制器：维护目标 `qpos` 并以最大角速度逼近目标

    术语解释：
    - mj_name2id：通过名字查找 MuJoCo 对象（这里是“关节”）的内部编号 id。
    - jnt_qposadr：每个关节在 `qpos`（位置数组）中的起始下标地址。
    - qpos：所有关节的“位置”（角度）集合；qvel：所有关节的“速度”（角速度）。
    - 目标表(target_qpos)：我们希望每个关节到达的角度，用于渐进逼近实现“平滑移动”。
    """
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.joint_ids = {}
        self.target_qpos = {}
        
        for finger, joint_names in FINGER_JOINTS.items():
            ids = []
            for name in joint_names:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid != -1:
                    ids.append(jid)
                    addr = model.jnt_qposadr[jid]
                    self.target_qpos[addr] = data.qpos[addr]
            self.joint_ids[finger] = ids
            
        self.arm_ids = {}
        for name in ARM_POSE.keys():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid != -1: self.arm_ids[name] = jid

    def set_initial_pose(self):
        """设定初始右臂姿态并前向计算一次，使仿真状态一致

        解释：
        - mj_forward：让 MuJoCo 根据当前 `qpos`/`qvel` 重新计算躯体状态（刚体变换等）。
        - 将浮动底座 z 坐标设为 1.0，把模型“抬高”到摄像机更易观察的高度。
        """
        for name, angle in ARM_POSE.items():
            if name in self.arm_ids:
                self.data.qpos[self.model.jnt_qposadr[self.arm_ids[name]]] = angle
        if self.model.nq >= 7: self.data.qpos[2] = 1.0 
        mujoco.mj_forward(self.model, self.data)

    def apply_gesture(self, gesture_name):
        """根据手势配置更新各手指关节的目标角度(弯曲/伸直)"""
        target_config = GESTURES.get(gesture_name, GESTURES["waiting"])
        for finger, is_bent in target_config.items():
            target_angle = BEND_VAL if is_bent else STRAIGHT_VAL
            if finger in self.joint_ids:
                for jid in self.joint_ids[finger]:
                    self.target_qpos[self.model.jnt_qposadr[jid]] = target_angle

    def update_servos(self, dt):
        """将当前角度以步长限制 `SERVO_SPEED*dt` 推进到目标角度，得到平滑控制

        解释：
        - step_limit：每一仿真步允许的最大角度变化，防止“瞬间跳变”。
        - np.clip：把变化量限制在 [-step_limit, +step_limit] 范围内，实现“匀速逼近”。
        """
        for addr, target in self.target_qpos.items():
            current = self.data.qpos[addr]
            step_limit = SERVO_SPEED * dt
            diff = target - current
            if abs(diff) > 1e-4:
                self.data.qpos[addr] = current + np.clip(diff, -step_limit, step_limit)

# ==========================================
# 5. 主程序
# ==========================================
def run_simulation():
    """主循环：摄像头采集 → 关键点推理 → 必胜手势 → 仿真步进/渲染

    解释：
    - viewer.launch_passive：以“被动模式”启动渲染窗口（我们自己控制 mj_step 和 viewer.sync）。
    - m.opt.timestep(dt)：每次仿真推进的时间量，影响“速度”和“平滑程度”。
    - sim_time_budget：给物理引擎的时间配额（例如 5ms），避免阻塞视觉推理。
    """
    # Load Model（统一接口）
    net, classes = load_gesture_mlp(MODEL_PATH, DEVICE)

    # Init MediaPipe (CPU Mode, but optimized)
    mp_hands = mp.solutions.hands
    # ⚠️ 降低置信度阈值：让它更早地检测到手，哪怕有点模糊
    hands = mp_hands.Hands(
        max_num_hands=1, 
        min_detection_confidence=0.3, 
        min_tracking_confidence=0.3
    )

    # Init MuJoCo
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    m.opt.gravity = (0, 0, 0)
    controller = RealisticServoController(m, d)
    controller.set_initial_pose()

    # Init Threaded Camera
    print("🎥 启动多线程摄像头采集...")
    cam = ThreadedCamera(0)
    time.sleep(1.0) # 等待摄像头预热

    current_robot_move = "waiting"
    dt = m.opt.timestep 
    
    # FPS Calculation
    fps_start = time.time()
    frames = 0
    inference_time = 0

    print("🚀 极速模式已启动 - 请测试")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and cam.status:
            loop_start = time.time()
            
            # 1. 获取最新帧 (非阻塞)
            ret, frame = cam.get_frame()
            if not ret: break
            
            # 2. 图像处理
            frame_flip = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2RGB)
            
            t0 = time.time()
            result = hands.process(rgb)
            # 解释：MediaPipe 对当前画面做手部检测与跟踪，返回关键点集合。
            inference_time = (time.time() - t0) * 1000 # ms
            
            user_move = "waiting"
            if result.multi_hand_landmarks:
                landmarks = []
                for lm in result.multi_hand_landmarks[0].landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                # MLP Inference
                x_tensor = torch.tensor(landmarks, dtype=torch.float32).to(DEVICE).unsqueeze(0)
                # 解释：把 63 维数字打包成“批量=1”的张量，送进神经网络。
                with torch.no_grad():
                    preds = net(x_tensor)
                    # 核心逻辑：只要网络输出对某类更倾向（分数更高），立刻当作该手势使用。
                    user_move = classes[torch.argmax(preds, dim=1).item()]

            # 3. 核心：根据玩家手势计算机器人必胜手势，并更新目标
            winning_move = get_winning_move(user_move)
            if winning_move != current_robot_move:
                current_robot_move = winning_move
                controller.apply_gesture(current_robot_move)

            # 4. 物理步进 (追赶时间)
            # 尽量保持物理循环不阻塞视觉循环
            sim_time_budget = 0.005 # 给物理引擎分配 5ms（时间配额）
            sim_start = time.time()
            while time.time() - sim_start < sim_time_budget:
                controller.update_servos(dt)
                d.qvel[:] = 0           # 解释：把角速度清零，避免累积速度导致抖动或失稳。
                d.qpos[0:3] = [0,0,1]   # 解释：固定底座位置（x,y,z），让模型不漂移。
                mujoco.mj_step(m, d)    # 解释：推进一次物理仿真步（基于当前 qpos/qvel）。
            
            viewer.sync()  # 解释：刷新渲染窗口到最新仿真状态。
            
            # 5. 信息显示
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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_simulation()
