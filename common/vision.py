import mediapipe as mp
import torch

# 文件用途：视觉管线工具
# 最后修改：2025-12-04
# 主要功能：
# - 初始化 MediaPipe Hands
# - 将 landmark 打包为 63 维向量
# - 模型推理得到类别与置信度
# 使用说明：入口脚本统一使用以减少重复代码。


def init_hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return hands, mp_drawing


def landmarks_to_vector(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords


def infer_label(model, classes, landmarks_vec, device, use_softmax=True):
    x = torch.tensor(landmarks_vec, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        if use_softmax:
            pred = torch.softmax(out, dim=1)
            idx = pred.argmax(1).item()
            conf = pred[0, idx].item()
        else:
            idx = out.argmax(1).item()
            conf = out[0, idx].item()
    return classes[idx], conf
