# 文件用途：手势识别模型与加载工具
# 最后修改：2025-12-04
# 主要功能：
# - 定义三层 MLP（GestureMLP）
# - 统一加载模型与类别集合（load_gesture_mlp）
# 使用说明：入口脚本统一从此处加载模型与类别。

import torch
import torch.nn as nn


class GestureMLP(nn.Module):
    """三层 MLP：输入 63 维手部关键点坐标，输出 3+ 类手势

    解释：
    - Linear：线性层（加权求和），用于特征变换。
    - ReLU：激活函数，负数置 0，增强非线性表达能力。
    - input_size=63：21 个关键点 × 每点 (x,y,z)。
    - num_classes：由训练时的类别集合决定（通常是 rock/paper/scissors）。
    """

    def __init__(self, input_size=63, hidden_size=128, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_gesture_mlp(model_path: str, device: str, input_size: int = 63, hidden_size: int = 128):
    """加载训练好的 MLP 模型并返回 (model, classes)

    参数：
    - model_path：训练后保存的权重文件路径（rps_mlp.pth）。
    - device：'cuda' 或 'cpu'，根据你的环境自动选择。
    - input_size/hidden_size：需与训练时保持一致（默认 63/128）。
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    classes = list(ckpt["classes"])  # 训练时的类别集合
    model = GestureMLP(input_size=input_size, hidden_size=hidden_size, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, classes

