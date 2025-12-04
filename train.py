import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# 文件用途：训练入口，生成手势识别模型权重
# 最后修改：2025-12-04
# 主要功能：
# - 读取 data/*.npy 并做标签编码
# - 训练 MLP 并评估分类结果
# - 保存权重与类别集合到 rps_mlp.pth
# 重要组件：MLP、DataLoader、CrossEntropyLoss、Adam
# 使用说明：采集数据后运行本文件以更新模型权重。
# ---------- 配置 ----------
DATA_DIR = "data"
MODEL_PATH = "rps_mlp.pth"
EPOCHS = 80
BATCH_SIZE = 32
LR = 1e-3
HIDDEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------

# 1. 读取数据
X = np.load(os.path.join(DATA_DIR, "dataset.npy"))
y = np.load(os.path.join(DATA_DIR, "labels.npy"))
print("✅ 数据加载完成:", X.shape, y.shape)
# 解释：X 形如 [样本数, 63]；y 是同等长度的标签数组（字符串）。

# 2. 标签编码
le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(le.classes_)
print("类别:", le.classes_)
# 解释：把字符串标签转成数字 0..(num_classes-1)，并保存类别名以便推理时显示。

# 3. 转 Tensor
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
# 解释：20% 测试集，固定随机种子 42 保证结果可复现；stratify 保证各类比例在训练/测试中大致相同。
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)
# 解释：
# - float32：特征用浮点数。
# - long：标签用整型（分类时需要）。

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
)
# 解释：DataLoader 会把数据按 batch 分组，训练时 shuffle=True 打乱顺序更利于学习。

# 4. 定义 MLP 模型
class MLP(nn.Module):
    """三层感知机，用于手势分类"""
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

model = MLP(X.shape[1], HIDDEN, num_classes).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
# 解释：
# - to(DEVICE)：如果支持 CUDA 就放到显卡上训练；否则用 CPU。
# - Adam：优化器；CrossEntropyLoss：多分类常用损失函数。

# 5. 训练
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out = model(xb)
        loss = criterion(out, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}/{EPOCHS} | loss={avg_loss:.4f}")

# 6. 评估
model.eval()
preds, targets = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        out = model(xb.to(DEVICE))
        preds.extend(out.argmax(1).cpu().numpy())
        targets.extend(yb.numpy())

print("✅ 分类结果：")
print(classification_report(targets, preds, target_names=le.classes_))

# 7. 保存模型和标签编码
torch.save({
    "model_state": model.state_dict(),
    "classes": le.classes_,
}, MODEL_PATH)
print(f"💾 模型已保存: {MODEL_PATH}")
# 解释：保存模型权重(state_dict)和类别名，方便推理脚本直接加载使用。
