import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

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

# 2. 标签编码
le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(le.classes_)
print("类别:", le.classes_)

# 3. 转 Tensor
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
)

# 4. 定义 MLP 模型
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

model = MLP(X.shape[1], HIDDEN, num_classes).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

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
