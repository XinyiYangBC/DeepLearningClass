import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ===========Pre-Setting=========
seq_len = 12
batch_size = 32
hidden_dim=64
lr = 5e-4
weight_decay=1e-6
epoch_max = 200

# ======================= 0. Reproducibility =======================
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================= 1. Load Data =======================
data_path = "/home/yangbuw/Program/DeepLearning/code/DeepLearning_Climate_Biomass_Human_Fire_Dataset_2001_2024_montly.xlsx"
# data_path = "/home/yangbuw/Program/DeepLearning/code/DeepLearning_Climate_Biomass_Human_Fire_Dataset_2001_2024_montly_grid.xlsx"
sheet_name = "Dataset"
df = pd.read_excel(data_path, sheet_name=sheet_name)

# ======================= 2. Features & Target =======================
# feat_cols = [
#     'Soil_Moisture', 'VPD', 'Surface_Temperature', 'Wind_Speed', "Precipitation", "Specific_Humidity", 
#     "Cropland_Area", "Pasture_Area", "Rangeland_Area", "Urban_Area","Population_Rural","Population_Urban", "Population_Total",
#     'EVI', "NDVI", "LAI"
# ]
feat_cols = [
    'Soil_Moisture', 'VPD', 'Surface_Temperature', 'Wind_Speed', #"Precipitation", "Specific_Humidity", 
    "Cropland_Area",#"Pasture_Area", "Rangeland_Area", "Urban_Area","Population_Rural","Population_Urban", "Population_Total",
    'EVI'
]
target_col = 'Burned_Area'

X = df[feat_cols].values.astype(np.float32)
y = df[target_col].values.astype(np.float32)
y = np.log1p(y) 

# =====================================================
# 3️. Construct time windows
# =====================================================
def make_windows(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len+1):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len-1])
    return np.array(Xs, np.float32), np.array(ys, np.float32).reshape(-1, 1)

X_win, y_win = make_windows(X, y, seq_len)

# =====================================================
# 4️. Train / Val / Test Split
# =====================================================
n_total = len(X_win)
n_train = int(0.7 * n_total)
n_val   = int(0.85 * n_total)

X_train, y_train = X_win[:n_train], y_win[:n_train]
X_val,   y_val   = X_win[n_train:n_val], y_win[n_train:n_val]
X_test,  y_test  = X_win[n_val:], y_win[n_val:]

x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train.reshape(-1, X.shape[1])).reshape(X_train.shape)
X_val   = x_scaler.transform(X_val.reshape(-1, X.shape[1])).reshape(X_val.shape)
X_test  = x_scaler.transform(X_test.reshape(-1, X.shape[1])).reshape(X_test.shape)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=batch_size, shuffle=False)

val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
    batch_size=batch_size, shuffle=False)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
    batch_size=batch_size, shuffle=False)

# =====================================================
# 5️. Model Definition (LSTM + Additive Attention)
# =====================================================
class LSTMFire_Attn(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        h_seq, _ = self.lstm(x)     # [B, T, H]
        scores = self.attn(h_seq)   # [B, T, 1]
        alpha = torch.softmax(scores, dim=1)  # attention weight
        context = torch.sum(alpha * h_seq, dim=1) 
        out = self.fc(context)
        return out, alpha

# =====================================================
# 6️. Training
# =====================================================
model = LSTMFire_Attn(input_dim=X.shape[1], hidden_dim=hidden_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

train_losses, val_losses = [], []

for epoch in range(1, epoch_max + 1):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred, _ = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    train_loss = total_loss / len(train_loader.dataset)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred, _ = model(xb)
            total_val_loss += criterion(pred, yb).item() * xb.size(0)
    val_loss = total_val_loss / len(val_loader.dataset)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

# =====================================================
# 7️. Plot Loss Curve
# =====================================================
plt.figure(figsize=(6,4))
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.legend(); plt.title("LSTM-Attn Training Curve"); plt.grid(True)
plt.tight_layout()
plt.show()

# =====================================================
# 8️. Evaluation on Test Set
# =====================================================
model.eval()
preds, trues, attn_all = [], [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        pred, alpha = model(xb)
        preds.append(pred.cpu())
        trues.append(yb)
        attn_all.append(alpha.cpu())

preds = torch.cat(preds).numpy().squeeze()
trues = torch.cat(trues).numpy().squeeze()
attn = torch.cat(attn_all).mean(dim=0).squeeze().numpy()

# Back-transform from log1p
preds_orig = np.expm1(preds)
trues_orig = np.expm1(trues)

r2 = r2_score(trues_orig, preds_orig)
print(f"Test R²_Score = {r2:.3f}")
r, p_value = pearsonr(trues_orig, preds_orig)
r2_pearson = r**2
print(f"Test R² = {r2_pearson:.3f}, p = {p_value:.3e}")

# =====================================================
# 9️. Plot Predictions
# =====================================================
plt.figure(figsize=(7,4))
plt.plot(trues_orig, label='True', lw=2)
plt.plot(preds_orig, label='Predicted', lw=2)
plt.legend(); plt.grid(True, alpha=0.4)
plt.title(f"LSTM-Attn Burned Area Prediction (R² = {r2:.2f})")
plt.tight_layout()
plt.show()

# =====================================================
# 10. Visualize Attention Weights
# =====================================================
plt.figure(figsize=(6,3))
plt.plot(attn, marker='o')
plt.title("Average Temporal Attention over 12 months")
plt.xlabel("Time Step (month lag)")
plt.ylabel("Attention Weight")
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()


# =====================================================
# 11. Plot Full Timeline: Train + Val + Test
# =====================================================

# ---- 1) 在 train & val 阶段也生成预测值 ----
model.eval()


train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=batch_size, shuffle=False)

# Train predictions
train_preds = []
with torch.no_grad():
    for xb, yb in train_loader:
        xb = xb.to(device)
        pred, _ = model(xb)
        train_preds.append(pred.cpu())
train_preds = torch.cat(train_preds).numpy().squeeze()
train_preds = np.expm1(train_preds)

# Val predictions
val_preds = []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        pred, _ = model(xb)
        val_preds.append(pred.cpu())
val_preds = torch.cat(val_preds).numpy().squeeze()
val_preds = np.expm1(val_preds)

# Test predictions (已经有，将重用)
test_preds = preds_orig

# ---- 2) 拼接所有预测值 ----
full_pred = np.concatenate([train_preds, val_preds, test_preds])
full_true = np.concatenate([np.expm1(y_train.squeeze()),
                            np.expm1(y_val.squeeze()),
                            np.expm1(y_test.squeeze())])

# ---- 3) 计算整体 R² + p-value ----
r, p_value = pearsonr(full_true, full_pred)
full_r2 = r**2

print(f"Full Timeline R² = {full_r2:.3f}, p = {p_value:.3e}")

# ---- 4) 绘图 ----
plt.figure(figsize=(12,4))
dates = pd.to_datetime(df["Date"].values)
y_dates = dates[seq_len-1:] 
plt.plot(y_dates, full_true, label='True', color='black', lw=1.8)
plt.plot(y_dates, full_pred, label='Predicted', color='orange', lw=1.8)

# 用竖线标记 Train/Val/Test 分段
plt.axvline(y_dates[len(train_preds)], color='gray', linestyle='--')
plt.axvline(y_dates[len(train_preds) + len(val_preds)], color='gray', linestyle='--')

plt.title(f"Full Time Series Prediction (R² = {full_r2:.2f}, p = {p_value:.1e})")
# plt.xlabel("Time Index (Monthly)")
plt.ylabel("Burned Area (Mha)")
plt.legend(ncol=1, loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
