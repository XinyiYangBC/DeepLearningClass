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
for var in ["VPD", "Surface_Temperature", "Precipitation", "Soil_Moisture", "Wind_Speed", "CAPE","EVI", "LAI", "NDVI"]:
    clim = df[var].groupby(df["Date"].dt.month).transform("mean")
    df[var + "_Anom"] = df[var] - clim
    df[f"{var}_Anom_Z"] = (df[var] - clim) / df[var].groupby(df["Date"].dt.month).transform("std")

df["EVI_Cubed"] = df["EVI"] ** 3
df["EVI_Anom_Cubed"] = df["EVI_Anom"] ** 3
df["VPD_Cubed"] = df["VPD"] ** 3
df["VPD_Anom_Cubed"] = df["VPD_Anom"] ** 3


# feat_cols = [
#     'Soil_Moisture', 'VPD', 'Surface_Temperature', 'Wind_Speed', "Precipitation", "Specific_Humidity", 
#     "Cropland_Area", "Pasture_Area", "Rangeland_Area", "Urban_Area","Population_Rural","Population_Urban", "Population_Total",
#     'EVI', "NDVI", "LAI"
# ]
feat_cols = [
    # absolute
    'Soil_Moisture', 'VPD',  'Wind_Speed', 'Surface_Temperature',  #"CAPE", #"Precipitation", #"Specific_Humidity", 
    "Cropland_Area",#"Pasture_Area", "Rangeland_Area", "Urban_Area","Population_Rural","Population_Urban", "Population_Total",
    'EVI',
    # anomalies
    'EVI_Anom', 'Soil_Moisture_Anom', 'VPD_Anom', 'Wind_Speed_Anom',#"CAPE_Anom",#'Surface_Temperature_Anom',
    # "EVI_SM_Anom_Interaction", 
    # Strong Nonlinearity
    # "EVI_Anom_Cubed","VPD_Anom_Cubed"
    
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
# 5️. Model Definition (LSTM + Dual Attention: Feature + Time)
# =====================================================
class LSTMFire_DualAttn(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # -------- Feature-wise Attention---------
        self.feat_attn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim) 
        )

        # -------- LSTM  --------
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # -------- Temporal Attention ---------
        self.time_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # -------- Fully Connected Layer --------
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        scores_feat = self.feat_attn(x)
        alpha_feat = torch.softmax(scores_feat, dim=-1)  # [B, T, F]
        x_weighted = alpha_feat * x  # [B, T, F]
        h_seq, _ = self.lstm(x_weighted)   # [B, T, H]
        scores_time = self.time_attn(h_seq)          # [B, T, 1]
        alpha_time = torch.softmax(scores_time, dim=1)  
        context = torch.sum(alpha_time * h_seq, dim=1)  # [B, H]
        out = self.fc(context)  # [B, 1]
        return out, alpha_time, alpha_feat


# =====================================================
# 6️. Training (with Best Model Saving)
# =====================================================
# model = LSTMFire_Attn(input_dim=X.shape[1], hidden_dim=hidden_dim).to(device)
model = LSTMFire_DualAttn(input_dim=X.shape[1], hidden_dim=hidden_dim).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

train_losses, val_losses = [], []

# ---- Best model saving setup ----
best_val_loss = float('inf')
best_model_path = "best_lstm_fire_model.pth"
patience = 100          # early stopping patience 
patience_counter = 0

for epoch in range(1, epoch_max + 1):

    # ---------------- Train ----------------
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        #pred, _ = model(xb)
        pred, _, _ = model(xb)   
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    train_loss = total_loss / len(train_loader.dataset)

    # ---------------- Val ----------------
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            #pred, _ = model(xb)
            pred, _, _ = model(xb)   
            total_val_loss += criterion(pred, yb).item() * xb.size(0)
    val_loss = total_val_loss / len(val_loader.dataset)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # ---------------- Print ----------------
    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    # ---------------- Save Best Model ----------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        patience_counter = 0
        # print(f"  -> Saved best model (val_loss = {val_loss:.4f})")
    else:
        patience_counter += 1

    # ---------------- Early Stopping ----------------
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}. Best val_loss = {best_val_loss:.4f}")
        break

# ---- Load best model before evaluation ----
model.load_state_dict(torch.load(best_model_path))
model.eval()
print("Loaded BEST model for evaluation.")

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
preds, trues = [], []
attn_time_all, attn_feat_all = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        pred, alpha_time, alpha_feat = model(xb)
        preds.append(pred.cpu())
        trues.append(yb)
        attn_time_all.append(alpha_time.cpu())  # [B, T, 1]
        attn_feat_all.append(alpha_feat.cpu())  # [B, T, F]

preds = torch.cat(preds).numpy().squeeze()
trues = torch.cat(trues).numpy().squeeze()


attn_time = torch.cat(attn_time_all, dim=0).mean(dim=0).squeeze().numpy()  # [T]


attn_feat = torch.cat(attn_feat_all, dim=0).mean(dim=0).mean(dim=0).numpy()  # [F]

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
# 10. Visualize Temporal Attention Weights
# =====================================================
plt.figure(figsize=(6,3))
plt.plot(attn_time, marker='o')
plt.title("Average Temporal Attention over 12 months")
plt.xlabel("Time Step (month lag)")
plt.ylabel("Attention Weight")
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()


# =====================================================
# 10b. Visualize Feature-wise Attention Weights
# =====================================================
plt.figure(figsize=(8,4))
plt.bar(range(len(feat_cols)), attn_feat)
plt.xticks(range(len(feat_cols)), feat_cols, rotation=45, ha='right')
plt.ylabel("Average Feature Attention Weight")
plt.title("Average Feature-wise Attention (across time & samples)")
plt.tight_layout()
plt.show()


# =====================================================
# 11. Plot Full Timeline: Train + Val + Test
# =====================================================
model.eval()


train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=batch_size, shuffle=False)

# Train predictions
train_preds = []
with torch.no_grad():
    for xb, yb in train_loader:
        xb = xb.to(device)
        pred, _, _ = model(xb)
        train_preds.append(pred.cpu())
train_preds = torch.cat(train_preds).numpy().squeeze()
train_preds = np.expm1(train_preds)

# Val predictions
val_preds = []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        pred, _, _ = model(xb)
        val_preds.append(pred.cpu())
val_preds = torch.cat(val_preds).numpy().squeeze()
val_preds = np.expm1(val_preds)

# Test predictions 
test_preds = preds_orig

full_pred = np.concatenate([train_preds, val_preds, test_preds])
full_true = np.concatenate([np.expm1(y_train.squeeze()),
                            np.expm1(y_val.squeeze()),
                            np.expm1(y_test.squeeze())])

r, p_value = pearsonr(full_true, full_pred)
full_r2 = r**2

print(f"Full Timeline R² = {full_r2:.3f}, p = {p_value:.3e}")

plt.figure(figsize=(12,4))
dates = pd.to_datetime(df["Date"].values)
y_dates = dates[seq_len-1:] 
plt.plot(y_dates, full_true, label='True', color='black', lw=1.8)
plt.plot(y_dates, full_pred, label='Predicted', color='orange', lw=2.8)

plt.axvline(y_dates[len(train_preds)], color='gray', linestyle='--')
plt.axvline(y_dates[len(train_preds) + len(val_preds)], color='gray', linestyle='--')

plt.title(f"Full Time Series Prediction (R² = {full_r2:.2f}, p = {p_value:.1e})")
# plt.xlabel("Time Index (Monthly)")
plt.ylabel("Burned Area (Mha)")
plt.legend(ncol=1, loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
