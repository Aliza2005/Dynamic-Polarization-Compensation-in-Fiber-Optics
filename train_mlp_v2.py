import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

print("====================================================")
print("     EPC MLP STATIC MAPPING - TRAINING v2           ")
print("====================================================\n")

# --- 1. Configuration ---
CSV_FILE       = "ml_training_data.csv"
BATCH_SIZE     = 32
EPOCHS         = 400
LR             = 5e-4
WARMUP_FRAC    = 0.2      # first 20% epochs use MSE warmup, then switch to angular

if not os.path.exists(CSV_FILE):
    print(f"ERROR: '{CSV_FILE}' not found.")
    exit()

# --- 2. Load & Validate Data ---
df = pd.read_csv(CSV_FILE)
v_cols = ['V0','V1','V2'] if 'V0' in df.columns else ['V1','V2','V3']
s_cols = ['S1','S2','S3']

X_raw = df[v_cols].values.astype(np.float32)
y_raw = df[s_cols].values.astype(np.float32)

# Always unit-normalise targets - we want direction, not magnitude
norms = np.linalg.norm(y_raw, axis=1, keepdims=True)
y_raw = y_raw / np.clip(norms, 1e-9, None)

n = len(df)
print(f"Loaded {n} samples.")

# Quick consistency check - warn if near-duplicate voltages give very different Stokes
print("Running quick data consistency check...")
from scipy.spatial import cKDTree
tree = cKDTree(X_raw)
pairs = tree.query_pairs(r=0.3)
if pairs:
    bad = 0
    for i,j in pairs:
        cos = np.clip(np.dot(y_raw[i], y_raw[j]), -1, 1)
        if np.degrees(np.arccos(cos)) > 10:
            bad += 1
    pct = bad / len(pairs) * 100
    print(f"  {len(pairs)} near-duplicate voltage pairs found.")
    if pct > 20:
        print(f"  WARNING: {pct:.0f}% have Stokes disagreement >10°.")
        print("  Data may not have settled fully - consider longer convergence wait.")
    else:
        print(f"  OK: only {pct:.0f}% have Stokes disagreement >10°. Data looks clean.")
else:
    print("  No near-duplicate voltage pairs found (dataset covers space well).")

# --- 3. Feature Engineering ---
# Raw voltages PLUS sin/cos of normalized voltage (helps MLP learn the
# sinusoidal/periodic EPC transfer function without discovering it from scratch)
def make_features(X):
    X_n = X / 10.0                      # [0,1]
    X_s = np.sin(np.pi * X_n)           # captures half-wave periodicity
    X_c = np.cos(np.pi * X_n)           # quadrature component
    return np.hstack([X_n, X_s, X_c])   # 9 features total

X_feat = make_features(X_raw).astype(np.float32)
N_FEAT = X_feat.shape[1]  # 9

# --- 4. Train / Val split ---
X_tr, X_val, y_tr, y_val = train_test_split(
    X_feat, y_raw, test_size=0.2, random_state=42)

X_tr_t  = torch.from_numpy(X_tr)
y_tr_t  = torch.from_numpy(y_tr)
X_val_t = torch.from_numpy(X_val)
y_val_t = torch.from_numpy(y_val)

loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                    batch_size=BATCH_SIZE, shuffle=True)

# --- 5. Deeper / Wider MLP ---
class PolarizationMLP(nn.Module):
    def __init__(self, in_f=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(128,  64), nn.ReLU(),
            nn.Linear(64,    3),
        )
    def forward(self, x):
        return self.net(x)

model = PolarizationMLP(in_f=N_FEAT)
opt   = optim.Adam(model.parameters(), lr=LR)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
mse   = nn.MSELoss()

def angular_loss(pred, tgt, eps=1e-7):
    pn = pred / (pred.norm(dim=1, keepdim=True) + eps)
    tn = tgt  / (tgt.norm(dim=1, keepdim=True)  + eps)
    return torch.acos((pn * tn).sum(1).clamp(-1+eps, 1-eps)).mean()

def val_angular_error_deg(model, Xv, yv):
    model.eval()
    with torch.no_grad():
        p = model(Xv).numpy()
    pn = p / np.linalg.norm(p, axis=1, keepdims=True)
    yn = yv.numpy() / np.linalg.norm(yv.numpy(), axis=1, keepdims=True)
    cos = np.clip((pn * yn).sum(1), -1, 1)
    errs = np.degrees(np.arccos(cos))
    return errs.mean(), np.median(errs), np.percentile(errs, 95), errs.max()

# --- 6. Training Loop ---
print(f"\nTraining MLP (9 features, 256-256-128-64-3)...")
print(f"Warmup: MSE for first {int(EPOCHS*WARMUP_FRAC)} epochs, then Angular loss.\n")

warmup_epochs = int(EPOCHS * WARMUP_FRAC)
best_mean_err = float('inf')
best_state    = None

for epoch in range(EPOCHS):
    model.train()
    ep_loss = 0.0
    use_angular = epoch >= warmup_epochs
    for bx, by in loader:
        opt.zero_grad()
        pred = model(bx)
        loss = angular_loss(pred, by) if use_angular else mse(pred, by)
        loss.backward()
        opt.step()
        ep_loss += loss.item() * bx.size(0)
    sched.step()

    if (epoch + 1) % 50 == 0 or epoch == 0:
        mean_e, med_e, p95_e, max_e = val_angular_error_deg(model, X_val_t, y_val_t)
        tag = "ANGULAR" if use_angular else "MSE-warmup"
        print(f" Epoch {epoch+1:03d}/{EPOCHS} [{tag}] | "
              f"Val: mean={mean_e:.2f}°  median={med_e:.2f}°  "
              f"p95={p95_e:.2f}°  max={max_e:.2f}°")
        if mean_e < best_mean_err:
            best_mean_err = mean_e
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

# --- 7. Final Evaluation ---
model.load_state_dict(best_state)
mean_e, med_e, p95_e, max_e = val_angular_error_deg(model, X_val_t, y_val_t)

print("\n--- MLP v2 METRIC VALIDATION REPORT ---")
print(f"Average Angular Error:    {mean_e:.2f}°")
print(f"Median  Angular Error:    {med_e:.2f}°")
print(f"95th Percentile Error:    {p95_e:.2f}°")
print(f"Maximum Angular Error:    {max_e:.2f}°")

THRESHOLD = 5.0
if mean_e <= THRESHOLD:
    print(f"\nSUCCESS: Mean error {mean_e:.2f}° is below {THRESHOLD}° target.")
    torch.save(best_state, "mlp_brain_weights.pt")
    # Save the feature function config alongside the scaler so inference matches
    joblib.dump({'v_cols': v_cols, 'n_feat': N_FEAT}, "mlp_config.joblib")
    print("Saved: mlp_brain_weights.pt  mlp_config.joblib")
    print("\nNext step: update app2.py inference to use this MLP.")
else:
    print(f"\nWARNING: Mean error {mean_e:.2f}° still above {THRESHOLD}° target.")
    if n < 3000:
        print(f"  -> Primary likely cause: only {n} samples.")
        print(f"     For a 3D voltage space, aim for 3000-5000 well-settled samples.")
        print(f"     Run app2_fixed.py and collect more before retraining.")
    if p95_e > 30:
        print(f"  -> p95={p95_e:.2f}° suggests sparse regions with poor coverage.")
        print(f"     More samples needed to cover the voltage space densely.")
    # Save anyway so you can inspect / use for partial compensation
    torch.save(best_state, "mlp_brain_weights.pt")
    joblib.dump({'v_cols': v_cols, 'n_feat': N_FEAT}, "mlp_config.joblib")
    print("  Saved best checkpoint anyway for inspection.")
