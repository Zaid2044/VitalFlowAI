"""
LSTM-based time-series risk predictor for VitalFlow AI.

Theory of Operation
-------------------
Each patient reading is a 6-dimensional feature vector:
    [blood_sugar, systolic_bp, diastolic_bp, heart_rate, temperature, spo2]

The LSTM processes a sliding window of the last SEQUENCE_LEN (10) readings
in chronological order.  The final hidden state of the last LSTM layer is
passed through a dropout + linear layer and a sigmoid activation, producing
a single probability in [0, 1]:

    P(alert event in next 24 hours | last 10 readings)

This is a binary classification problem.  The training label for a window is
1 if any reading that arrives within 24 hours after the window end has
`alert_triggered = True`, and 0 otherwise.  No manual labelling is required —
the label is derived entirely from the existing `alert_triggered` column in
the `readings` table.

Trend Detection (post-inference)
---------------------------------
After the LSTM outputs a risk probability, the module compares the mean of
the first half of the input window against the mean of the second half for
blood_sugar and systolic_bp.  A rising delta (> 5 units) is reported as
"Worsening"; a falling delta (< -5) as "Improving"; otherwise "Stable".
"""

import os
import pickle
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ── Constants ─────────────────────────────────────────────────────────────────

SEQUENCE_LEN = 10          # number of sequential readings fed to the LSTM
FEATURES: List[str] = [
    "blood_sugar", "systolic_bp", "diastolic_bp",
    "heart_rate", "temperature", "spo2",
]
N_FEATURES = len(FEATURES)

HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2
EPOCHS      = 40
BATCH_SIZE  = 32
LR          = 1e-3

MODEL_DIR   = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(MODEL_DIR, "lstm_risk_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "lstm_scaler.pkl")

# Population-level means used to impute missing vitals (None values in the DB)
FEATURE_MEANS: Dict[str, float] = {
    "blood_sugar":  100.0,
    "systolic_bp":  120.0,
    "diastolic_bp":  80.0,
    "heart_rate":    75.0,
    "temperature":   36.8,
    "spo2":          98.0,
}

# Module-level cache — model and scaler are loaded once per process
_model:  Optional[nn.Module]       = None
_scaler: Optional[StandardScaler]  = None


# ── Model Architecture ────────────────────────────────────────────────────────

class LSTMRiskModel(nn.Module):
    """
    Stacked LSTM → Dropout → Linear → Sigmoid.

    Input:  (batch, SEQUENCE_LEN, N_FEATURES)
    Output: (batch,)  — scalar event probability per sample
    """

    def __init__(
        self,
        n_features:  int,
        hidden_size: int,
        num_layers:  int,
        dropout:     float,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # Dropout between LSTM layers (no effect if num_layers == 1)
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x        : (batch, seq_len, n_features)
        # h_n      : (num_layers, batch, hidden_size)
        _, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1])           # take the last layer's hidden state
        return torch.sigmoid(self.fc(out)).squeeze(-1)   # (batch,)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ── Data Preparation ──────────────────────────────────────────────────────────

def _reading_to_row(r) -> List[float]:
    """Convert a SQLAlchemy Reading ORM object to a float list, imputing Nones."""
    return [
        r.blood_sugar  if r.blood_sugar  is not None else FEATURE_MEANS["blood_sugar"],
        r.systolic_bp  if r.systolic_bp  is not None else FEATURE_MEANS["systolic_bp"],
        r.diastolic_bp if r.diastolic_bp is not None else FEATURE_MEANS["diastolic_bp"],
        r.heart_rate   if r.heart_rate   is not None else FEATURE_MEANS["heart_rate"],
        r.temperature  if r.temperature  is not None else FEATURE_MEANS["temperature"],
        r.spo2         if r.spo2         is not None else FEATURE_MEANS["spo2"],
    ]


def extract_sequences_from_db(db) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (window, label) pairs from all patients in the live database.

    For each patient with >= SEQUENCE_LEN + 1 readings, we slide a window
    of SEQUENCE_LEN across their chronological reading history.  The label
    for window [i .. i+SEQUENCE_LEN-1] is 1 if any reading with a timestamp
    within 24 hours after the window's last timestamp has alert_triggered=True.

    Requires a SQLAlchemy Session that has `models.Patient` and `models.Reading`
    registered.
    """
    # Local imports prevent circular dependency when this module is imported
    # before FastAPI sets up the application context.
    import models  # noqa: PLC0415

    all_X: List[np.ndarray] = []
    all_y: List[int]        = []

    patients = db.query(models.Patient).all()
    for patient in patients:
        readings = (
            db.query(models.Reading)
            .filter(models.Reading.patient_id == patient.id)
            .order_by(models.Reading.timestamp.asc())
            .all()
        )
        if len(readings) < SEQUENCE_LEN + 1:
            continue

        matrix     = np.array([_reading_to_row(r) for r in readings], dtype=np.float32)
        timestamps = np.array([r.timestamp for r in readings], dtype="datetime64[s]")
        alerts     = np.array([1 if r.alert_triggered else 0 for r in readings])

        for i in range(len(readings) - SEQUENCE_LEN):
            window        = matrix[i : i + SEQUENCE_LEN]
            window_end_ts = timestamps[i + SEQUENCE_LEN - 1]

            future_mask = (
                (timestamps > window_end_ts) &
                (timestamps <= window_end_ts + np.timedelta64(24, "h"))
            )
            label = int(np.any(alerts[future_mask]))

            all_X.append(window)
            all_y.append(label)

    if not all_X:
        return np.empty((0, SEQUENCE_LEN, N_FEATURES), dtype=np.float32), np.empty(0, dtype=np.float32)

    return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.float32)


def generate_synthetic_sequences(
    n_patients: int = 150,
    readings_per_patient: int = 35,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fallback training data when the live DB contains insufficient history.

    Simulates stable patients (70%) and high-risk patients (30%).  High-risk
    patients have vitals that drift toward dangerous values over time.
    """
    np.random.seed(42)
    all_X: List[np.ndarray] = []
    all_y: List[int]        = []

    feature_noise_std = np.array([10.0, 8.0, 5.0, 8.0, 0.3, 1.0])
    high_risk_drift   = np.array([60.0, 25.0, 15.0, 30.0, 1.2, -4.0])
    feature_clip_lo   = np.array([50.0, 80.0, 50.0, 40.0, 35.0, 85.0])
    feature_clip_hi   = np.array([350.0, 200.0, 130.0, 160.0, 40.0, 100.0])

    for _ in range(n_patients):
        baseline = np.array([
            np.random.uniform(85, 130),    # blood_sugar
            np.random.uniform(110, 130),   # systolic_bp
            np.random.uniform(70, 82),     # diastolic_bp
            np.random.uniform(62, 88),     # heart_rate
            np.random.uniform(36.2, 37.0), # temperature
            np.random.uniform(96, 99),     # spo2
        ])
        is_high_risk = np.random.rand() < 0.30

        readings: List[np.ndarray] = []
        for t in range(readings_per_patient):
            drift = (t / readings_per_patient) if is_high_risk else 0.0
            noise = np.random.randn(N_FEATURES) * feature_noise_std
            row   = np.clip(baseline + drift * high_risk_drift + noise,
                            feature_clip_lo, feature_clip_hi)
            readings.append(row)

        matrix = np.array(readings, dtype=np.float32)
        for i in range(len(readings) - SEQUENCE_LEN):
            window = matrix[i : i + SEQUENCE_LEN]
            # Label = 1 only in the deteriorating tail of a high-risk patient
            label  = 1 if (is_high_risk and i > readings_per_patient * 0.55) else 0
            all_X.append(window)
            all_y.append(label)

    return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.float32)


# ── Training ──────────────────────────────────────────────────────────────────

def train_lstm(db=None) -> Tuple[nn.Module, StandardScaler]:
    """
    Train the LSTM on real DB data if sufficient, else on synthetic data.
    Saves weights to MODEL_PATH and scaler to SCALER_PATH.
    """
    global _model, _scaler

    X, y = np.empty(0), np.empty(0)
    if db is not None:
        X, y = extract_sequences_from_db(db)

    if len(X) < 50:
        print("[LSTM] Insufficient real data — training on synthetic sequences.")
        X, y = generate_synthetic_sequences()

    n_samples = len(X)
    # Flatten → fit StandardScaler → reshape back
    flat        = X.reshape(-1, N_FEATURES)
    scaler      = StandardScaler()
    flat_scaled = scaler.fit_transform(flat)
    X_scaled    = flat_scaled.reshape(n_samples, SEQUENCE_LEN, N_FEATURES).astype(np.float32)

    dataset = SequenceDataset(X_scaled, y)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model     = LSTMRiskModel(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg = epoch_loss / len(loader)
            print(f"[LSTM] Epoch {epoch+1:3d}/{EPOCHS}  avg_loss={avg:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    with open(SCALER_PATH, "wb") as fh:
        pickle.dump(scaler, fh)
    print(f"[LSTM] Model saved: {MODEL_PATH}")

    _model, _scaler = model, scaler
    return model, scaler


def load_lstm() -> Tuple[nn.Module, StandardScaler]:
    """Return cached (model, scaler), training if the saved files don't exist."""
    global _model, _scaler
    if _model is not None:
        return _model, _scaler

    if not os.path.exists(MODEL_PATH):
        _model, _scaler = train_lstm()
        return _model, _scaler

    model = LSTMRiskModel(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    with open(SCALER_PATH, "rb") as fh:
        scaler = pickle.load(fh)

    _model, _scaler = model, scaler
    return model, scaler


# ── Inference ─────────────────────────────────────────────────────────────────

def _dict_to_row(entry: dict) -> List[float]:
    """Convert a reading dict (possibly sparse) to an imputed feature list."""
    return [
        entry.get("blood_sugar")  or FEATURE_MEANS["blood_sugar"],
        entry.get("systolic_bp")  or FEATURE_MEANS["systolic_bp"],
        entry.get("diastolic_bp") or FEATURE_MEANS["diastolic_bp"],
        entry.get("heart_rate")   or FEATURE_MEANS["heart_rate"],
        entry.get("temperature")  or FEATURE_MEANS["temperature"],
        entry.get("spo2")         or FEATURE_MEANS["spo2"],
    ]


def predict_risk_lstm(reading_sequence: List[dict]) -> dict:
    """
    Predict the probability of a health alert event in the next 24 hours
    given the patient's recent reading history.

    Args:
        reading_sequence: List of reading dicts in chronological order (oldest
            first).  Each dict may have any subset of the keys:
            blood_sugar, systolic_bp, diastolic_bp, heart_rate, temperature, spo2.
            If fewer than SEQUENCE_LEN entries are provided the window is
            front-padded with population-mean values.

    Returns:
        {
            "risk_score":            float in [0, 1],
            "risk_level":            "Low" | "Medium" | "High",
            "trend":                 "Improving" | "Stable" | "Worsening",
            "sequence_length_used":  int,
            "model":                 "LSTM",
        }
    """
    model, scaler = load_lstm()

    # Take the most recent SEQUENCE_LEN readings
    recent = reading_sequence[-SEQUENCE_LEN:]
    rows   = [_dict_to_row(e) for e in recent]

    # Front-pad with population means if the sequence is too short
    while len(rows) < SEQUENCE_LEN:
        rows.insert(0, list(FEATURE_MEANS.values()))

    arr        = np.array(rows, dtype=np.float32)          # (SEQUENCE_LEN, N_FEATURES)
    arr_scaled = scaler.transform(arr)
    tensor     = torch.tensor(arr_scaled, dtype=torch.float32).unsqueeze(0)  # (1, seq, feat)

    model.eval()
    with torch.no_grad():
        prob = float(model(tensor).item())

    # ── Trend: compare first half vs second half of the window ───────────────
    # Use blood_sugar (idx 0) and systolic_bp (idx 1) as primary trend signals
    mid         = SEQUENCE_LEN // 2
    first_half  = arr[:mid,  [0, 1]].mean(axis=0)
    second_half = arr[mid:,  [0, 1]].mean(axis=0)
    delta       = float((second_half - first_half).mean())

    if delta > 5.0:
        trend = "Worsening"
    elif delta < -5.0:
        trend = "Improving"
    else:
        trend = "Stable"

    risk_level = "High" if prob >= 0.65 else ("Medium" if prob >= 0.35 else "Low")

    return {
        "risk_score":           round(prob, 3),
        "risk_level":           risk_level,
        "trend":                trend,
        "sequence_length_used": len(reading_sequence),
        "model":                "LSTM",
    }


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Training LSTM risk model on synthetic data …")
    train_lstm()

    # Smoke test with 10 synthetic readings
    test_seq = [
        {"blood_sugar": 95,  "systolic_bp": 118, "heart_rate": 72, "spo2": 98},
        {"blood_sugar": 98,  "systolic_bp": 120, "heart_rate": 74, "spo2": 97},
        {"blood_sugar": 105, "systolic_bp": 122, "heart_rate": 76, "spo2": 97},
        {"blood_sugar": 112, "systolic_bp": 125, "heart_rate": 78, "spo2": 96},
        {"blood_sugar": 120, "systolic_bp": 128, "heart_rate": 80, "spo2": 96},
        {"blood_sugar": 130, "systolic_bp": 132, "heart_rate": 84, "spo2": 95},
        {"blood_sugar": 145, "systolic_bp": 138, "heart_rate": 88, "spo2": 95},
        {"blood_sugar": 160, "systolic_bp": 144, "heart_rate": 92, "spo2": 94},
        {"blood_sugar": 175, "systolic_bp": 150, "heart_rate": 96, "spo2": 93},
        {"blood_sugar": 195, "systolic_bp": 158, "heart_rate": 102,"spo2": 92},
    ]
    result = predict_risk_lstm(test_seq)
    print("Inference result:", result)
