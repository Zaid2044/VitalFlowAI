import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


MODEL_PATH = os.path.join(os.path.dirname(__file__), "risk_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "risk_scaler.pkl")


def generate_synthetic_data(n_samples: int = 1000):
    """
    Generate synthetic patient data for training.
    Features: adherence_7day, missed_streak, avg_energy, symptom_score, age_group, disease_encoded
    Target: risk_level (0=Low, 1=Medium, 2=High)
    """
    np.random.seed(42)

    adherence_7day = np.random.uniform(0, 100, n_samples)
    missed_streak = np.random.randint(0, 14, n_samples)
    avg_energy = np.random.uniform(1, 10, n_samples)
    symptom_score = np.random.uniform(0, 10, n_samples)
    age_group = np.random.randint(1, 5, n_samples)         # 1=young,2=mid,3=senior,4=elderly
    disease_encoded = np.random.randint(0, 5, n_samples)   # 0-4 disease types

    X = np.column_stack([
        adherence_7day, missed_streak, avg_energy,
        symptom_score, age_group, disease_encoded
    ])

    # Risk logic: low adherence + high missed + high symptoms = higher risk
    risk_score = (
        (100 - adherence_7day) * 0.4 +
        missed_streak * 3.0 +
        (10 - avg_energy) * 2.0 +
        symptom_score * 3.0 +
        age_group * 1.5
    )

    y = np.where(risk_score > 65, 2, np.where(risk_score > 35, 1, 0))

    return X, y


def train_model():
    X, y = generate_synthetic_data(1000)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print("Risk model trained and saved.")
    return model, scaler


def load_model():
    if not os.path.exists(MODEL_PATH):
        return train_model()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def predict_risk(
    adherence_7day: float,
    missed_streak: int,
    avg_energy: float,
    symptom_score: float,
    age: int,
    disease_type: str
) -> dict:
    """
    Returns risk_score (0-1), risk_level (Low/Medium/High)
    """
    disease_map = {
        "diabetes": 0, "hypertension": 1, "heart disease": 2,
        "respiratory": 3, "other": 4
    }
    disease_encoded = disease_map.get(disease_type.lower(), 4)

    age_group = 1 if age < 30 else (2 if age < 50 else (3 if age < 70 else 4))

    features = np.array([[
        adherence_7day, missed_streak, avg_energy,
        symptom_score, age_group, disease_encoded
    ]])

    model, scaler = load_model()
    features_scaled = scaler.transform(features)

    proba = model.predict_proba(features_scaled)[0]
    predicted_class = int(np.argmax(proba))

    # Use probability of high risk (class 2) as the score
    risk_score = float(proba[2])

    level_map = {0: "Low", 1: "Medium", 2: "High"}
    risk_level = level_map[predicted_class]

    return {
        "risk_score": round(risk_score, 3),
        "risk_level": risk_level,
        "probabilities": {
            "low": round(float(proba[0]), 3),
            "medium": round(float(proba[1]), 3),
            "high": round(float(proba[2]), 3),
        }
    }


if __name__ == "__main__":
    train_model()
    result = predict_risk(60.0, 3, 5.0, 6.0, 55, "diabetes")
    print("Test prediction:", result)
