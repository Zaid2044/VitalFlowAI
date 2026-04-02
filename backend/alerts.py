from typing import Optional, List, Tuple


THRESHOLDS = {
    "blood_sugar": {
        "high": 180.0,
        "low": 70.0,
        "unit": "mg/dL",
        "label": "Blood Sugar"
    },
    "systolic_bp": {
        "high": 140.0,
        "low": 90.0,
        "unit": "mmHg",
        "label": "Systolic BP"
    },
    "diastolic_bp": {
        "high": 90.0,
        "low": 60.0,
        "unit": "mmHg",
        "label": "Diastolic BP"
    },
    "heart_rate": {
        "high": 100.0,
        "low": 50.0,
        "unit": "bpm",
        "label": "Heart Rate"
    },
    "temperature": {
        "high": 38.5,
        "low": 35.0,
        "unit": "°C",
        "label": "Temperature"
    },
    "spo2": {
        "high": None,
        "low": 94.0,
        "unit": "%",
        "label": "SpO2"
    },
}


def check_thresholds(
    blood_sugar: Optional[float] = None,
    systolic_bp: Optional[float] = None,
    diastolic_bp: Optional[float] = None,
    heart_rate: Optional[float] = None,
    temperature: Optional[float] = None,
    spo2: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Returns (alert_triggered: bool, alert_message: str)
    """
    alerts: List[str] = []
    values = {
        "blood_sugar": blood_sugar,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "heart_rate": heart_rate,
        "temperature": temperature,
        "spo2": spo2,
    }

    for key, value in values.items():
        if value is None:
            continue
        threshold = THRESHOLDS[key]
        label = threshold["label"]
        unit = threshold["unit"]

        if threshold["high"] is not None and value > threshold["high"]:
            alerts.append(f"{label} is HIGH: {value} {unit} (threshold: >{threshold['high']})")
        if threshold["low"] is not None and value < threshold["low"]:
            alerts.append(f"{label} is LOW: {value} {unit} (threshold: <{threshold['low']})")

    if alerts:
        return True, " | ".join(alerts)
    return False, ""


def get_thresholds():
    return THRESHOLDS
