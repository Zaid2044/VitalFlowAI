from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
import json
import logging
from datetime import date, datetime, timezone

from database import get_db
import models
from auth import verify_password, create_access_token, get_current_patient
from alerts import check_thresholds
from ai_suggestions import get_ai_suggestions
from ml.risk_model import predict_risk

router = APIRouter(prefix="/patient", tags=["patient"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class ReadingCreate(BaseModel):
    blood_sugar: Optional[float] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    heart_rate: Optional[float] = None
    temperature: Optional[float] = None
    spo2: Optional[float] = None
    notes: Optional[str] = None


class AdherenceUpdate(BaseModel):
    prescription_id: int
    date: str           # YYYY-MM-DD
    scheduled_time: str # HH:MM
    status: str         # taken / missed / late


class RiskCalculate(BaseModel):
    adherence_7day: float
    missed_streak: int
    avg_energy: float
    symptom_score: float


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/login")
def login_patient(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # username = login_code, password = patient password
    patient = db.query(models.Patient).filter(
        models.Patient.login_code == form_data.username
    ).first()
    if not patient or not verify_password(form_data.password, patient.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid login code or password")

    token = create_access_token({"patient_id": patient.id, "name": patient.name})
    return {
        "access_token": token,
        "token_type": "bearer",
        "patient_id": patient.id,
        "name": patient.name,
        "login_code": patient.login_code
    }


@router.get("/me")
def get_me(patient: models.Patient = Depends(get_current_patient)):
    return {
        "id": patient.id,
        "name": patient.name,
        "age": patient.age,
        "gender": patient.gender,
        "disease_type": patient.disease_type,
        "contact_number": patient.contact_number,
        "login_code": patient.login_code,
        "doctor_id": patient.doctor_id,
    }


@router.get("/prescriptions")
def get_prescriptions(
    patient: models.Patient = Depends(get_current_patient),
    db: Session = Depends(get_db)
):
    prescriptions = db.query(models.Prescription).filter(
        models.Prescription.patient_id == patient.id,
        models.Prescription.is_active == True
    ).all()

    result = []
    for p in prescriptions:
        result.append({
            "id": p.id,
            "medicine_name": p.medicine_name,
            "dosage": p.dosage,
            "frequency": p.frequency,
            "timing": json.loads(p.timing) if p.timing else [],
            "instructions": p.instructions,
            "start_date": p.start_date,
            "end_date": p.end_date,
        })
    return result


@router.post("/readings")
def submit_reading(
    data: ReadingCreate,
    patient: models.Patient = Depends(get_current_patient),
    db: Session = Depends(get_db)
):
    alert_triggered, alert_message = check_thresholds(
        blood_sugar=data.blood_sugar,
        systolic_bp=data.systolic_bp,
        diastolic_bp=data.diastolic_bp,
        heart_rate=data.heart_rate,
        temperature=data.temperature,
        spo2=data.spo2
    )

    reading = models.Reading(
        patient_id=patient.id,
        blood_sugar=data.blood_sugar,
        systolic_bp=data.systolic_bp,
        diastolic_bp=data.diastolic_bp,
        heart_rate=data.heart_rate,
        temperature=data.temperature,
        spo2=data.spo2,
        notes=data.notes,
        alert_triggered=alert_triggered,
        alert_message=alert_message if alert_triggered else None
    )
    db.add(reading)
    db.commit()
    db.refresh(reading)

    return {
        "id": reading.id,
        "timestamp": reading.timestamp,
        "alert_triggered": alert_triggered,
        "alert_message": alert_message
    }


@router.get("/readings")
def get_readings(
    patient: models.Patient = Depends(get_current_patient),
    db: Session = Depends(get_db)
):
    readings = db.query(models.Reading).filter(
        models.Reading.patient_id == patient.id
    ).order_by(models.Reading.timestamp.desc()).limit(30).all()
    return readings


@router.get("/risk")
def get_risk(
    patient: models.Patient = Depends(get_current_patient),
    db: Session = Depends(get_db)
):
    latest_risk = db.query(models.RiskResult).filter(
        models.RiskResult.patient_id == patient.id
    ).order_by(models.RiskResult.timestamp.desc()).first()

    all_risks = db.query(models.RiskResult).filter(
        models.RiskResult.patient_id == patient.id
    ).order_by(models.RiskResult.timestamp.desc()).limit(10).all()

    return {
        "latest": {
            "risk_score": latest_risk.risk_score if latest_risk else None,
            "risk_level": latest_risk.risk_level if latest_risk else "Unknown",
            "timestamp": latest_risk.timestamp if latest_risk else None,
        },
        "history": [
            {"risk_score": r.risk_score, "risk_level": r.risk_level, "timestamp": r.timestamp}
            for r in all_risks
        ]
    }


@router.post("/adherence")
def log_adherence(
    data: AdherenceUpdate,
    patient: models.Patient = Depends(get_current_patient),
    db: Session = Depends(get_db)
):
    existing = db.query(models.AdherenceLog).filter(
        models.AdherenceLog.patient_id == patient.id,
        models.AdherenceLog.prescription_id == data.prescription_id,
        models.AdherenceLog.date == data.date,
        models.AdherenceLog.scheduled_time == data.scheduled_time
    ).first()

    if existing:
        existing.status = data.status
        existing.logged_at = datetime.now(timezone.utc)
        db.commit()
        return {"message": "Updated", "id": existing.id}

    log = models.AdherenceLog(
        patient_id=patient.id,
        prescription_id=data.prescription_id,
        date=data.date,
        scheduled_time=data.scheduled_time,
        status=data.status,
        logged_at=datetime.now(timezone.utc)
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return {"message": "Logged", "id": log.id}


@router.post("/calculate-risk")
def calculate_risk(
    data: RiskCalculate,
    patient: models.Patient = Depends(get_current_patient),
    db: Session = Depends(get_db)
):
    result = predict_risk(
        adherence_7day=data.adherence_7day,
        missed_streak=data.missed_streak,
        avg_energy=data.avg_energy,
        symptom_score=data.symptom_score,
        age=patient.age,
        disease_type=patient.disease_type
    )
    risk_entry = models.RiskResult(
        patient_id=patient.id,
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        features_used=json.dumps({
            "adherence_7day": data.adherence_7day,
            "missed_streak": data.missed_streak,
            "avg_energy": data.avg_energy,
            "symptom_score": data.symptom_score
        })
    )
    db.add(risk_entry)
    db.commit()
    return result


@router.get("/adherence/today")
def get_today_adherence(
    patient: models.Patient = Depends(get_current_patient),
    db: Session = Depends(get_db)
):
    today = date.today().isoformat()
    logs = db.query(models.AdherenceLog).filter(
        models.AdherenceLog.patient_id == patient.id,
        models.AdherenceLog.date == today
    ).all()
    return [
        {
            "prescription_id": l.prescription_id,
            "scheduled_time": l.scheduled_time,
            "status": l.status,
            "logged_at": l.logged_at.isoformat() if l.logged_at else None
        }
        for l in logs
    ]


@router.get("/suggestions")
def get_suggestions(
    patient: models.Patient = Depends(get_current_patient),
    db: Session = Depends(get_db)
):
    latest_reading = db.query(models.Reading).filter(
        models.Reading.patient_id == patient.id
    ).order_by(models.Reading.timestamp.desc()).first()

    latest_risk = db.query(models.RiskResult).filter(
        models.RiskResult.patient_id == patient.id
    ).order_by(models.RiskResult.timestamp.desc()).first()

    readings_dict = {}
    if latest_reading:
        readings_dict = {
            "blood_sugar": latest_reading.blood_sugar,
            "systolic_bp": latest_reading.systolic_bp,
            "diastolic_bp": latest_reading.diastolic_bp,
            "heart_rate": latest_reading.heart_rate,
            "temperature": latest_reading.temperature,
            "spo2": latest_reading.spo2,
        }

    risk_level = latest_risk.risk_level if latest_risk else "Unknown"

    try:
        suggestions = get_ai_suggestions(
            patient_name=patient.name,
            disease_type=patient.disease_type,
            age=patient.age,
            risk_level=risk_level,
            latest_readings=readings_dict
        )
        return {"suggestions": suggestions}
    except Exception as e:
        logging.exception("AI suggestions failed for patient %s", patient.id)
        return {"suggestions": "Unable to generate suggestions at this time. Please try again later."}
