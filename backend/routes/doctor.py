from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import Optional, List
import random
import string

from database import get_db
import models
from auth import (
    verify_password, get_password_hash,
    create_access_token, get_current_doctor
)

router = APIRouter(prefix="/doctor", tags=["doctor"])


# ── Schemas ──────────────────────────────────────────────────────────────────

class DoctorRegister(BaseModel):
    name: str
    email: str
    password: str
    specialization: Optional[str] = None


class PatientCreate(BaseModel):
    name: str
    age: int
    gender: Optional[str] = None
    disease_type: str
    contact_number: Optional[str] = None
    password: str


class PatientOut(BaseModel):
    id: int
    name: str
    age: int
    gender: Optional[str]
    disease_type: str
    contact_number: Optional[str]
    login_code: str

    class Config:
        from_attributes = True


class DoctorOut(BaseModel):
    id: int
    name: str
    email: str
    specialization: Optional[str]

    class Config:
        from_attributes = True


# ── Helpers ───────────────────────────────────────────────────────────────────

def generate_login_code(db: Session, length: int = 8) -> str:
    while True:
        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
        existing = db.query(models.Patient).filter(models.Patient.login_code == code).first()
        if not existing:
            return code


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/register", response_model=DoctorOut)
def register_doctor(data: DoctorRegister, db: Session = Depends(get_db)):
    existing = db.query(models.Doctor).filter(models.Doctor.email == data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    doctor = models.Doctor(
        name=data.name,
        email=data.email,
        specialization=data.specialization,
        hashed_password=get_password_hash(data.password)
    )
    db.add(doctor)
    db.commit()
    db.refresh(doctor)
    return doctor


@router.post("/login")
def login_doctor(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    doctor = db.query(models.Doctor).filter(models.Doctor.email == form_data.username).first()
    if not doctor or not verify_password(form_data.password, doctor.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"doctor_id": doctor.id, "name": doctor.name})
    return {
        "access_token": token,
        "token_type": "bearer",
        "doctor_id": doctor.id,
        "name": doctor.name
    }


@router.post("/patients", response_model=PatientOut)
def register_patient(
    data: PatientCreate,
    db: Session = Depends(get_db),
    doctor: models.Doctor = Depends(get_current_doctor)
):
    login_code = generate_login_code(db)
    patient = models.Patient(
        doctor_id=doctor.id,
        name=data.name,
        age=data.age,
        gender=data.gender,
        disease_type=data.disease_type,
        contact_number=data.contact_number,
        login_code=login_code,
        hashed_password=get_password_hash(data.password)
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


@router.get("/patients", response_model=List[PatientOut])
def get_my_patients(
    db: Session = Depends(get_db),
    doctor: models.Doctor = Depends(get_current_doctor)
):
    return db.query(models.Patient).filter(models.Patient.doctor_id == doctor.id).all()


@router.get("/patients/{patient_id}", response_model=PatientOut)
def get_patient(
    patient_id: int,
    db: Session = Depends(get_db),
    doctor: models.Doctor = Depends(get_current_doctor)
):
    patient = db.query(models.Patient).filter(
        models.Patient.id == patient_id,
        models.Patient.doctor_id == doctor.id
    ).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.get("/patients/{patient_id}/readings")
def get_patient_readings(
    patient_id: int,
    db: Session = Depends(get_db),
    doctor: models.Doctor = Depends(get_current_doctor)
):
    patient = db.query(models.Patient).filter(
        models.Patient.id == patient_id,
        models.Patient.doctor_id == doctor.id
    ).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    readings = db.query(models.Reading).filter(
        models.Reading.patient_id == patient_id
    ).order_by(models.Reading.timestamp.desc()).limit(30).all()

    return readings


@router.get("/patients/{patient_id}/alerts")
def get_patient_alerts(
    patient_id: int,
    db: Session = Depends(get_db),
    doctor: models.Doctor = Depends(get_current_doctor)
):
    patient = db.query(models.Patient).filter(
        models.Patient.id == patient_id,
        models.Patient.doctor_id == doctor.id
    ).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    alerts = db.query(models.Reading).filter(
        models.Reading.patient_id == patient_id,
        models.Reading.alert_triggered == True
    ).order_by(models.Reading.timestamp.desc()).all()

    return alerts


@router.get("/alerts")
def get_all_alerts(
    db: Session = Depends(get_db),
    doctor: models.Doctor = Depends(get_current_doctor)
):
    """Get all alerts across all patients of this doctor."""
    patient_ids = [p.id for p in db.query(models.Patient).filter(
        models.Patient.doctor_id == doctor.id
    ).all()]

    if not patient_ids:
        return []

    alerts = db.query(models.Reading).filter(
        models.Reading.patient_id.in_(patient_ids),
        models.Reading.alert_triggered == True
    ).order_by(models.Reading.timestamp.desc()).limit(50).all()

    patient_map = {p.id: p.name for p in db.query(models.Patient).filter(
        models.Patient.id.in_(patient_ids)
    ).all()}

    result = []
    for alert in alerts:
        result.append({
            "patient_id": alert.patient_id,
            "patient_name": patient_map.get(alert.patient_id, "Unknown"),
            "timestamp": alert.timestamp,
            "alert_message": alert.alert_message,
            "reading_id": alert.id
        })
    return result


@router.get("/me", response_model=DoctorOut)
def get_me(doctor: models.Doctor = Depends(get_current_doctor)):
    return doctor
