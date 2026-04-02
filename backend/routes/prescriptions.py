from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
import json

from database import get_db
import models
from auth import get_current_doctor, get_current_patient
from ml.risk_model import predict_risk

router = APIRouter(prefix="/prescriptions", tags=["prescriptions"])


class PrescriptionCreate(BaseModel):
    patient_id: int
    medicine_name: str
    dosage: str
    frequency: str
    timing: List[str]
    instructions: Optional[str] = None
    start_date: str
    end_date: Optional[str] = None


class RiskCalculate(BaseModel):
    patient_id: int
    adherence_7day: float
    missed_streak: int
    avg_energy: float
    symptom_score: float


@router.post("")
def create_prescription(
    data: PrescriptionCreate,
    db: Session = Depends(get_db),
    doctor: models.Doctor = Depends(get_current_doctor)
):
    patient = db.query(models.Patient).filter(
        models.Patient.id == data.patient_id,
        models.Patient.doctor_id == doctor.id
    ).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    prescription = models.Prescription(
        patient_id=data.patient_id,
        medicine_name=data.medicine_name,
        dosage=data.dosage,
        frequency=data.frequency,
        timing=json.dumps(data.timing),
        instructions=data.instructions,
        start_date=data.start_date,
        end_date=data.end_date
    )
    db.add(prescription)
    db.commit()
    db.refresh(prescription)

    return {
        "id": prescription.id,
        "medicine_name": prescription.medicine_name,
        "dosage": prescription.dosage,
        "timing": data.timing,
        "start_date": prescription.start_date
    }


@router.get("/patient/{patient_id}")
def get_patient_prescriptions(
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

    prescriptions = db.query(models.Prescription).filter(
        models.Prescription.patient_id == patient_id
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
            "is_active": p.is_active,
        })
    return result


@router.delete("/{prescription_id}")
def deactivate_prescription(
    prescription_id: int,
    db: Session = Depends(get_db),
    doctor: models.Doctor = Depends(get_current_doctor)
):
    prescription = db.query(models.Prescription).filter(
        models.Prescription.id == prescription_id
    ).first()
    if not prescription:
        raise HTTPException(status_code=404, detail="Prescription not found")

    patient = db.query(models.Patient).filter(
        models.Patient.id == prescription.patient_id,
        models.Patient.doctor_id == doctor.id
    ).first()
    if not patient:
        raise HTTPException(status_code=403, detail="Not authorized")

    prescription.is_active = False
    db.commit()
    return {"message": "Prescription deactivated"}


@router.post("/calculate-risk")
def calculate_risk(
    data: RiskCalculate,
    db: Session = Depends(get_db),
    doctor: models.Doctor = Depends(get_current_doctor)
):
    patient = db.query(models.Patient).filter(
        models.Patient.id == data.patient_id,
        models.Patient.doctor_id == doctor.id
    ).first()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

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