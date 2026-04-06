from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base
from crypto import EncryptedFloat  # transparent AES-128 encryption for vital-sign columns


class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    specialization = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    patients = relationship("Patient", back_populates="doctor")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=True)
    disease_type = Column(String, nullable=False)
    contact_number = Column(String, nullable=True)
    login_code = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    doctor = relationship("Doctor", back_populates="patients")
    prescriptions = relationship("Prescription", back_populates="patient")
    readings = relationship("Reading", back_populates="patient")
    risk_results = relationship("RiskResult", back_populates="patient")
    adherence_logs = relationship("AdherenceLog", back_populates="patient")


class Prescription(Base):
    __tablename__ = "prescriptions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    medicine_name = Column(String, nullable=False)
    dosage = Column(String, nullable=False)
    frequency = Column(String, nullable=False)        # e.g. "twice daily"
    timing = Column(Text, nullable=False)             # JSON string: ["08:00", "20:00"]
    instructions = Column(Text, nullable=True)
    start_date = Column(String, nullable=False)
    end_date = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    patient = relationship("Patient", back_populates="prescriptions")
    adherence_logs = relationship("AdherenceLog", back_populates="prescription")


class Reading(Base):
    __tablename__ = "readings"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    blood_sugar = Column(EncryptedFloat, nullable=True)   # mg/dL   — AES-encrypted at rest
    systolic_bp = Column(EncryptedFloat, nullable=True)   # mmHg    — AES-encrypted at rest
    diastolic_bp = Column(EncryptedFloat, nullable=True)  # mmHg    — AES-encrypted at rest
    heart_rate = Column(EncryptedFloat, nullable=True)    # bpm     — AES-encrypted at rest
    temperature = Column(EncryptedFloat, nullable=True)   # Celsius — AES-encrypted at rest
    spo2 = Column(EncryptedFloat, nullable=True)          # %       — AES-encrypted at rest

    notes = Column(Text, nullable=True)
    alert_triggered = Column(Boolean, default=False)
    alert_message = Column(Text, nullable=True)
    alert_solved = Column(Boolean, default=False)
    solved_at = Column(DateTime(timezone=True), nullable=True)

    patient = relationship("Patient", back_populates="readings")


class RiskResult(Base):
    __tablename__ = "risk_results"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    risk_score = Column(Float, nullable=False)        # 0.0 to 1.0
    risk_level = Column(String, nullable=False)       # Low / Medium / High
    features_used = Column(Text, nullable=True)       # JSON of input features

    patient = relationship("Patient", back_populates="risk_results")


class AdherenceLog(Base):
    __tablename__ = "adherence_logs"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    prescription_id = Column(Integer, ForeignKey("prescriptions.id"), nullable=False)
    date = Column(String, nullable=False)             # YYYY-MM-DD
    scheduled_time = Column(String, nullable=False)   # HH:MM
    status = Column(String, default="pending")        # pending / taken / missed / late
    logged_at = Column(DateTime(timezone=True), nullable=True)

    patient = relationship("Patient", back_populates="adherence_logs")
    prescription = relationship("Prescription", back_populates="adherence_logs")
