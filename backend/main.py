from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from database import engine, Base
import models  # noqa: F401 - ensures models are registered

from routes.doctor import router as doctor_router
from routes.patient import router as patient_router
from routes.prescriptions import router as prescription_router

# Create all tables
Base.metadata.create_all(bind=engine)

# Train risk model on startup if not exists
from ml.risk_model import load_model
load_model()

app = FastAPI(
    title="VitalFlow AI",
    description="Smart Patient Recovery Monitoring System",
    version="1.0.0"
)

# CORS - allow both portals to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(doctor_router)
app.include_router(patient_router)
app.include_router(prescription_router)


@app.get("/")
def root():
    return {
        "message": "VitalFlow AI API is running",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
