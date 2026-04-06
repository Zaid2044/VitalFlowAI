from fastapi import FastAPI, Request, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import os
from dotenv import load_dotenv

load_dotenv()

# Fail fast on missing critical environment variables so issues are caught at startup,
# not silently on the first request that needs them.
if not os.getenv("SECRET_KEY"):
    raise RuntimeError("SECRET_KEY environment variable is not set")
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY environment variable is not set — AI suggestions will not work")
if not os.getenv("FIELD_ENCRYPTION_KEY"):
    raise RuntimeError(
        "FIELD_ENCRYPTION_KEY environment variable is not set — vital-sign encryption will fail.\n"
        "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
    )

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

# CORS - set ALLOWED_ORIGINS in .env as comma-separated list, e.g. "http://localhost:3000,http://localhost:5173"
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def private_network_access(request: Request, call_next):
    """Allow Chrome's Private Network Access preflight (file:// → localhost)."""
    if (request.method == "OPTIONS" and
            "access-control-request-private-network" in request.headers):
        response = PlainTextResponse("OK", status_code=200)
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        response.headers["Access-Control-Allow-Origin"] = request.headers.get("origin", "*")
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response
    return await call_next(request)


# Routers
app.include_router(doctor_router)
app.include_router(patient_router)
app.include_router(prescription_router)


# ── Real-time WebSocket endpoint ───────────────────────────────────────────────

from ws_manager import doctor_ws_endpoint  # noqa: E402


@app.websocket("/ws/doctor/{doctor_id}")
async def ws_doctor(
    websocket: WebSocket,
    doctor_id: int,
    token: str = Query(..., description="JWT access token for authentication"),
):
    """
    Persistent WebSocket connection for real-time alert delivery to doctors.

    Connect from the doctor portal with:
        const ws = new WebSocket(
            `ws://localhost:8000/ws/doctor/${doctorId}?token=${accessToken}`
        );

    Incoming message types from server:
        { "type": "alert",     "patient_name": "...", "alert_message": "..." }
        { "type": "keepalive" }
        { "type": "pong" }

    Send from client to keep connection alive through proxies:
        ws.send(JSON.stringify({ "type": "ping" }))
    """
    await doctor_ws_endpoint(websocket, doctor_id, token)


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
