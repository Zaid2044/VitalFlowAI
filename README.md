<div align="center">

# 🫀 VitalFlow AI

### Smart Patient Recovery Monitoring System

*From reactive care to proactive, AI-driven health intelligence*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![License](https://img.shields.io/badge/License-MIT-10B981?style=for-the-badge)](LICENSE)

---

</div>

## Overview

**VitalFlow AI** is a full-stack healthcare monitoring platform built for post-operative and chronic disease recovery. It bridges the gap between in-hospital care and at-home recovery by giving doctors real-time visibility into patient vitals and giving patients intelligent, AI-generated health guidance — all through a lightweight web interface with no app install required.

> Built as a Final Year Project in Computer Science — demonstrating production-grade backend architecture, machine learning, and clinical AI integration.

---

## ✨ Features at a Glance

| Category | Capability |
|---|---|
| **Monitoring** | Real-time vital sign submission (Blood Sugar, BP, HR, SpO₂, Temp) |
| **Alerts** | Automatic threshold detection with instant WebSocket push to doctor |
| **AI Suggestions** | LLaMA 3.3-70B powered health advice grounded in medical guidelines (RAG) |
| **Risk Scoring** | LSTM time-series model predicting 24-hour health event probability |
| **Adherence** | Medication tracking — Mark Taken, auto-missed after 3 hours |
| **Security** | JWT auth, field-level AES encryption, prompt injection sanitisation |
| **Doctor Portal** | Live alert dashboard, patient management, alert history with Solved tracking |
| **Patient Portal** | Personal dashboard, prescription schedule, risk score, AI suggestions |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        VitalFlow AI                             │
│                                                                  │
│  ┌─────────────────┐          ┌────────────────────────────┐   │
│  │  Doctor Portal  │          │      Patient Portal         │   │
│  │  (HTML/JS)      │          │      (HTML/JS)              │   │
│  │                 │          │                              │   │
│  │ • Patient Mgmt  │          │ • Log Vitals                │   │
│  │ • Live Alerts   │◄── WS ──►│ • Prescriptions             │   │
│  │ • Alert History │          │ • Risk Score                │   │
│  │ • Risk View     │          │ • AI Suggestions            │   │
│  └────────┬────────┘          └─────────────┬──────────────┘   │
│           │                                  │                   │
│           └─────────────┬────────────────────┘                  │
│                         │ REST API + WebSocket                   │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   FastAPI Backend                         │   │
│  │                                                           │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │   │
│  │  │ /doctor  │  │ /patient │  │ /prescr  │  │  /ws   │  │   │
│  │  │  routes  │  │  routes  │  │  routes  │  │ doctor │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └────────┘  │   │
│  │                                                           │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐  │   │
│  │  │ JWT Auth   │  │  Threshold   │  │  WebSocket      │  │   │
│  │  │ (HS256)    │  │  Alerts      │  │  Manager        │  │   │
│  │  └────────────┘  └──────────────┘  └─────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                         │                                        │
│          ┌──────────────┼──────────────────┐                    │
│          ▼              ▼                  ▼                    │
│  ┌─────────────┐ ┌─────────────┐  ┌──────────────────────┐    │
│  │  SQLite DB  │ │  ML Models  │  │   AI Layer           │    │
│  │  (Fernet    │ │             │  │                       │    │
│  │  encrypted) │ │ • LSTM      │  │ • Groq LLaMA 3.3-70B │    │
│  │             │ │   (PyTorch) │  │ • FAISS Vector Search │    │
│  │  Patients   │ │ • Logistic  │  │ • Medical Guidelines  │    │
│  │  Readings   │ │   Regression│  │   Knowledge Base (RAG)│    │
│  │  Adherence  │ │ • Risk Score│  └──────────────────────┘    │
│  │  Risk Data  │ └─────────────┘                               │
│  └─────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Backend
- **[FastAPI](https://fastapi.tiangolo.com)** — async REST API + WebSocket server
- **[SQLAlchemy](https://sqlalchemy.org)** — ORM with SQLite
- **[python-jose](https://github.com/mpdavis/python-jose)** — JWT authentication
- **[bcrypt](https://github.com/pyca/bcrypt)** — password hashing
- **[cryptography](https://cryptography.io)** — Fernet field-level encryption

### Machine Learning
- **[PyTorch](https://pytorch.org)** — LSTM time-series risk model
- **[scikit-learn](https://scikit-learn.org)** — Logistic Regression baseline + StandardScaler
- **[FAISS](https://github.com/facebookresearch/faiss)** — vector similarity search for RAG

### AI / NLP
- **[Groq](https://groq.com)** — LLaMA 3.3-70B inference (free tier, 14,400 RPD)
- **[sentence-transformers](https://sbert.net)** — `all-MiniLM-L6-v2` embeddings for RAG

### Frontend
- Vanilla HTML / CSS / JavaScript — no framework, no build step
- DM Serif Display + DM Sans typography
- Glassmorphism dark navy design system

---

## Project Structure

```
VitalFlow/
├── backend/
│   ├── main.py                     # FastAPI app, CORS, WebSocket endpoint
│   ├── models.py                   # SQLAlchemy ORM models
│   ├── database.py                 # Engine & session setup
│   ├── auth.py                     # JWT encode/decode, password hashing
│   ├── alerts.py                   # Vital sign threshold definitions
│   ├── ai_suggestions.py           # Groq LLaMA health suggestions
│   ├── rag_suggestions.py          # RAG pipeline (FAISS + sentence-transformers)
│   ├── ws_manager.py               # WebSocket connection manager
│   ├── crypto.py                   # Fernet field-level encryption
│   ├── routes/
│   │   ├── doctor.py               # Doctor auth, patient CRUD, alerts
│   │   ├── patient.py              # Patient auth, readings, adherence, risk
│   │   └── prescriptions.py        # Prescription management
│   ├── ml/
│   │   ├── lstm_risk_model.py      # LSTM model — definition, training, inference
│   │   └── risk_model.py           # Logistic Regression fallback model
│   ├── knowledge_base/
│   │   └── medical_guidelines.txt  # Clinical guidelines for RAG (ADA, WHO, ACC/AHA)
│   └── scripts/
│       └── migrate_encrypt.py      # One-time field encryption migration
│
├── doctor-portal/
│   ├── index.html                  # Doctor login & registration
│   ├── dashboard.html              # Patient list, live alerts, alert history
│   └── patient.html                # Individual patient detail view
│
├── patient-portal/
│   ├── index.html                  # Patient login
│   └── dashboard.html              # Vitals, prescriptions, risk score, AI suggestions
│
├── requirements_v2.txt             # Full dependency list
├── run.bat                         # One-click Windows launcher
└── .gitignore
```

---

## Getting Started

### Prerequisites
- Python **3.10** or higher
- A free [Groq API key](https://console.groq.com) — no credit card required

### 1. Clone & set up environment

```bash
git clone https://github.com/your-username/VitalFlow.git
cd VitalFlow

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements_v2.txt
```

### 2. Configure environment variables

Create `backend/.env`:

```env
SECRET_KEY=your-random-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

GROQ_API_KEY=gsk_your_groq_key_here

ALLOWED_ORIGINS=http://127.0.0.1:5500,http://localhost:5500,null
```

> Generate a strong SECRET_KEY:
> ```bash
> python -c "import secrets; print(secrets.token_hex(32))"
> ```

### 3. Start the backend

```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

| URL | Description |
|---|---|
| `http://localhost:8000` | API root |
| `http://localhost:8000/docs` | Interactive Swagger UI |
| `http://localhost:8000/health` | Health check |

### 4. Open the portals

Serve the frontend with [VS Code Live Server](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) or any static file server:

| Portal | File |
|---|---|
| Doctor | `doctor-portal/index.html` |
| Patient | `patient-portal/index.html` |

> **Windows quick start:** double-click `run.bat` — sets up the virtualenv, installs dependencies, and starts the server automatically.

---

## Security Architecture

### Authentication
- Separate JWT token flows for doctors and patients (HS256)
- Tokens expire after 24 hours (configurable via `ACCESS_TOKEN_EXPIRE_MINUTES`)
- All sensitive routes protected by `get_current_doctor` / `get_current_patient` FastAPI dependencies
- Ownership validation — doctors can only access patients assigned to them

### Data Privacy
- **Field-Level Encryption** — vital sign values are encrypted with Fernet (AES-128-CBC + HMAC-SHA256) before writing to SQLite. Raw health data is unreadable even if the database file is stolen.
- **Prompt Injection Protection** — all patient-supplied strings are sanitised (strip `{}`, newlines, truncate) before being inserted into AI prompts.

### Environment Variables
All secrets (`SECRET_KEY`, `GROQ_API_KEY`, `FIELD_ENCRYPTION_KEY`) are loaded exclusively from `.env`. The application raises `RuntimeError` at startup if any required secret is missing — no insecure defaults.

---

## ML & AI Details

### LSTM Risk Model

A 2-layer LSTM network (PyTorch) processes the **last 10 sequential readings** as a time series to predict the probability of a health alert event in the next 24 hours.

```
Input  →  (batch, 10, 6)   — 10 readings × 6 vital features
LSTM   →  hidden=64, layers=2, dropout=0.2
Output →  sigmoid scalar ∈ [0, 1]   — P(alert in next 24h)
```

A **trend detector** compares the first half vs second half of the input window for blood sugar and systolic BP, labelling the direction as `Improving`, `Stable`, or `Worsening`.

Training happens automatically on first startup using real patient data if available, or synthetic data as a fallback.

### RAG-Grounded AI Suggestions

```
Patient context
      │
      ▼
Embed query  ──►  all-MiniLM-L6-v2  ──►  384-dim vector
      │
      ▼
FAISS cosine search  ──►  Top-3 relevant clinical guideline chunks
      │
      ▼
Inject into Groq prompt  ──►  LLaMA 3.3-70B generates grounded response
```

The knowledge base covers: glucose targets (ADA), DASH diet, BP classification (ACC/AHA), SpO₂ thresholds, fever protocols, COPD management, medication adherence impact, and risk stratification principles.

### Real-Time Alert Flow

```
Patient submits reading
        │
        ▼
Threshold check  →  alert triggered?
        │ yes
        ▼
ws_manager.notify_doctor_sync(doctor_id, payload)
        │
        ▼
JSON pushed to all open doctor WebSocket connections  (<10ms)
```

---

## API Reference

### Authentication

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/doctor/register` | Register a new doctor |
| `POST` | `/doctor/login` | Doctor login → JWT |
| `POST` | `/patient/login` | Patient login → JWT |

### Doctor Routes *(Bearer token required)*

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/doctor/me` | Current doctor profile |
| `GET` | `/doctor/patients` | List all patients |
| `POST` | `/doctor/patients` | Register a new patient |
| `GET` | `/doctor/alerts` | Live (unsolved) alerts |
| `GET` | `/doctor/alerts/history` | All alerts including solved |
| `POST` | `/doctor/alerts/{id}/solve` | Mark alert as solved |
| `WS` | `/ws/doctor/{id}?token=...` | Real-time alert stream |

### Patient Routes *(Bearer token required)*

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/patient/me` | Current patient profile |
| `POST` | `/patient/readings` | Submit vital signs |
| `GET` | `/patient/readings` | Last 30 readings |
| `GET` | `/patient/prescriptions` | Active prescriptions |
| `POST` | `/patient/adherence` | Log dose taken / missed / late |
| `GET` | `/patient/adherence/today` | Today's adherence log |
| `POST` | `/patient/calculate-risk` | Run LSTM risk model |
| `GET` | `/patient/suggestions` | AI health suggestions |

---

## Database Schema

```
doctors               patients              readings
───────               ────────              ────────
id (PK)               id (PK)               id (PK)
name                  doctor_id (FK)        patient_id (FK)
email (unique)        name                  timestamp
specialization        age                   blood_sugar *
hashed_password       disease_type          systolic_bp *
created_at            login_code (unique)   diastolic_bp *
                      hashed_password       heart_rate *
                      created_at            temperature *
                                            spo2 *
prescriptions         adherence_logs        alert_triggered
─────────────         ──────────────        alert_message
id (PK)               id (PK)               alert_solved
patient_id (FK)       patient_id (FK)       solved_at
medicine_name         prescription_id (FK)
dosage                date                  risk_results
frequency             scheduled_time        ────────────
timing (JSON)         status                id (PK)
is_active             logged_at             patient_id (FK)
start_date                                  risk_score
end_date                                    risk_level
                                            features_used (JSON)
                                            timestamp

  * = Fernet-encrypted at rest when FIELD_ENCRYPTION_KEY is set
```

---

## Roadmap

- [ ] Mobile App — React Native patient app with push notifications
- [ ] FHIR Integration — HL7 FHIR R4 export for EHR compatibility
- [ ] Multi-language Support — Urdu / Arabic localisation for regional deployment
- [ ] Wearable Integration — Bluetooth vital sign ingestion from pulse oximeters and BP cuffs
- [ ] Federated Learning — Train risk models across hospitals without sharing raw patient data
- [ ] PDF Report Generation — Weekly patient health summary for doctor review

---

## Author

**Mohammed Zaid Ahmed**
Final Year Computer Science Project · 2025–2026

---

<div align="center">

*Built with care for better patient outcomes.*

**VitalFlow AI** — Monitoring Recovery, Intelligently.

</div>
