# VitalFlow AI — Setup Guide

## Requirements
- Python 3.10 or higher
- An Anthropic API key (get one at console.anthropic.com)

---

## Step 1: Configure your API key

Open `backend/.env` and replace the placeholder:

```
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

Change it to your real key:

```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxx...
```

Also change the SECRET_KEY to something random for security:
```
SECRET_KEY=any-long-random-string-here-123456
```

---

## Step 2: Run the project (Windows)

Double-click `run.bat` OR open a terminal in this folder and run:

```
run.bat
```

This will:
1. Create a Python virtual environment
2. Install all dependencies
3. Start the FastAPI server on http://localhost:8000

---

## Step 3: Open the portals

- **Doctor Portal:** Open `doctor-portal/index.html` in your browser
- **Patient Portal:** Open `patient-portal/index.html` in your browser

Both portals talk to the API at http://localhost:8000

---

## API Documentation

Once the server is running, visit:
- http://localhost:8000/docs — Interactive API docs (Swagger UI)

---

## Project Structure

```
vitalflow/
├── backend/
│   ├── main.py              ← FastAPI app entry point
│   ├── database.py          ← SQLite connection
│   ├── models.py            ← Database tables
│   ├── auth.py              ← JWT authentication
│   ├── alerts.py            ← Threshold-based alert logic
│   ├── ai_suggestions.py    ← Claude AI suggestions
│   ├── routes/
│   │   ├── doctor.py        ← Doctor API routes
│   │   ├── patient.py       ← Patient API routes
│   │   └── prescriptions.py ← Prescription + risk routes
│   ├── ml/
│   │   └── risk_model.py    ← Logistic Regression risk model
│   └── vitalflow.db         ← SQLite database (auto-created)
├── doctor-portal/           ← Doctor web interface
├── patient-portal/          ← Patient web interface
├── requirements.txt
├── run.bat                  ← Windows startup script
└── README.md
```

---

## Demo Flow

1. Start server with `run.bat`
2. Open doctor portal → Register as a doctor
3. Log in → Register a patient (note the login code shown)
4. Add a prescription for the patient
5. Open patient portal in another tab
6. Log in with the patient's login code + password
7. Submit a reading (try a high blood sugar like 200 to trigger alert)
8. Go back to doctor portal → see the alert
9. Check the patient's risk score and AI suggestions
