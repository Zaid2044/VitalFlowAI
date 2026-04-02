import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_ai_suggestions(
    patient_name: str,
    disease_type: str,
    age: int,
    risk_level: str,
    latest_readings: dict,
) -> str:
    readings_text = []
    if latest_readings.get("blood_sugar") is not None:
        readings_text.append(f"Blood Sugar: {latest_readings['blood_sugar']} mg/dL")
    if latest_readings.get("systolic_bp") is not None:
        readings_text.append(f"Blood Pressure: {latest_readings['systolic_bp']}/{latest_readings.get('diastolic_bp', '?')} mmHg")
    if latest_readings.get("heart_rate") is not None:
        readings_text.append(f"Heart Rate: {latest_readings['heart_rate']} bpm")
    if latest_readings.get("temperature") is not None:
        readings_text.append(f"Temperature: {latest_readings['temperature']}°C")
    if latest_readings.get("spo2") is not None:
        readings_text.append(f"SpO2: {latest_readings['spo2']}%")

    readings_summary = "\n".join(readings_text) if readings_text else "No recent readings available."

    prompt = f"""You are a medical health assistant giving personalized recovery advice.

Patient Profile:
- Name: {patient_name}
- Age: {age}
- Condition: {disease_type}
- Current Risk Level: {risk_level}

Latest Vitals:
{readings_summary}

Please provide:
1. **Food Recommendations** (5-6 specific foods that help with their condition, and 3-4 foods to avoid)
2. **Lifestyle Tips** (3-4 practical daily habits for recovery)
3. **Warning Signs** (2-3 symptoms they should watch out for given their condition and current readings)

Keep it concise, friendly, and practical. Do not provide a diagnosis. Format clearly with the three sections."""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )
    return response.text


if __name__ == "__main__":
    result = get_ai_suggestions(
        patient_name="Test Patient",
        disease_type="diabetes",
        age=55,
        risk_level="Medium",
        latest_readings={"blood_sugar": 160, "systolic_bp": 135, "diastolic_bp": 85}
    )
    print(result)