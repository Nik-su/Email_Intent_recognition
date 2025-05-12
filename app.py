# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import subprocess

app = FastAPI()

class EmailText(BaseModel):
    email_text: str

@app.post("/predict")
def predict_intent(data: EmailText):
    try:
        # Call Inference_test.py with the email text
        result = subprocess.run(
            ["python3", "Inference_test.py", data.email_text],
            capture_output=True,
            text=True,
            check=True
        )

        output = result.stdout.strip()
        predicted_intent, confidence = output.split("|")

        return {
            "email_text": data.email_text,
            "predicted_intent": predicted_intent,
            "confidence": float(confidence)
        }

    except subprocess.CalledProcessError as e:
        return {"error": "Prediction script failed", "details": e.stderr}
    except Exception as e:
        return {"error": str(e)}
