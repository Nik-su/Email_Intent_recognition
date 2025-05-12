# # app.py

# from fastapi import FastAPI
# from pydantic import BaseModel
# import subprocess

# app = FastAPI()

# class EmailText(BaseModel):
#     email_text: str

# @app.post("/predict")
# def predict_intent(data: EmailText):
#     try:
#         # Call Inference_test.py with the email text
#         result = subprocess.run(
#             ["python3", "Inference_test.py", data.email_text],
#             capture_output=True,
#             text=True,
#             check=True
#         )

#         output = result.stdout.strip()
#         predicted_intent, confidence = output.split("|")

#         return {
#             "email_text": data.email_text,
#             "predicted_intent": predicted_intent,
#             "confidence": float(confidence)
#         }

#     except subprocess.CalledProcessError as e:
#         return {"error": "Prediction script failed", "details": e.stderr}
#     except Exception as e:
#         return {"error": str(e)}

# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import nest_asyncio
from pyngrok import ngrok
import uvicorn

# Apply nest_asyncio to allow uvicorn.run inside notebooks or scripts
nest_asyncio.apply()

# Create the FastAPI app
app = FastAPI()

# Define input model
class EmailText(BaseModel):
    email_text: str

# Define the /predict endpoint
@app.post("/predict")
def predict_intent(data: EmailText):
    try:
        # Call Inference_test.py with the email text as argument
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

# Expose with ngrok
if __name__ == "__main__":
    # OPTIONAL: Add your ngrok authtoken (only needed once per machine)
    ngrok.set_auth_token("2ljGDXBe5aMRPGsbmCMZCkBXAoB_43ekyReP1bEYvLmLgfcPE")

    # Start ngrok tunnel on port 8000
    public_url = ngrok.connect(8000)
    print("🚀 Your public URL:", public_url)

    # Run the FastAPI app
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
