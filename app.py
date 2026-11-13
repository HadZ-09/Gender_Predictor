
# app.py

import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision import models, transforms
from io import BytesIO

# ---------------- CONFIGURATION ----------------

MODEL_PATH = "./weights/gender_resnet18.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSES = {
    0: "Male",
    1: "Female"
}

# ---------------- FASTAPI ----------------

app = FastAPI(
    title="Gender Classification API",
    description="ResNet18-based facial image classification API",
    version="1.0"
)

# ---------------- TRANSFORM ----------------

transform = transforms.Compose([
    transforms.Resize((144, 144)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---------------- LOAD MODEL ----------------

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

# Supports both the new checkpoint and your
# original notebook's plain state_dict.

if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()

# ---------------- OPENCV FACE DETECTOR ----------------

cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

def detect_face(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return image, False

    # Select the largest detected face.

    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    # Add some padding around the face.

    pad = int(0.15 * max(w, h))

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)

    x2 = min(image.shape[1], x + w + pad)
    y2 = min(image.shape[0], y + h + pad)

    face = image[y1:y2, x1:x2]

    return face, True

# ---------------- PREDICTION FUNCTION ----------------

def predict_image(image):

    # OpenCV BGR -> RGB

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(image)

    input_tensor = transform(
        pil_image
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        confidence, prediction = torch.max(
            probabilities,
            dim=0
        )

    class_id = prediction.item()

    return {
        "prediction": CLASSES[class_id],
        "class_id": class_id,
        "confidence": round(
            confidence.item() * 100, 2
        ),
        "probabilities": {
            "Male": round(
                probabilities[0].item() * 100, 2
            ),
            "Female": round(
                probabilities[1].item() * 100, 2
            )
        }
    }

# ---------------- ROUTES ----------------

@app.get("/")
def home():

    return {
        "message": "Gender Classification API is running",
        "device": str(DEVICE)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPEG, PNG or WEBP image."
        )

    contents = await file.read()

    if len(contents) == 0 or len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image is empty or exceeds 10 MB."
        )

    # Decode using OpenCV.

    image_array = np.frombuffer(
        contents,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image."
        )

    if image.shape[0] < 40 or image.shape[1] < 40:
        raise HTTPException(
            status_code=400,
            detail="Image resolution is too small."
        )

    # Face detection.

    face, face_detected = detect_face(image)

    result = predict_image(face)

    result["face_detected"] = face_detected

    return result