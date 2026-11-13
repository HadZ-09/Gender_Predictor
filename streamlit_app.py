
# streamlit_app.py

import streamlit as st
import requests

from PIL import Image
from io import BytesIO

# ---------------- CONFIGURATION ----------------

API_URL = "http://127.0.0.1:8000/predict"

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Gender Classification",
    page_icon="👤",
    layout="centered"
)

# ---------------- TITLE ----------------

st.title("Gender Classification")
st.caption(
    "Computer Vision | PyTorch | ResNet18 | FastAPI"
)

st.divider()

# ---------------- INPUT ----------------

st.subheader("Choose an input method")

input_method = st.radio(
    "Select input",
    ["Upload Image", "Webcam"],
    horizontal=True
)

image_file = None

if input_method == "Upload Image":

    image_file = st.file_uploader(
        "Upload a facial image",
        type=["jpg", "jpeg", "png", "webp"]
    )

else:

    image_file = st.camera_input(
        "Take a photograph"
    )

# ---------------- PREDICTION ----------------

if image_file is not None:

    image_bytes = image_file.getvalue()

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    st.image(
        image,
        caption="Input Image",
        use_container_width=True
    )

    if st.button(
        "Predict",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Analyzing image..."):

            try:

                files = {
                    "file": (
                        "image.jpg",
                        image_bytes,
                        "image/jpeg"
                    )
                }

                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=60
                )

                response.raise_for_status()

                result = response.json()

                # ---------------- RESULTS ----------------

                st.divider()

                st.subheader("Prediction Results")

                prediction = result["prediction"]
                confidence = result["confidence"]

                st.success(
                    f"Predicted class: {prediction}"
                )

                st.metric(
                    "Model confidence",
                    f"{confidence:.2f}%"
                )

                st.progress(
                    confidence / 100
                )

                st.write("### Class probabilities")

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Male",
                        f"{result['probabilities']['Male']:.2f}%"
                    )

                with col2:

                    st.metric(
                        "Female",
                        f"{result['probabilities']['Female']:.2f}%"
                    )

                if result["face_detected"]:

                    st.info(
                        "A face was detected and cropped using OpenCV."
                    )

                else:

                    st.warning(
                        "No face was detected. "
                        "The model classified the full image instead."
                    )

                st.caption(
                    "These probabilities represent model outputs, "
                    "not certainty about a person's gender identity."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to FastAPI. "
                    "Make sure the backend is running."
                )

            except requests.exceptions.RequestException as e:

                st.error(f"API error: {e}")

            except Exception as e:

                st.error(f"Unexpected error: {e}")

st.divider()

st.caption(
    "Powered by PyTorch, ResNet18, OpenCV, FastAPI and Streamlit."
)