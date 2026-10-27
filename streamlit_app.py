import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# -------------------- Configuration --------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_CANDIDATES = [
    BASE_DIR / "models" / "asl_mobilenetv2.keras",
    BASE_DIR / "asl_mobilenetv2.keras",
]

LABEL_CANDIDATES = [
    BASE_DIR / "models" / "class_names.json",
    BASE_DIR / "class_names.json",
]

MODEL_PATH = next(
    (p for p in MODEL_CANDIDATES if p.is_file()),
    MODEL_CANDIDATES[0],
)

LABELS_PATH = next(
    (p for p in LABEL_CANDIDATES if p.is_file()),
    LABEL_CANDIDATES[0],
)
IMAGE_SIZE = 160
CONFIDENCE_THRESHOLD = 0.50

st.set_page_config(
    page_title="ASL Alphabet Recognition",
    page_icon="🤟",
    layout="centered",
)

@st.cache_resource
def load_asl_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if isinstance(labels, dict):
        labels = [
            name for name, index in sorted(
                labels.items(), key=lambda item: int(item[1])
            )
        ]
    labels = [str(x) for x in labels]
    if len(labels) != model.output_shape[-1]:
        raise ValueError(
            f"Model has {model.output_shape[-1]} outputs but "
            f"{len(labels)} class labels were found."
        )
    return model, labels

def predict_image(image, model, labels):
    # Match the original app's RGB, 160x160, float32 0-255 preprocessing.
    image = image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    top_indices = np.argsort(probabilities)[-3:][::-1]
    best_index = int(top_indices[0])
    return (
        labels[best_index],
        float(probabilities[best_index]),
        [(labels[int(i)], float(probabilities[int(i)])) for i in top_indices],
    )

st.title("🤟 ASL Alphabet Recognition")
st.write(
    "Recognize **static American Sign Language (ASL) alphabet signs** "
    "using a trained MobileNetV2 model. Allow camera access or upload an image."
)
st.info(
    "For best results, show one clear hand sign, keep the whole hand in frame, "
    "use good lighting, and keep the background simple."
)

try:
    model, class_names = load_asl_model()
except Exception as exc:
    st.error("The model could not be loaded.")
    st.write("Repository directory:", str(BASE_DIR))
    st.write("Model path:", str(MODEL_PATH))
    st.write("Model exists:", MODEL_PATH.is_file())
    st.write("Labels path:", str(LABELS_PATH))
    st.write("Labels exist:", LABELS_PATH.is_file())

    st.write("Root directory contents:")
    st.write([p.name for p in BASE_DIR.iterdir()])

    models_dir = BASE_DIR / "models"
    st.write("Models directory exists:", models_dir.is_dir())

    if models_dir.is_dir():
        st.write("Models directory contents:")
        st.write([p.name for p in models_dir.iterdir()])

    st.exception(exc)
    st.stop()

source = st.radio(
    "Choose image source",
    ["Use camera", "Upload an image"],
    horizontal=True,
)

image = None
if source == "Use camera":
    camera_file = st.camera_input("Capture your ASL hand sign")
    if camera_file is not None:
        image = Image.open(camera_file)
else:
    uploaded_file = st.file_uploader(
        "Upload a hand-sign image", type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

if image is not None:
    st.image(image, caption="Image submitted for prediction", use_container_width=True)
    if st.button("Recognize sign", type="primary", use_container_width=True):
        with st.spinner("Recognizing sign..."):
            try:
                label, confidence, top_predictions = predict_image(
                    image, model, class_names
                )
            except Exception as exc:
                st.error("Prediction failed.")
                st.exception(exc)
                st.stop()

        if confidence >= CONFIDENCE_THRESHOLD:
            st.success(f"Prediction: **{label.upper()}**")
        else:
            st.warning(
                f"Low confidence: top prediction is **{label.upper()}**. "
                "Try another image with clearer lighting and hand positioning."
            )
        st.metric("Confidence", f"{confidence * 100:.1f}%")
        st.subheader("Top 3 predictions")
        for name, score in top_predictions:
            st.write(f"**{name.upper()}** — {score * 100:.1f}%")
            st.progress(min(max(score, 0.0), 1.0))

with st.expander("About this project"):
    st.write(
        "This demo recognizes static alphabet signs only. It is not designed "
        "to interpret dynamic signs, words, or complete sentences. Predictions "
        "can vary with lighting, camera angle, hand position, and differences "
        "between real-world images and the training dataset."
    )
    st.caption("Model: MobileNetV2 · Input size: 160 × 160 · Classes: 29")
