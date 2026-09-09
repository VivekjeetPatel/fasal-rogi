import os
import json

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# ── Path helpers ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "trained_model", "plant_disease_model.h5")
JSON_PATH   = os.path.join(BASE_DIR, "class_indices.json")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered",
)

# ── Custom CSS for a premium look ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        min-height: 100vh;
    }

    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #56ab2f, #a8e063);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .hero-sub {
        text-align: center;
        color: #a0b4c0;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
    }

    .result-box {
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        border-left: 4px solid #56ab2f;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        margin-top: 1rem;
        color: #d8f3dc;
        font-size: 1.1rem;
        font-weight: 600;
    }

    .stButton > button {
        background: linear-gradient(90deg, #56ab2f, #a8e063) !important;
        color: #0f2027 !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 0.6rem 1.6rem !important;
        font-size: 1rem !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(86, 171, 47, 0.4) !important;
    }

    .stFileUploader label {
        color: #a8e063 !important;
        font-weight: 600 !important;
    }

    .label-tag {
        display: inline-block;
        background: rgba(86, 171, 47, 0.2);
        color: #a8e063;
        border: 1px solid #56ab2f;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_class_indices():
    if not os.path.exists(JSON_PATH):
        return {}
    with open(JSON_PATH, "r") as f:
        return json.load(f)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🌿 Plant Disease Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Upload a leaf image and let the CNN identify the disease.</div>',
    unsafe_allow_html=True,
)

# ── File uploader ─────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose a JPG or PNG image of a plant leaf",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)
st.markdown("</div>", unsafe_allow_html=True)

# ── Main inference panel ──────────────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<span class="label-tag">PREVIEW</span>', unsafe_allow_html=True)
        st.image(image.resize((150, 150)), use_column_width=False, width=150)

    with col2:
        st.markdown('<span class="label-tag">CLASSIFY</span>', unsafe_allow_html=True)
        st.write("")  # spacing
        classify_btn = st.button("🔬 Classify", key="classify_btn")

    if classify_btn:
        model = load_model()
        if model is None:
            st.error(
                f"⚠️ Model file not found at `{MODEL_PATH}`. "
                "Please run `train_model.ipynb` or place `plant_disease_model.h5` inside `app/trained_model/`."
            )
        else:
            with st.spinner("Analysing image…"):
                # ── Pre-processing ─────────────────────────────────────────────
                img_resized  = image.resize((224, 224))
                img_array    = np.array(img_resized, dtype=np.float32) / 255.0
                img_expanded = np.expand_dims(img_array, axis=0)   # (1, 224, 224, 3)

                # ── Inference ─────────────────────────────────────────────────
                class_indices = load_class_indices()

                predictions   = model.predict(img_expanded)
                pred_index    = int(np.argmax(predictions[0]))
                confidence    = float(np.max(predictions[0])) * 100
                disease_name  = class_indices.get(str(pred_index), f"Class {pred_index}")

            # ── Result display ─────────────────────────────────────────────────
            st.success(f"Prediction: {disease_name.replace('_', ' ')}")

            st.markdown(
                f"""
                <div class="result-box">
                    🌱 &nbsp;<strong>Predicted Disease:</strong>&nbsp;
                    {disease_name.replace("_", " ").title()}
                    &nbsp;|&nbsp; <strong>Confidence:</strong> {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Confidence bar
            st.progress(int(confidence))

        # Top-5 breakdown
        with st.expander("📊 Top-5 Predictions"):
            top5_indices = np.argsort(predictions[0])[::-1][:5]
            for idx in top5_indices:
                label = class_indices.get(str(int(idx)), "Unknown")
                prob  = float(predictions[0][idx]) * 100
                st.write(f"**{label.replace('_', ' ').title()}** — {prob:.2f}%")
                st.progress(int(prob))

else:
    st.info("⬆️  Upload a leaf image above to get started.", icon="🌿")
