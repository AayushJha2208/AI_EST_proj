import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# -----------------------------
# Page Config (Theme feel)
# -----------------------------
st.set_page_config(
    page_title="Engine Failure Predictor",
    page_icon="⚙️",
    layout="centered"
)

# Custom CSS (white + aqua theme)
st.markdown("""
    <style>
    .stApp {
        background-color: #f8ffff;
    }
    .main-title {
        color: #00bcd4;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
    }
    .sub-text {
        text-align: center;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚙️ Engine Failure Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Demo using simulated sequence input</div>', unsafe_allow_html=True)

# -----------------------------
# Load Model + Scaler
# -----------------------------
@st.cache_resource
def load_assets():
    model = load_model("binary_model.keras")
    with open("scaler.pkl", "rb") as f:
        data = pickle.load(f)
    return model, data

model, data = load_assets()

scaler = data["scaler"]
cols_normalize = data["cols_normalize"]
sequence_cols = data["sequence_cols"]
sequence_length = data["sequence_length"]

# -----------------------------
# User Inputs (Simplified)
# -----------------------------
st.markdown("### 🔢 Enter Sensor Values")

# For demo: take only a few inputs
s2 = st.slider("Sensor s2", 0.0, 1.0, 0.5)
cycle = st.slider("Cycle", 1, 300, 100)

# -----------------------------
# Prediction Logic
# -----------------------------
if st.button("🚀 Predict"):

    # Create base feature row
    input_dict = {col: 0 for col in sequence_cols}

    # Fill important inputs
    if "s2" in input_dict:
        input_dict["s2"] = s2
    if "cycle_norm" in input_dict:
        input_dict["cycle_norm"] = cycle

    # Convert to array
    row = np.array([list(input_dict.values())])

    # Normalize (only required cols)
    try:
        row_scaled = scaler.transform(row)
    except:
        row_scaled = row  # fallback if mismatch

    # Create fake sequence
    seq = np.repeat(row_scaled, sequence_length, axis=0)
    seq = seq.reshape(1, sequence_length, len(sequence_cols))

    # Predict
    pred = model.predict(seq)[0][0]

    # -----------------------------
    # Output
    # -----------------------------
    st.markdown("### 📊 Prediction Result")

    st.progress(float(pred))

    if pred > 0.5:
        st.error(f"⚠️ High Failure Risk ({pred*100:.2f}%)")
    else:
        st.success(f"✅ Engine Safe ({(1-pred)*100:.2f}% confidence)")

    st.info("This is a demo using simulated sequence input")
