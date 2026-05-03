import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Engine Failure Predictor",
    page_icon="⚙️",
    layout="centered"
)

# -----------------------------
# UI Theme
# -----------------------------
st.markdown("""
<style>
.stApp { background-color: #f5feff; }
.title { color: #00c2d1; text-align: center; font-size: 36px; font-weight: bold; }
.subtitle { text-align: center; color: #555; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">⚙️ Engine Failure Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Demo using simulated sequence input</div>', unsafe_allow_html=True)

# -----------------------------
# Build Model Architecture (IMPORTANT)
# -----------------------------
def build_model(input_shape):
    model = Sequential()

    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(LSTM(50))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))

    return model

# -----------------------------
# Load Assets
# -----------------------------
@st.cache_resource
def load_assets():
    with open("scaler.pkl", "rb") as f:
        data = pickle.load(f)

    sequence_cols = data["sequence_cols"]
    sequence_length = data["sequence_length"]

    model = build_model((sequence_length, len(sequence_cols)))
    model.load_weights("model.h5")

    return model, data

model, data = load_assets()

scaler = data["scaler"]
sequence_cols = data["sequence_cols"]
sequence_length = data["sequence_length"]

# -----------------------------
# Inputs
# -----------------------------
st.markdown("### 🔢 Input Parameters")

s2 = st.slider("Sensor s2", 0.0, 1.0, 0.5)
cycle = st.slider("Cycle", 1, 300, 100)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🚀 Predict"):

    try:
        input_dict = {col: 0 for col in sequence_cols}

        if "s2" in input_dict:
            input_dict["s2"] = s2
        if "cycle_norm" in input_dict:
            input_dict["cycle_norm"] = cycle

        row = np.array([list(input_dict.values())], dtype=float)
        row_scaled = scaler.transform(row)

        seq = np.repeat(row_scaled, sequence_length, axis=0)
        seq = seq.reshape(1, sequence_length, len(sequence_cols))

        pred = float(model.predict(seq, verbose=0)[0][0])

        st.markdown("### 📊 Prediction Result")
        st.progress(pred)

        if pred > 0.5:
            st.error(f"⚠️ High Failure Risk ({pred*100:.2f}%)")
        else:
            st.success(f"✅ Engine Safe ({(1-pred)*100:.2f}% confidence)")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.text(str(e))

st.caption("Demo project using LSTM with simulated input")
