import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Engine Failure Predictor",
    page_icon="⚙️",
    layout="centered"
)

# -----------------------------
# Custom Theme (White + Aqua)
# -----------------------------
st.markdown("""
    <style>
    .stApp {
        background-color: #f5feff;
    }
    .title {
        color: #00c2d1;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #555;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">⚙️ Engine Failure Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Demo using simulated sequence input</div>', unsafe_allow_html=True)

# -----------------------------
# Load Model + Scaler
# -----------------------------
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("model.h5", compile=False)
    
    with open("scaler.pkl", "rb") as f:
        data = pickle.load(f)

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
        # Create base input
        input_dict = {col: 0 for col in sequence_cols}

        if "s2" in input_dict:
            input_dict["s2"] = s2
        if "cycle_norm" in input_dict:
            input_dict["cycle_norm"] = cycle

        # Convert to array
        row = np.array([list(input_dict.values())], dtype=float)

        # Normalize
        row_scaled = scaler.transform(row)

        # Create fake sequence
        seq = np.repeat(row_scaled, sequence_length, axis=0)
        seq = seq.reshape(1, sequence_length, len(sequence_cols))

        # Predict
        pred = float(model.predict(seq, verbose=0)[0][0])

        # -----------------------------
        # Output
        # -----------------------------
        st.markdown("### 📊 Prediction Result")

        st.progress(pred)

        if pred > 0.5:
            st.error(f"⚠️ High Failure Risk ({pred*100:.2f}%)")
        else:
            st.success(f"✅ Engine Safe ({(1-pred)*100:.2f}% confidence)")

    except Exception as e:
        st.error("❌ Error in prediction. Check model/scaler compatibility.")
        st.text(str(e))

# -----------------------------
# Footer
# -----------------------------
st.caption("Demo project using RNN/LSTM with simulated sequence input")
