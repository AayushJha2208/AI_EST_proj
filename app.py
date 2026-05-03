import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("model.h5")

st.title("🔧 Engine Failure Prediction (Demo)")

st.write("Enter sensor values")

# You can keep 1 or multiple inputs
s2 = st.number_input("Sensor s2", value=0.5)

# OPTIONAL: more sensors (if your model uses 25 features)
# s3 = st.number_input("Sensor s3", value=0.5)
# s4 = st.number_input("Sensor s4", value=0.5)

if st.button("Predict"):

    # --- FAKE SEQUENCE ---
    sequence_length = 50
    features = 1   # change to 25 if using full model

    # Create sequence (repeat same value)
    seq = np.array([[s2] * sequence_length])
    seq = seq.reshape(1, sequence_length, features)

    # Prediction
    pred = model.predict(seq)

    # Output
    if pred[0][0] > 0.5:
        st.error("⚠️ Engine likely to fail soon")
    else:
        st.success("✅ Engine is safe")
