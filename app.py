import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Engine Failure Predictor", page_icon="⚙️")

st.title("⚙️ Engine Failure Prediction")
st.caption("Upload engine data file to predict failure risk")

# -----------------------------
# Model Architecture
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
# Load assets
# -----------------------------
@st.cache_resource
def load_assets():
    with open("scaler.pkl", "rb") as f:
        data = pickle.load(f)

    sequence_cols = data["sequence_cols"]
    sequence_length = data["sequence_length"]

    model = build_model((sequence_length, len(sequence_cols)))
    model.load_weights("model_weights.h5")

    return model, data

model, data = load_assets()

scaler = data["scaler"]
sequence_cols = data["sequence_cols"]
sequence_length = data["sequence_length"]

# -----------------------------
# Instructions
# -----------------------------
st.markdown("### 📄 Upload Input File")

st.info("""
Upload a `.txt` file containing engine sensor data.

Format requirements:
- Space-separated values (like training dataset)
- Must include ALL required columns:
    id, cycle, setting1, setting2, setting3, s1...s21
- File must contain at least 50 rows
""")

uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:
    try:
        # Load txt file (space separated)
        df = pd.read_csv(uploaded_file, sep=" ", header=None)

        # Drop empty columns (important)
        df.dropna(axis=1, inplace=True)

        # Column names (same as training)
        cols_names = [
            'id','cycle','setting1','setting2','setting3',
            's1','s2','s3','s4','s5','s6','s7','s8','s9','s10',
            's11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21'
        ]

        df.columns = cols_names

        st.success("✅ File loaded successfully")

        # -----------------------------
        # Preprocessing
        # -----------------------------
        # Add cycle_norm
        df["cycle_norm"] = df["cycle"]

        # Keep only required columns
        df_input = df[sequence_cols]

        # Normalize
        df_scaled = scaler.transform(df_input)

        # Take last 50 rows
        if len(df_scaled) < sequence_length:
            st.error("❌ File must contain at least 50 rows")
        else:
            seq = df_scaled[-sequence_length:]
            seq = seq.reshape(1, sequence_length, len(sequence_cols))

            # -----------------------------
            # Prediction
            # -----------------------------
            pred = float(model.predict(seq, verbose=0)[0][0])

            st.markdown("### 📊 Prediction Result")

            st.progress(pred)

            if pred > 0.5:
                st.error(f"⚠️ High Failure Risk ({pred*100:.2f}%)")
            else:
                st.success(f"✅ Engine Safe ({(1-pred)*100:.2f}% confidence)")

    except Exception as e:
        st.error("❌ Error processing file")
        st.text(str(e))

# -----------------------------
# Footer
# -----------------------------
st.caption("Model predicts failure within next 30 cycles based on last 50 timesteps")
