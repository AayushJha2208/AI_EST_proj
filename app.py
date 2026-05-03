import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="wide")

# ── Load model + scaler ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    model  = load_model("model.h5", compile=False)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

SEQUENCE_LENGTH = 50

# ── Page ──────────────────────────────────────────────────────────────────────
st.title("⚙️ Predictive Maintenance — LSTM")
st.caption("Enter current sensor readings to predict if the engine will fail within 30 cycles.")
st.markdown("---")

# ── Sidebar info ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ How it works")
    st.markdown("""
1. Enter the **current sensor values** for the engine
2. Click **Predict**
3. The model will tell you if the engine is likely to fail within the next **30 cycles**

> Sensor values should be in their **raw/original scale** — the app normalizes them automatically.
""")
    st.markdown("---")
    st.markdown("**Tip:** Use values from your test dataset for realistic results.")

# ── Operational settings ──────────────────────────────────────────────────────
st.subheader("⚙️ Operational Settings")
col1, col2, col3, col4 = st.columns(4)
with col1:
    setting1   = st.number_input("Setting 1",   value=0.0,  step=0.1, format="%.2f")
with col2:
    setting2   = st.number_input("Setting 2",   value=0.0,  step=0.1, format="%.2f")
with col3:
    setting3   = st.number_input("Setting 3",   value=0.0,  step=0.1, format="%.2f")
with col4:
    cycle_norm = st.number_input("Cycle (norm)", value=100.0, step=1.0, format="%.1f")

st.markdown("---")

# ── Sensor inputs ─────────────────────────────────────────────────────────────
st.subheader("🔬 Sensor Readings")

# Row 1: s1 - s7
cols = st.columns(7)
sensors = {}
for i, col in enumerate(cols, start=1):
    sensors[f's{i}'] = col.number_input(f"s{i}", value=0.5, step=0.01, format="%.3f")

# Row 2: s8 - s14
cols = st.columns(7)
for i, col in enumerate(cols, start=8):
    sensors[f's{i}'] = col.number_input(f"s{i}", value=0.5, step=0.01, format="%.3f")

# Row 3: s15 - s21
cols = st.columns(7)
for i, col in enumerate(cols, start=15):
    sensors[f's{i}'] = col.number_input(f"s{i}", value=0.5, step=0.01, format="%.3f")

st.markdown("---")

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True):

    # Build one feature vector from user inputs
    # Order: setting1, setting2, setting3, cycle_norm, s1..s21  (25 features)
    single_step = np.array([[
        setting1, setting2, setting3, cycle_norm,
        sensors['s1'],  sensors['s2'],  sensors['s3'],  sensors['s4'],
        sensors['s5'],  sensors['s6'],  sensors['s7'],  sensors['s8'],
        sensors['s9'],  sensors['s10'], sensors['s11'], sensors['s12'],
        sensors['s13'], sensors['s14'], sensors['s15'], sensors['s16'],
        sensors['s17'], sensors['s18'], sensors['s19'], sensors['s20'],
        sensors['s21']
    ]])  # shape (1, 25)

    # Normalize using the saved scaler
    single_step_scaled = scaler.transform(single_step)  # shape (1, 25)

    # Build fake sequence by repeating the same timestep 50 times
    # shape → (1, 50, 25)
    sequence = np.tile(single_step_scaled, (SEQUENCE_LENGTH, 1))
    sequence = sequence.reshape(1, SEQUENCE_LENGTH, 25).astype(np.float32)

    # Predict
    prob = model.predict(sequence, verbose=0)[0][0]

    # ── Output ────────────────────────────────────────────────────────────────
    st.markdown("## 📊 Prediction Result")

    c1, c2 = st.columns(2)
    c1.metric("Failure Probability", f"{prob*100:.1f}%")
    c2.metric("Decision Threshold",  "50%")

    st.progress(float(prob))

    if prob > 0.5:
        st.error(f"⚠️ Engine is likely to **FAIL** within 30 cycles  (confidence: {prob*100:.1f}%)")
    else:
        st.success(f"✅ Engine is **SAFE**  (failure probability: {prob*100:.1f}%)")

    # Probability gauge bar
    st.markdown("### Risk Level")
    if prob < 0.33:
        st.markdown("🟢 **Low risk**")
    elif prob < 0.66:
        st.markdown("🟡 **Medium risk**")
    else:
        st.markdown("🔴 **High risk**")
