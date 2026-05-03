import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

st.set_page_config(page_title="PredictiveMaint · LSTM", page_icon="⚙️", layout="wide")

# ── Fix paths to always point to same folder as app.py ───────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "binary_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# ── Debug: show what files exist (remove after fixing) ───────────────────────
with st.sidebar:
    st.markdown("## ⚙️ System Status")
    st.markdown("**Files found in app folder:**")
    try:
        files = os.listdir(BASE_DIR)
        for fn in files:
            st.caption(fn)
    except Exception as e:
        st.caption(f"Could not list files: {e}")
    st.markdown("---")

# ── Load model + scaler ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    model, meta = None, None
    errors = []

    # Try loading model — accepts both .h5 and .keras
    for path in [
        os.path.join(BASE_DIR, "binary_model.keras"),
        os.path.join(BASE_DIR, "binary_model.h5"),
        os.path.join(BASE_DIR, "model.keras"),
        os.path.join(BASE_DIR, "model.h5"),
    ]:
        if os.path.isfile(path):
            try:
                from tensorflow.keras.models import load_model
                model = load_model(path, compile=False)
                st.sidebar.success(f"Model loaded: {os.path.basename(path)}")
                break
            except Exception as e:
                errors.append(f"Model error ({os.path.basename(path)}): {e}")

    # Try loading scaler
    for path in [
        os.path.join(BASE_DIR, "scaler.pkl"),
        os.path.join(BASE_DIR, "model_bundle.pkl"),
    ]:
        if os.path.isfile(path):
            try:
                with open(path, "rb") as f:
                    meta = pickle.load(f)
                st.sidebar.success(f"Scaler loaded: {os.path.basename(path)}")
                break
            except Exception as e:
                errors.append(f"Scaler error ({os.path.basename(path)}): {e}")

    return model, meta, errors

model, meta, load_errors = load_assets()
assets_ready = model is not None and meta is not None

with st.sidebar:
    if not assets_ready:
        st.error("❌ Assets not found")
        for err in load_errors:
            st.caption(err)
        st.markdown("**Make sure these files are in your GitHub repo:**")
        st.code("binary_model.h5\nscaler.pkl", language=None)
    else:
        SEQUENCE_LENGTH = meta['sequence_length']
        SEQUENCE_COLS   = meta['sequence_cols']
        W1              = meta['w1']
        st.markdown(f"**Sequence length:** `{SEQUENCE_LENGTH}`")
        st.markdown(f"**Failure window:** `{W1}` cycles")
        st.markdown(f"**Features:** `{len(SEQUENCE_COLS)}`")

    st.markdown("---")
    st.markdown("**Input format:** space-separated `.txt`, no header")
    st.code("id · cycle · setting1-3 · s1-s21", language=None)

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("⚙️ Predictive Maintenance · LSTM")
st.caption("Binary classification — will the engine fail within the next 30 cycles?")
st.markdown("---")

if not assets_ready:
    st.error("⚠️ Model or scaler not found. Check the sidebar — it lists all files currently in the app folder.")
    st.markdown("**Common fixes:**")
    st.markdown("- Make sure `binary_model.h5` and `scaler.pkl` are pushed to your GitHub repo")
    st.markdown("- File names are case-sensitive — check they match exactly")
    st.markdown("- If your model file has a different name, rename it to `binary_model.h5`")
    st.stop()

# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess(raw_df):
    df = raw_df.copy()
    col_names = ['id','cycle','setting1','setting2','setting3'] + [f's{i}' for i in range(1,22)]
    df.columns = col_names[:len(df.columns)]
    df.sort_values(['id','cycle'], inplace=True)
    df['cycle_norm'] = df['cycle']
    scaler         = meta['scaler']
    cols_normalize = meta['cols_normalize']
    norm = pd.DataFrame(
        scaler.transform(df[cols_normalize]),
        columns=cols_normalize, index=df.index)
    df = df[df.columns.difference(cols_normalize)].join(norm)
    return df

def make_sequences(df):
    seqs, ids = [], []
    for eid in df['id'].unique():
        edf = df[df['id'] == eid]
        if len(edf) >= SEQUENCE_LENGTH:
            seqs.append(edf[SEQUENCE_COLS].values[-SEQUENCE_LENGTH:])
            ids.append(eid)
    return np.asarray(seqs).astype(np.float32), ids

def predict(seqs):
    probs = model.predict(seqs, verbose=0).flatten()
    preds = (probs > 0.5).astype(int)
    return probs, preds

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Evaluate", "📖 How It Works"])

# ─── TAB 1: PREDICT ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload Test File")
    f = st.file_uploader("PM_test.txt — space separated, no header", type=["txt","csv"])

    if f:
        raw = pd.read_csv(f, sep=r"\s+", header=None)
        raw.dropna(axis=1, inplace=True)

        try:
            df        = preprocess(raw)
            seqs, ids = make_sequences(df)
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()

        if len(seqs) == 0:
            st.warning(f"No engine has ≥ {SEQUENCE_LENGTH} cycles.")
            st.stop()

        probs, preds = predict(seqs)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Engines evaluated",   len(ids))
        c2.metric("⚠ Failure predicted", int(preds.sum()))
        c3.metric("✅ Safe",              int((preds==0).sum()))
        c4.metric("Avg failure prob",    f"{probs.mean():.1%}")
        st.markdown("---")

        results = pd.DataFrame({
            'Engine ID':        ids,
            'Failure Prob (%)': (probs * 100).round(1),
            'Prediction':       ['⚠ FAILURE' if p else '✅ SAFE' for p in preds]
        })
        st.dataframe(results, use_container_width=True, height=300)

        fig, ax = plt.subplots(figsize=(12, 3))
        colors = ['#e74c3c' if p else '#2ecc71' for p in preds]
        ax.bar(range(len(ids)), probs * 100, color=colors, width=0.6)
        ax.axhline(50, color='orange', linestyle='--', linewidth=1.2, label='Threshold 50%')
        ax.set_xlabel("Engine index")
        ax.set_ylabel("Failure probability (%)")
        ax.set_title("Failure Probability per Engine")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.download_button("⬇ Download predictions (.csv)",
                           results.to_csv(index=False).encode(),
                           "predictions.csv", "text/csv")
    else:
        st.info("Upload a PM_test.txt file to get predictions.")

# ─── TAB 2: EVALUATE ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("Upload Test + Ground Truth Files")
    col1, col2 = st.columns(2)
    with col1:
        test_f  = st.file_uploader("PM_test.txt",  type=["txt","csv"], key="ev_test")
    with col2:
        truth_f = st.file_uploader("PM_truth.txt", type=["txt","csv"], key="ev_truth")

    if test_f and truth_f:
        raw_test  = pd.read_csv(test_f,  sep=r"\s+", header=None)
        raw_truth = pd.read_csv(truth_f, sep=r"\s+", header=None)
        raw_test.dropna(axis=1, inplace=True)
        raw_truth.dropna(axis=1, inplace=True)

        df_test       = preprocess(raw_test)
        truth_df      = raw_truth.copy()
        truth_df.columns = ['additional_rul']
        truth_df['id']   = truth_df.index + 1

        rul_max = df_test.groupby('id')['cycle'].max().reset_index()
        rul_max.columns = ['id','max']
        truth_df['max'] = rul_max['max'] + truth_df['additional_rul']
        truth_df.drop('additional_rul', axis=1, inplace=True)

        df_test = df_test.merge(truth_df, on=['id'], how='left')
        df_test['RUL'] = df_test['max'] - df_test['cycle']
        df_test.drop('max', axis=1, inplace=True)
        df_test['failure_within_w1'] = np.where(df_test['RUL'] <= W1, 1, 0)

        y_mask = [len(df_test[df_test['id']==i]) >= SEQUENCE_LENGTH
                  for i in df_test['id'].unique()]
        seqs   = [df_test[df_test['id']==i][SEQUENCE_COLS].values[-SEQUENCE_LENGTH:]
                  for i in df_test['id'].unique()
                  if len(df_test[df_test['id']==i]) >= SEQUENCE_LENGTH]
        seqs   = np.asarray(seqs).astype(np.float32)

        y_true       = df_test.groupby('id')['failure_within_w1'].nth(-1)[y_mask].values.reshape(-1,1)
        probs, preds = predict(seqs)
        preds        = preds.reshape(-1,1)

        acc  = float((preds == y_true).mean())
        prec = precision_score(y_true, preds)
        rec  = recall_score(y_true, preds)
        f1   = f1_score(y_true, preds)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{acc:.3f}")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall",    f"{rec:.3f}")
        c4.metric("F1-Score",  f"{f1:.3f}")
        st.markdown("---")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Confusion Matrix**")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_true, preds), annot=True, fmt='d',
                        xticklabels=['Pred: Safe','Pred: Fail'],
                        yticklabels=['True: Safe','True: Fail'], ax=ax)
            st.pyplot(fig)
            plt.close(fig)

        with col_b:
            st.markdown("**Score Distribution**")
            fig, ax = plt.subplots()
            ax.hist(probs[y_true.flatten()==0]*100, bins=20, alpha=0.7, color='green', label='Safe')
            ax.hist(probs[y_true.flatten()==1]*100, bins=20, alpha=0.7, color='red',   label='Failure')
            ax.axvline(50, color='orange', linestyle='--', label='Threshold')
            ax.set_xlabel("Failure probability (%)")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Upload both PM_test.txt and PM_truth.txt to evaluate.")

# ─── TAB 3: HOW IT WORKS ─────────────────────────────────────────────────────
with tab3:
    st.markdown("### Architecture")
    st.code("""
Input  (batch, 50 time-steps, 25 features)
  ↓
LSTM  (100 units, return_sequences=True)
  ↓
Dropout 0.2
  ↓
LSTM  (50 units, return_sequences=False)
  ↓
Dropout 0.2
  ↓
Dense  (1 unit, sigmoid)
  ↓
Output  P(failure within next 30 cycles)
    """, language=None)

    st.markdown("### Feature Set (25 columns)")
    feat_df = pd.DataFrame({
        "Column":      ['setting1','setting2','setting3','cycle_norm'] + [f's{i}' for i in range(1,22)],
        "Description": ['Operational setting 1','Operational setting 2',
                        'Operational setting 3','Normalised cycle counter'] +
                       [f'Sensor reading {i}' for i in range(1,22)],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
