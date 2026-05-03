import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="wide")

# ── Load everything from single pkl ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_bundle():
    with open('model_bundle.pkl', 'rb') as f:
        bundle = pickle.load(f)

    # Rebuild model from saved config + weights
    from tensorflow.keras.models import model_from_config
    model = model_from_config(bundle['model_config'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.set_weights(bundle['model_weights'])

    return model, bundle

try:
    model, bundle = load_bundle()
    SCALER          = bundle['scaler']
    COLS_NORMALIZE  = bundle['cols_normalize']
    SEQUENCE_COLS   = bundle['sequence_cols']
    SEQUENCE_LENGTH = bundle['sequence_length']
    W1              = bundle['w1']
    assets_ok = True
except Exception as e:
    assets_ok = False
    load_err  = str(e)

# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess(raw_df):
    df = raw_df.copy()
    col_names = ['id','cycle','setting1','setting2','setting3'] + [f's{i}' for i in range(1,22)]
    df.columns = col_names[:len(df.columns)]
    df.sort_values(['id','cycle'], inplace=True)
    df['cycle_norm'] = df['cycle']
    norm = pd.DataFrame(
        SCALER.transform(df[COLS_NORMALIZE]),
        columns=COLS_NORMALIZE, index=df.index)
    df = df[df.columns.difference(COLS_NORMALIZE)].join(norm)
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Status")
    if assets_ok:
        st.success("Model loaded ✓")
        st.markdown(f"**Sequence length:** `{SEQUENCE_LENGTH}`")
        st.markdown(f"**Failure window:** `{W1}` cycles")
        st.markdown(f"**Features:** `{len(SEQUENCE_COLS)}`")
    else:
        st.error("model_bundle.pkl not found")
        st.caption(load_err)

    st.markdown("---")
    st.markdown("**Input format:** space-separated `.txt`, no header  \n`id · cycle · setting1-3 · s1-s21`")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("⚙️ Predictive Maintenance — LSTM")
st.caption("Will the engine fail within the next 30 cycles?")
st.markdown("---")

if not assets_ok:
    st.error("Could not load model_bundle.pkl — check sidebar.")
    st.stop()

tab1, tab2 = st.tabs(["🔍 Predict", "📊 Evaluate"])

# ─── TAB 1: PREDICT ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload test file")
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

        # Summary
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Engines",         len(ids))
        c2.metric("⚠ Failure",       int(preds.sum()))
        c3.metric("✅ Safe",          int((preds==0).sum()))
        c4.metric("Avg probability", f"{probs.mean():.1%}")

        st.markdown("---")

        # Table
        results = pd.DataFrame({
            'Engine ID':        ids,
            'Failure Prob (%)': (probs * 100).round(1),
            'Prediction':       ['⚠ FAILURE' if p else '✅ SAFE' for p in preds]
        })
        st.dataframe(results, use_container_width=True, height=300)

        # Bar chart
        fig, ax = plt.subplots(figsize=(12, 3))
        colors  = ['#e74c3c' if p else '#2ecc71' for p in preds]
        ax.bar(range(len(ids)), probs * 100, color=colors)
        ax.axhline(50, color='orange', linestyle='--', linewidth=1, label='Threshold 50%')
        ax.set_xlabel("Engine index")
        ax.set_ylabel("Failure probability (%)")
        ax.set_title("Failure Probability per Engine")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        # Download
        st.download_button("⬇ Download predictions (.csv)",
                           results.to_csv(index=False).encode(),
                           "predictions.csv", "text/csv")
    else:
        st.info("Upload a PM_test.txt file to get predictions.")

# ─── TAB 2: EVALUATE ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("Upload test + ground truth to evaluate model performance")
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

        df_test  = preprocess(raw_test)
        truth_df = raw_truth.copy()
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

        y_true = df_test.groupby('id')['failure_within_w1'].nth(-1)[y_mask].values.reshape(-1,1)
        probs, preds = predict(seqs)
        preds  = preds.reshape(-1,1)

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
            ax.axvline(50, color='orange', linestyle='--')
            ax.set_xlabel("Failure probability (%)")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

    else:
        st.info("Upload both files to evaluate.")
