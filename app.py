# train_and_save_model.py
# Run this script ONCE to train the LSTM model and save it along with the scaler.
# Output: binary_model.keras  +  scaler.pkl

import numpy as np
import pandas as pd
import pickle
import os
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

np.random.seed(1234)

# ─────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv('PM_train.txt', sep=" ", header=None)
test_df  = pd.read_csv('PM_test.txt',  sep=" ", header=None)
truth_df = pd.read_csv('PM_truth.txt', sep=" ", header=None)

train_df.dropna(axis=1, inplace=True)
test_df.dropna(axis=1, inplace=True)
truth_df.dropna(axis=1, inplace=True)

cols_names = ['id','cycle','setting1','setting2','setting3',
              's1','s2','s3','s4','s5','s6','s7','s8','s9','s10',
              's11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']
train_df.columns = cols_names
test_df.columns  = cols_names

# ─────────────────────────────────────────
# 2.  PREPROCESSING – TRAIN
# ─────────────────────────────────────────
train_df.sort_values(['id','cycle'], inplace=True)
test_df.sort_values(['id','cycle'],  inplace=True)

rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id','max']
train_df = train_df.merge(rul, on=['id'], how='left')
train_df['RUL'] = train_df['max'] - train_df['cycle']
train_df.drop('max', axis=1, inplace=True)

w1 = 30
train_df['failure_within_w1'] = np.where(train_df['RUL'] <= w1, 1, 0)
train_df['cycle_norm'] = train_df['cycle']

cols_normalize = train_df.columns.difference(['id','cycle','RUL','failure_within_w1'])
min_max_scaler = preprocessing.MinMaxScaler()
norm_train_df  = pd.DataFrame(
    min_max_scaler.fit_transform(train_df[cols_normalize]),
    columns=cols_normalize, index=train_df.index)

join_df   = train_df[['id','cycle','RUL','failure_within_w1']].join(norm_train_df)
train_df  = join_df.reindex(columns=train_df.columns)

# ─────────────────────────────────────────
# 3.  PREPROCESSING – TEST
# ─────────────────────────────────────────
test_df['cycle_norm'] = test_df['cycle']
norm_test_df = pd.DataFrame(
    min_max_scaler.transform(test_df[cols_normalize]),
    columns=cols_normalize, index=test_df.index)
test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
test_df = test_join_df.reindex(columns=test_df.columns).reset_index(drop=True)

rul_test = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
rul_test.columns = ['id','max']
truth_df.columns = ['additional_rul']
truth_df['id']   = truth_df.index + 1
truth_df['max']  = rul_test['max'] + truth_df['additional_rul']
truth_df.drop('additional_rul', axis=1, inplace=True)

test_df = test_df.merge(truth_df, on=['id'], how='left')
test_df['RUL'] = test_df['max'] - test_df['cycle']
test_df.drop('max', axis=1, inplace=True)
test_df['failure_within_w1'] = np.where(test_df['RUL'] <= w1, 1, 0)

# ─────────────────────────────────────────
# 4.  SEQUENCE GENERATION
# ─────────────────────────────────────────
sequence_length = 50
sensor_cols    = ['s' + str(i) for i in range(1, 22)]
sequence_cols  = ['setting1','setting2','setting3','cycle_norm'] + sensor_cols

def sequence_generator(df, seq_len, seq_cols):
    arr = df[seq_cols].values
    n   = arr.shape[0]
    for start, stop in zip(range(0, n - seq_len), range(seq_len, n)):
        yield arr[start:stop, :]

def label_generator(df, seq_len, label):
    arr = df[label].values
    return arr[seq_len:arr.shape[0], :]

print("Generating sequences...")
seq_gen = (list(sequence_generator(train_df[train_df['id']==id], sequence_length, sequence_cols))
           for id in train_df['id'].unique())
seq_set = np.concatenate(list(seq_gen)).astype(np.float32)

label_gen = [label_generator(train_df[train_df['id']==id], sequence_length, ['failure_within_w1'])
             for id in train_df['id'].unique()]
label_set = np.concatenate(label_gen).astype(np.float32)

print(f"seq_set shape : {seq_set.shape}")
print(f"label_set shape: {label_set.shape}")

# ─────────────────────────────────────────
# 5.  BUILD & TRAIN LSTM
# ─────────────────────────────────────────
features_dim = seq_set.shape[2]
out_dim      = label_set.shape[1]

model = Sequential([
    LSTM(units=100, input_shape=(sequence_length, features_dim), return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=out_dim, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

MODEL_PATH = 'binary_model.keras'   # native Keras format (recommended over .h5)

history = model.fit(
    seq_set, label_set,
    epochs=200, batch_size=200,
    validation_split=0.05, verbose=2,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, mode='min'),
        ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, mode='min')
    ]
)

# ─────────────────────────────────────────
# 6.  EVALUATE ON TEST SET
# ─────────────────────────────────────────
y_mask = [len(test_df[test_df['id']==id]) >= sequence_length for id in test_df['id'].unique()]
last_test_seq = [test_df[test_df['id']==id][sequence_cols].values[-sequence_length:]
                 for id in test_df['id'].unique() if len(test_df[test_df['id']==id]) >= sequence_length]
last_test_seq   = np.asarray(last_test_seq).astype(np.float32)
last_test_label = test_df.groupby('id')['failure_within_w1'].nth(-1)[y_mask].values
last_test_label = last_test_label.reshape(-1, 1).astype(np.float32)

estimator    = keras.models.load_model(MODEL_PATH)
scores_test  = estimator.evaluate(last_test_seq, last_test_label, verbose=2)
y_pred_test  = (estimator.predict(last_test_seq) > 0.5).astype("int32")

precision = precision_score(last_test_label, y_pred_test)
recall    = recall_score(last_test_label,    y_pred_test)
f1        = 2 * precision * recall / (precision + recall)

print(f"\n✅ Test Accuracy : {scores_test[1]:.4f}")
print(f"   Precision     : {precision:.4f}")
print(f"   Recall        : {recall:.4f}")
print(f"   F1-Score      : {f1:.4f}")
print(f"   Confusion Matrix:\n{confusion_matrix(last_test_label, y_pred_test)}")

# ─────────────────────────────────────────
# 7.  SAVE SCALER  (needed by Streamlit app)
# ─────────────────────────────────────────
SCALER_PATH = 'scaler.pkl'
with open(SCALER_PATH, 'wb') as f:
    pickle.dump({
        'scaler': min_max_scaler,
        'cols_normalize': list(cols_normalize),
        'sequence_cols': sequence_cols,
        'sequence_length': sequence_length,
        'w1': w1
    }, f)

print(f"\n💾 Model  saved → {MODEL_PATH}")
print(f"💾 Scaler saved → {SCALER_PATH}")
print("\nYou can now run:  streamlit run app.py")