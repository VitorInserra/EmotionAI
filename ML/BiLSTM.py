# ============================================================
# bilstm_lkocv_classification.py
#
# Purpose:
#   - Read the base EEG dataset
#   - Build engineered EEG features from PSD rows
#   - Build trial-level variable-length sequences
#   - Run leave-k-out cross-validation across trials/images
#   - Train a basic TensorFlow BiLSTM for 4-class classification
#   - Save metrics, predictions, and plots to outputs/
#
# Important:
#   - The target column is "user_predicted_code"
#   - Default evaluation is true LOOCV because LEAVE_K_OUT = 1
#   - Increase LEAVE_K_OUT to run exact leave-k-out CV
#
# Input:
#   datasets/base_dataset.csv   (preferred)
#   base_dataset.csv            (fallback)
# ============================================================

from __future__ import annotations

import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # important for SLURM / headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeavePOut
from sklearn.preprocessing import StandardScaler

from utils import HOMOLOGOUS_PAIRS, compute_asymmetry_from_psd

import eegproc as eeg
import joblib


# =========================
# CONFIG
# =========================
INPUT_FEATURE_TABLE_CANDIDATES = [
    "datasets/base_dataset.csv",
    "base_dataset.csv",
]

SEQUENCE_ID_COLUMN = "trial_id"
GROUP_COLUMN = "session_id"
TIME_COLUMN = "timestep_idx"
TARGET_COLUMN = "user_predicted_code"
IMAGE_COLUMN = "image_name"
TRIAL_INDEX_COLUMN = "trial_index"
SAMPLE_TIME_COLUMN = "sample_timestamp"

CLASS_ORDER = ["LALV", "LAHV", "HAHV", "HALV"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(CLASS_ORDER)}
INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}

# Leave-k-out CV
LEAVE_K_OUT = 1

# Training
MAX_EPOCHS = 250
EARLY_STOPPING_PATIENCE = 30
SEED = 42
VERBOSE_FIT = 0

# Output management
OUTPUT_ROOT = Path("outputs")
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_OUTPUT_DIR = OUTPUT_ROOT / f"lkocv_bilstm_{RUN_STAMP}"
MODEL_DIR = Path("models")

# Final model save
SAVE_FINAL_MODEL = True

# Hyperparameters
HYPERPARAMS = {
    "lstm_units": 64,
    "dense_units": 32,
    "dropout": 0.30,
    "learning_rate": 1e-3,
    "batch_size": 8,
}

CHANNELS = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]
BANDS = ["theta", "alpha", "betaL", "betaH", "gamma"]
POW_COLUMNS = [f"{ch}_{band}" for ch in CHANNELS for band in BANDS]
ENTROPY_COLUMNS = [f"{ch}_{band}_entropy" for band in BANDS for ch in CHANNELS]
ASYMMETRY_COLUMNS = [
    f"{right}_{left}_{band}_{kind}"
    for left, right in HOMOLOGOUS_PAIRS
    for band in BANDS
    for kind in ("da", "ra")
]
FEATURE_COLUMNS = POW_COLUMNS + ENTROPY_COLUMNS + ASYMMETRY_COLUMNS

# Metadata columns to exclude from features after feature table creation
META_COLUMNS = {
    "session_id",
    "trial_id",
    "trial_index",
    "timestep_idx",
    "image_name",
    "trial_started_at",
    "trial_ended_at",
    "time_elapsed_seconds",
    "sample_timestamp",
    "user_predicted_key",
    TARGET_COLUMN,
    "user_predicted_label",
    "model_predicted_key",
    "model_predicted_code",
    "model_predicted_label",
    "model_match",
    "sensor_contact_quality",
    "row_index",
    "__row_order__",
}


# =========================
# REPRODUCIBILITY
# =========================
def set_all_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =========================
# OUTPUT HELPERS
# =========================
def prepare_output_dir() -> Path:
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return RUN_OUTPUT_DIR


def save_text_summary(text: str, output_dir: Path, filename: str = "run_summary.txt") -> None:
    path = output_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_plot_fold_accuracy(predictions_df: pd.DataFrame, output_dir: Path) -> None:
    fold_acc = (
        predictions_df.groupby("fold", as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "fold_accuracy"})
    )

    plt.figure(figsize=(8, 4))
    plt.bar(fold_acc["fold"], fold_acc["fold_accuracy"])
    plt.ylim(0.0, 1.05)
    plt.xlabel(f"Leave-{LEAVE_K_OUT}-out Fold")
    plt.ylabel("Accuracy")
    plt.title(f"BiLSTM Leave-{LEAVE_K_OUT}-out Fold Accuracy")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "fold_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_plot_confusion_matrix(cm: np.ndarray, output_dir: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"BiLSTM Leave-{LEAVE_K_OUT}-out Confusion Matrix")
    plt.colorbar()

    ticks = np.arange(len(CLASS_ORDER))
    plt.xticks(ticks, CLASS_ORDER)
    plt.yticks(ticks, CLASS_ORDER)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_plot_final_history(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    if "loss" in history.history:
        plt.plot(history.history["loss"], label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")
    if "accuracy" in history.history:
        plt.plot(history.history["accuracy"], label="train_accuracy")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Final Training History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "final_training_history.png", dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# DATA HELPERS
# =========================
def resolve_input_feature_table() -> Path:
    for candidate in INPUT_FEATURE_TABLE_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find base dataset. Looked for: "
        + ", ".join(INPUT_FEATURE_TABLE_CANDIDATES)
    )


def load_feature_table() -> pd.DataFrame:
    path = resolve_input_feature_table()

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        expanded_header = header[:6] + [SAMPLE_TIME_COLUMN] + header[6:]
        normalized_rows: list[list[str]] = []

        for row in reader:
            if len(row) == len(header):
                row = row[:6] + [""] + row[6:]
            elif len(row) != len(header) + 1:
                raise ValueError(
                    f"Unexpected row length {len(row)} in {path}. "
                    f"Expected {len(header)} or {len(header)+1}. Row head: {row[:10]}"
                )
            normalized_rows.append(row)

    df = pd.DataFrame(normalized_rows, columns=expanded_header)
    df["__row_order__"] = np.arange(len(df))

    numeric_cols = [TRIAL_INDEX_COLUMN, "time_elapsed_seconds", "sensor_contact_quality", *POW_COLUMNS]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in [SAMPLE_TIME_COLUMN, "trial_started_at", "trial_ended_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.upper().str.strip()
    df[SEQUENCE_ID_COLUMN] = (
        df[GROUP_COLUMN].astype(str)
        + "__"
        + df[TRIAL_INDEX_COLUMN].astype("Int64").astype(str)
    )

    sort_keys = [GROUP_COLUMN, TRIAL_INDEX_COLUMN]
    if SAMPLE_TIME_COLUMN in df.columns:
        sort_keys.append(SAMPLE_TIME_COLUMN)
    sort_keys.append("__row_order__")

    df = df.sort_values(sort_keys, kind="stable").reset_index(drop=True)
    df[TIME_COLUMN] = df.groupby(SEQUENCE_ID_COLUMN).cumcount()
    return df



def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in POW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing PSD columns: {missing[:10]}")

    psd = (
        df[POW_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    entropy = eeg.shannons_entropy(psd, fs=128)
    asymm = compute_asymmetry_from_psd(psd).reset_index(drop=True)

    feature_frame = pd.concat([df.reset_index(drop=True), psd.reset_index(drop=True), entropy, asymm], axis=1)

    for col in FEATURE_COLUMNS:
        if col not in feature_frame.columns:
            feature_frame[col] = 0.0

    return feature_frame.fillna(0.0)


def build_trial_sequences(df: pd.DataFrame):
    """
    Returns:
        X_list         : list of arrays, each [timesteps, n_features]
        y              : class index, one per trial
        groups         : session_id, one per trial
        trial_ids      : one per trial
        image_names    : image_name, one per trial
        feature_cols   : list of feature column names
        global_max_len : max timesteps across all trials
    """
    feature_cols = [c for c in FEATURE_COLUMNS if c not in META_COLUMNS]
    if not feature_cols:
        raise ValueError("No feature columns found after excluding metadata columns.")

    X_list = []
    y_list = []
    groups_list = []
    trial_ids_list = []
    image_names_list = []

    for trial_id, g in df.groupby(SEQUENCE_ID_COLUMN, sort=False):
        g = g.sort_values(TIME_COLUMN, kind="stable")

        label_text = str(g[TARGET_COLUMN].iloc[0]).upper().strip()
        if label_text not in LABEL_TO_INDEX:
            continue

        x = (
            g[feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )

        if len(x) == 0:
            continue

        y = int(LABEL_TO_INDEX[label_text])
        group = g[GROUP_COLUMN].iloc[0]
        image_name = g[IMAGE_COLUMN].iloc[0] if IMAGE_COLUMN in g.columns else ""

        X_list.append(x)
        y_list.append(y)
        groups_list.append(group)
        trial_ids_list.append(trial_id)
        image_names_list.append(image_name)

    if not X_list:
        raise ValueError("No valid trial sequences found.")

    y = np.asarray(y_list, dtype=np.int32)
    groups = np.asarray(groups_list)
    trial_ids = np.asarray(trial_ids_list)
    image_names = np.asarray(image_names_list)
    global_max_len = max(seq.shape[0] for seq in X_list)

    return X_list, y, groups, trial_ids, image_names, feature_cols, global_max_len


def subset_list_by_indices(lst, indices):
    return [lst[i] for i in indices]


def fit_feature_scaler(seq_list: List[np.ndarray]) -> StandardScaler:
    """
    Fit on stacked training timesteps only.
    """
    stacked = np.vstack(seq_list)
    scaler = StandardScaler()
    scaler.fit(stacked)
    return scaler


def transform_sequence_list(seq_list: List[np.ndarray], scaler: StandardScaler) -> List[np.ndarray]:
    return [scaler.transform(seq).astype(np.float32) for seq in seq_list]


def pad_sequence_list(seq_list: List[np.ndarray], max_len: int, n_features: int) -> np.ndarray:
    X = np.zeros((len(seq_list), max_len, n_features), dtype=np.float32)

    for i, seq in enumerate(seq_list):
        seq_len = min(seq.shape[0], max_len)
        X[i, :seq_len, :] = seq[:seq_len, :]

    return X


def choose_validation_index(y_train: np.ndarray) -> int:
    """
    Pick a validation example from the training fold while trying not to erase
    the only example of a class when possible.
    """
    if len(y_train) <= 1:
        return 0

    counts = pd.Series(y_train).value_counts()
    candidate_classes = [cls for cls, count in counts.items() if count > 1]
    if candidate_classes:
        chosen_class = int(counts.loc[candidate_classes].idxmax())
        return int(np.where(y_train == chosen_class)[0][-1])
    return len(y_train) - 1


# =========================
# MODEL
# =========================
def build_bilstm_classifier(max_len: int, n_features: int, hp: dict) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(max_len, n_features), name="sequence_input")

    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hp["lstm_units"], return_sequences=False)
    )(x)
    x = tf.keras.layers.Dropout(hp["dropout"])(x)
    x = tf.keras.layers.Dense(hp["dense_units"], activation="relu")(x)
    x = tf.keras.layers.Dropout(hp["dropout"])(x)
    outputs = tf.keras.layers.Dense(len(CLASS_ORDER), activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=hp["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =========================
# TRAIN / EVAL HELPERS
# =========================
def compute_class_weights(y_train: np.ndarray) -> dict[int, float]:
    counts = pd.Series(y_train).value_counts().to_dict()
    n_classes = len(CLASS_ORDER)
    n_samples = len(y_train)

    weights: dict[int, float] = {}
    for class_idx in range(n_classes):
        count = counts.get(class_idx, 0)
        weights[class_idx] = 1.0 if count == 0 else n_samples / (n_classes * count)
    return weights


def train_one_lkocv_fold(
    X_train_seq: List[np.ndarray],
    y_train: np.ndarray,
    X_test_seq: List[np.ndarray],
    y_test: np.ndarray,
    hp: dict,
    max_len: int,
    n_features: int,
):
    val_idx = choose_validation_index(y_train)

    X_val_seq = [X_train_seq[val_idx]]
    y_val = np.asarray([y_train[val_idx]], dtype=np.int32)

    X_fit_seq = [seq for i, seq in enumerate(X_train_seq) if i != val_idx]
    y_fit = np.asarray([label for i, label in enumerate(y_train) if i != val_idx], dtype=np.int32)

    if len(X_fit_seq) == 0:
        X_fit_seq = X_train_seq
        y_fit = y_train

    scaler = fit_feature_scaler(X_fit_seq)

    X_fit_scaled = transform_sequence_list(X_fit_seq, scaler)
    X_val_scaled = transform_sequence_list(X_val_seq, scaler)
    X_test_scaled = transform_sequence_list(X_test_seq, scaler)

    X_fit_pad = pad_sequence_list(X_fit_scaled, max_len=max_len, n_features=n_features)
    X_val_pad = pad_sequence_list(X_val_scaled, max_len=max_len, n_features=n_features)
    X_test_pad = pad_sequence_list(X_test_scaled, max_len=max_len, n_features=n_features)

    model = build_bilstm_classifier(max_len=max_len, n_features=n_features, hp=hp)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        )
    ]

    history = model.fit(
        X_fit_pad,
        y_fit,
        validation_data=(X_val_pad, y_val),
        epochs=MAX_EPOCHS,
        batch_size=hp["batch_size"],
        verbose=VERBOSE_FIT,
        callbacks=callbacks,
        class_weight=compute_class_weights(y_fit),
    )

    probs = model.predict(X_test_pad, verbose=0)
    preds = probs.argmax(axis=1)

    val_losses = history.history["val_loss"]
    best_epoch = int(np.argmin(val_losses) + 1)
    best_val_loss = float(np.min(val_losses))

    fold_accuracy = float(accuracy_score(y_test, preds))
    fold_f1_macro = float(f1_score(y_test, preds, average="macro", zero_division=0))

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "test_accuracy": fold_accuracy,
        "test_f1_macro": fold_f1_macro,
        "preds": preds,
        "probs": probs,
        "scaler": scaler,
    }


def train_final_model(
    X_list: List[np.ndarray],
    y: np.ndarray,
    hp: dict,
    final_epochs: int,
    max_len: int,
    n_features: int,
):
    val_idx = choose_validation_index(y)

    X_val_seq = [X_list[val_idx]]
    y_val = np.asarray([y[val_idx]], dtype=np.int32)

    X_fit_seq = [seq for i, seq in enumerate(X_list) if i != val_idx]
    y_fit = np.asarray([label for i, label in enumerate(y) if i != val_idx], dtype=np.int32)

    if len(X_fit_seq) == 0:
        X_fit_seq = X_list
        y_fit = y

    scaler = fit_feature_scaler(X_fit_seq)

    X_fit_scaled = transform_sequence_list(X_fit_seq, scaler)
    X_val_scaled = transform_sequence_list(X_val_seq, scaler)
    X_all_scaled = transform_sequence_list(X_list, scaler)

    X_fit_pad = pad_sequence_list(X_fit_scaled, max_len=max_len, n_features=n_features)
    X_val_pad = pad_sequence_list(X_val_scaled, max_len=max_len, n_features=n_features)
    X_all_pad = pad_sequence_list(X_all_scaled, max_len=max_len, n_features=n_features)

    model = build_bilstm_classifier(max_len=max_len, n_features=n_features, hp=hp)

    history = model.fit(
        X_fit_pad,
        y_fit,
        validation_data=(X_val_pad, y_val),
        epochs=final_epochs,
        batch_size=hp["batch_size"],
        verbose=VERBOSE_FIT,
        class_weight=compute_class_weights(y_fit),
    )

    probs = model.predict(X_all_pad, verbose=0)
    preds = probs.argmax(axis=1)

    train_accuracy = float(accuracy_score(y, preds))
    train_f1_macro = float(f1_score(y, preds, average="macro", zero_division=0))

    return {
        "model": model,
        "scaler": scaler,
        "history": history,
        "preds": preds,
        "probs": probs,
        "train_accuracy": train_accuracy,
        "train_f1_macro": train_f1_macro,
    }


def save_final_model_bundle(
    model: tf.keras.Model,
    scaler: StandardScaler,
    feature_cols: list[str],
    global_max_len: int,
    hp: dict,
    output_dir: Path,
    overall_accuracy: float,
    overall_f1_macro: float,
) -> None:
    keras_path = MODEL_DIR / "quadrant_bilstm_lkocv.keras"
    bundle_path = MODEL_DIR / "quadrant_bilstm_lkocv_bundle.joblib"

    model.save(keras_path)
    joblib.dump(
        {
            "model_type": "keras_bilstm_classifier",
            "keras_model_path": str(keras_path.resolve()),
            "feature_names": feature_cols,
            "pow_columns": POW_COLUMNS,
            "entropy_columns": ENTROPY_COLUMNS,
            "asymmetry_columns": ASYMMETRY_COLUMNS,
            "class_order": CLASS_ORDER,
            "label_to_index": LABEL_TO_INDEX,
            "index_to_label": INDEX_TO_LABEL,
            "max_len": int(global_max_len),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "hyperparameters": hp,
            "leave_k_out": int(LEAVE_K_OUT),
            "lkocv_accuracy": float(overall_accuracy),
            "lkocv_f1_macro": float(overall_f1_macro),
        },
        bundle_path,
    )

    with open(output_dir / "saved_model_paths.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "keras_model_path": str(keras_path.resolve()),
                "bundle_path": str(bundle_path.resolve()),
            },
            f,
            indent=2,
        )


# =========================
# LKOCV
# =========================
def run_lkocv():
    output_dir = prepare_output_dir()
    set_all_seeds(SEED)

    raw_df = load_feature_table()
    df = build_feature_table(raw_df)
    X_list, y, groups, trial_ids, image_names, feature_cols, global_max_len = build_trial_sequences(df)

    n_trials = len(X_list)
    n_features = len(feature_cols)
    total_folds = math.comb(n_trials, LEAVE_K_OUT)

    log_lines = []

    def log(msg: str):
        print(msg)
        log_lines.append(str(msg))

    log(f"\nOutput directory: {output_dir.resolve()}")
    log(f"Trials: {n_trials}")
    log(f"Sessions: {len(np.unique(groups))}")
    log(f"Feature dimension: {n_features}")
    log(f"Global max sequence length: {global_max_len}")
    log(f"Target column: {TARGET_COLUMN}")
    log(f"Leave-k-out setting: K = {LEAVE_K_OUT}")
    log(f"Total folds: {total_folds}")
    log("Classification target order: " + ", ".join(CLASS_ORDER))

    splitter = LeavePOut(p=LEAVE_K_OUT)
    dummy_X = np.zeros((n_trials, 1), dtype=np.float32)

    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_prediction_rows = []
    best_epochs = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(dummy_X), start=1):
        log("\n" + "=" * 70)
        log(f"FOLD {fold_idx}/{total_folds}")

        X_train = subset_list_by_indices(X_list, train_idx)
        y_train = y[train_idx]
        groups_train = groups[train_idx]
        trial_ids_train = trial_ids[train_idx]

        X_test = subset_list_by_indices(X_list, test_idx)
        y_test = y[test_idx]
        groups_test = groups[test_idx]
        trial_ids_test = trial_ids[test_idx]
        image_names_test = image_names[test_idx]

        log(f"Train sessions: {np.unique(groups_train)}")
        log(f"Held-out trials: {trial_ids_test.tolist()}")
        log(f"Held-out images: {image_names_test.tolist()}")

        set_all_seeds(SEED + fold_idx)
        fold_out = train_one_lkocv_fold(
            X_train_seq=X_train,
            y_train=y_train,
            X_test_seq=X_test,
            y_test=y_test,
            hp=HYPERPARAMS,
            max_len=global_max_len,
            n_features=n_features,
        )

        preds = fold_out.pop("preds")
        probs = fold_out.pop("probs")
        _ = fold_out.pop("scaler")
        best_epochs.append(fold_out["best_epoch"])

        all_y_true.append(y_test.copy())
        all_y_pred.append(preds.copy())

        for i in range(len(y_test)):
            prob_row = {f"prob_{CLASS_ORDER[j]}": float(probs[i, j]) for j in range(len(CLASS_ORDER))}
            all_prediction_rows.append(
                {
                    "fold": fold_idx,
                    "trial_id": trial_ids_test[i],
                    "session_id": groups_test[i],
                    "image_name": image_names_test[i],
                    "y_true_idx": int(y_test[i]),
                    "y_pred_idx": int(preds[i]),
                    "y_true": INDEX_TO_LABEL[int(y_test[i])],
                    "y_pred": INDEX_TO_LABEL[int(preds[i])],
                    "correct": int(preds[i] == y_test[i]),
                    **prob_row,
                }
            )

        fold_result = {
            "fold": fold_idx,
            "n_train_trials": len(train_idx),
            "n_test_trials": len(test_idx),
            "n_train_sessions": len(np.unique(groups_train)),
            "n_test_sessions": len(np.unique(groups_test)),
            **fold_out,
        }
        fold_results.append(fold_result)

        log("\nFold test results:")
        for k, v in fold_out.items():
            log(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(output_dir / "fold_results.csv", index=False)

    log("\n" + "=" * 70)
    log("FINAL LKOCV RESULTS")
    log(results_df.to_string(index=False))

    numeric_results = results_df.select_dtypes(include=[np.number])
    summary_df = pd.DataFrame(
        {
            "metric": numeric_results.columns,
            "mean": numeric_results.mean(numeric_only=True).values,
            "std": numeric_results.std(numeric_only=True).values,
            "min": numeric_results.min(numeric_only=True).values,
            "max": numeric_results.max(numeric_only=True).values,
        }
    )
    summary_df.to_csv(output_dir / "fold_summary_stats.csv", index=False)

    log("\nMean across folds:")
    log(numeric_results.mean(numeric_only=True).to_string())

    y_true_all = np.concatenate(all_y_true).reshape(-1)
    y_pred_all = np.concatenate(all_y_pred).reshape(-1)

    overall_accuracy = float(accuracy_score(y_true_all, y_pred_all))
    overall_f1_macro = float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0))
    overall_f1_weighted = float(f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(CLASS_ORDER))))
    class_report_text = classification_report(
        y_true_all,
        y_pred_all,
        labels=list(range(len(CLASS_ORDER))),
        target_names=CLASS_ORDER,
        zero_division=0,
    )

    overall_metrics_df = pd.DataFrame(
        [
            {
                "overall_accuracy": overall_accuracy,
                "overall_f1_macro": overall_f1_macro,
                "overall_f1_weighted": overall_f1_weighted,
                "n_trials_total": int(len(y_true_all)),
                "leave_k_out": int(LEAVE_K_OUT),
                "n_folds": int(total_folds),
                "median_best_epoch": int(np.median(best_epochs)),
            }
        ]
    )
    overall_metrics_df.to_csv(output_dir / "overall_prediction_metrics.csv", index=False)

    predictions_df = pd.DataFrame(all_prediction_rows)
    predictions_df.to_csv(output_dir / "all_lkocv_predictions.csv", index=False)

    save_plot_fold_accuracy(predictions_df, output_dir)
    save_plot_confusion_matrix(cm, output_dir)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(class_report_text)

    if SAVE_FINAL_MODEL:
        final_epochs = int(np.median(best_epochs)) if best_epochs else MAX_EPOCHS
        final_out = train_final_model(
            X_list=X_list,
            y=y,
            hp=HYPERPARAMS,
            final_epochs=final_epochs,
            max_len=global_max_len,
            n_features=n_features,
        )

        save_final_model_bundle(
            model=final_out["model"],
            scaler=final_out["scaler"],
            feature_cols=feature_cols,
            global_max_len=global_max_len,
            hp=HYPERPARAMS,
            output_dir=output_dir,
            overall_accuracy=overall_accuracy,
            overall_f1_macro=overall_f1_macro,
        )
        save_plot_final_history(final_out["history"], output_dir)

        final_training_df = pd.DataFrame(
            {
                "trial_id": trial_ids,
                "session_id": groups,
                "image_name": image_names,
                "y_true_idx": y,
                "y_pred_idx": final_out["preds"],
                "y_true": [INDEX_TO_LABEL[int(v)] for v in y],
                "y_pred": [INDEX_TO_LABEL[int(v)] for v in final_out["preds"]],
                "correct": (final_out["preds"] == y).astype(int),
            }
        )
        final_training_df.to_csv(output_dir / "final_training_predictions.csv", index=False)

    log("\nSaved files:")
    log(f"  - {output_dir / 'fold_results.csv'}")
    log(f"  - {output_dir / 'fold_summary_stats.csv'}")
    log(f"  - {output_dir / 'overall_prediction_metrics.csv'}")
    log(f"  - {output_dir / 'all_lkocv_predictions.csv'}")
    log(f"  - {output_dir / 'classification_report.txt'}")
    log(f"  - {output_dir / 'fold_accuracy.png'}")
    log(f"  - {output_dir / 'confusion_matrix.png'}")
    if SAVE_FINAL_MODEL:
        log(f"  - {output_dir / 'final_training_history.png'}")
        log(f"  - {output_dir / 'final_training_predictions.csv'}")

    save_text_summary("\n".join(log_lines), output_dir, filename="run_summary.txt")

    summary_json = {
        "input_feature_table": str(resolve_input_feature_table().resolve()),
        "n_trials": int(n_trials),
        "n_sessions": int(len(np.unique(groups))),
        "feature_dim": int(n_features),
        "global_max_sequence_length": int(global_max_len),
        "leave_k_out": int(LEAVE_K_OUT),
        "n_folds": int(total_folds),
        "overall_accuracy": overall_accuracy,
        "overall_f1_macro": overall_f1_macro,
        "overall_f1_weighted": overall_f1_weighted,
        "median_best_epoch": int(np.median(best_epochs)) if best_epochs else None,
        "class_order": CLASS_ORDER,
        "hyperparameters": HYPERPARAMS,
        "entropy_backend": "eegproc" if eeg is not None else "fallback_shannon",
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(json.dumps(summary_json, indent=2))
    print("\nClassification report:\n")
    print(class_report_text)

    return results_df


if __name__ == "__main__":
    run_lkocv()
