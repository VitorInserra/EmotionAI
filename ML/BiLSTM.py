from __future__ import annotations

import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from utils import compute_asymmetry_from_psd

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
SAMPLE_TIME_COLUMN = "sample_timestamp"
TRIAL_START_COLUMN = "trial_started_at"
TRIAL_END_COLUMN = "trial_ended_at"

CLASS_ORDER = ["LALV", "LAHV", "HAHV", "HALV"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(CLASS_ORDER)}
INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}

LEAVE_K_OUT = 15  # fold size in held-out pictures, not combinatorial leave-p-out
MAX_EPOCHS = 250
EARLY_STOPPING_PATIENCE = 30
SEED = 42
VERBOSE_FIT = 0
FS = 128

OUTPUT_ROOT = Path("outputs")
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_OUTPUT_DIR = OUTPUT_ROOT / f"lkocv_bilstm_{RUN_STAMP}"
MODEL_DIR = Path("models")
SAVE_FINAL_MODEL = True

HYPERPARAMS = {
    "lstm_units": 64,
    "dense_units": 32,
    "dropout": 0.30,
    "learning_rate": 1e-4,
    "batch_size": 8,
}

CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]
BANDS = ["theta", "alpha", "betaL", "betaH", "gamma"]
POW_COLUMNS = [f"{ch}_{band}" for ch in CHANNELS for band in BANDS]

META_COLUMNS = {
    GROUP_COLUMN,
    SEQUENCE_ID_COLUMN,
    TIME_COLUMN,
    IMAGE_COLUMN,
    TRIAL_START_COLUMN,
    TRIAL_END_COLUMN,
    "time_elapsed_seconds",
    SAMPLE_TIME_COLUMN,
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
    with open(output_dir / filename, "w", encoding="utf-8") as f:
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
    plt.xlabel(f"Fold (held-out size = {LEAVE_K_OUT})")
    plt.ylabel("Accuracy")
    plt.title("BiLSTM Non-Overlapping Fold Accuracy")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "fold_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_plot_confusion_matrix(cm: np.ndarray, output_dir: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("BiLSTM Confusion Matrix")
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
    if "accuracy" in history.history:
        plt.plot(history.history["accuracy"], label="train_accuracy")
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
        "Could not find base dataset. Looked for: " + ", ".join(INPUT_FEATURE_TABLE_CANDIDATES)
    )


def _deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df
    return df.loc[:, ~df.columns.duplicated()].copy()


def _normalize_timestamp_column(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _build_trial_key_strings(df: pd.DataFrame) -> pd.Series:
    image_key = df[IMAGE_COLUMN].fillna("missing_image").astype(str)
    start_key = df[TRIAL_START_COLUMN].dt.strftime("%Y%m%dT%H%M%S.%fZ").fillna("missing_start")
    end_key = df[TRIAL_END_COLUMN].dt.strftime("%Y%m%dT%H%M%S.%fZ").fillna("missing_end")
    return image_key + "__" + start_key + "__" + end_key


def load_feature_table() -> pd.DataFrame:
    path = resolve_input_feature_table()

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        has_sample_timestamp = SAMPLE_TIME_COLUMN in header

        if has_sample_timestamp:
            expanded_header = header
        else:
            expanded_header = header[:6] + [SAMPLE_TIME_COLUMN] + header[6:]

        normalized_rows: list[list[str]] = []
        for row in reader:
            if has_sample_timestamp:
                if len(row) != len(header):
                    raise ValueError(
                        f"Unexpected row length {len(row)} in {path}. Expected {len(header)}. Row head: {row[:10]}"
                    )
                normalized_rows.append(row)
            else:
                if len(row) == len(header):
                    row = row[:6] + [""] + row[6:]
                elif len(row) != len(header) + 1:
                    raise ValueError(
                        f"Unexpected row length {len(row)} in {path}. Expected {len(header)} or {len(header)+1}. Row head: {row[:10]}"
                    )
                normalized_rows.append(row)

    df = pd.DataFrame(normalized_rows, columns=expanded_header)
    df = _deduplicate_columns(df)
    df["__row_order__"] = np.arange(len(df))

    numeric_cols = ["time_elapsed_seconds", "sensor_contact_quality", *POW_COLUMNS]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in [SAMPLE_TIME_COLUMN, TRIAL_START_COLUMN, TRIAL_END_COLUMN]:
        if col in df.columns:
            df[col] = _normalize_timestamp_column(df[col])

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.upper().str.strip()
    df[SEQUENCE_ID_COLUMN] = _build_trial_key_strings(df)

    sort_keys = [SEQUENCE_ID_COLUMN]
    if SAMPLE_TIME_COLUMN in df.columns:
        sort_keys.append(SAMPLE_TIME_COLUMN)
    sort_keys.append("__row_order__")
    df = df.sort_values(sort_keys, kind="stable").reset_index(drop=True)
    df[TIME_COLUMN] = df.groupby(SEQUENCE_ID_COLUMN).cumcount()
    return df


def build_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    missing = [c for c in POW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing PSD columns: {missing[:10]}")

    psd = (
        df[POW_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    entropy = eeg.shannons_entropy(psd, fs=FS)
    if not isinstance(entropy, pd.DataFrame):
        raise TypeError("eegproc.shannons_entropy did not return a DataFrame.")
    entropy = entropy.reset_index(drop=True)

    asymm = compute_asymmetry_from_psd(psd).reset_index(drop=True)

    feature_frame = df.reset_index(drop=True).copy()
    feature_frame[POW_COLUMNS] = psd.reset_index(drop=True)
    feature_frame = pd.concat([feature_frame, entropy, asymm], axis=1)
    feature_frame = _deduplicate_columns(feature_frame).fillna(0.0)

    feature_cols = [c for c in feature_frame.columns if c not in META_COLUMNS]
    if not feature_cols:
        raise ValueError("No feature columns found after excluding metadata columns.")

    return feature_frame, feature_cols, list(entropy.columns), list(asymm.columns)


def build_trial_sequences(df: pd.DataFrame, feature_cols: list[str]):
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    groups_list: list[str] = []
    trial_ids_list: list[str] = []
    image_names_list: list[str] = []

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

        X_list.append(x)
        y_list.append(int(LABEL_TO_INDEX[label_text]))
        groups_list.append(str(g[GROUP_COLUMN].iloc[0]))
        trial_ids_list.append(str(trial_id))
        image_names_list.append(str(g[IMAGE_COLUMN].iloc[0]))

    if not X_list:
        raise ValueError("No valid trial sequences found.")

    y = np.asarray(y_list, dtype=np.int32)
    groups = np.asarray(groups_list)
    trial_ids = np.asarray(trial_ids_list)
    image_names = np.asarray(image_names_list)
    global_max_len = max(seq.shape[0] for seq in X_list)

    return X_list, y, groups, trial_ids, image_names, global_max_len


def subset_list_by_indices(lst, indices):
    return [lst[i] for i in indices]


def fit_feature_scaler(seq_list: List[np.ndarray]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(np.vstack(seq_list))
    return scaler


def transform_sequence_list(seq_list: List[np.ndarray], scaler: StandardScaler) -> List[np.ndarray]:
    return [scaler.transform(seq).astype(np.float32) for seq in seq_list]


def pad_sequence_list(seq_list: List[np.ndarray], max_len: int, n_features: int) -> np.ndarray:
    X = np.zeros((len(seq_list), max_len, n_features), dtype=np.float32)
    for i, seq in enumerate(seq_list):
        seq_len = min(seq.shape[0], max_len)
        X[i, :seq_len, :] = seq[:seq_len, :]
    return X


def make_non_overlapping_folds(
    fold_keys: np.ndarray,
    y: np.ndarray,
    fold_size: int,
    seed: int = SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Non-overlapping fold builder.

    If you pass image_names as fold_keys:
        - all rows/trials from the same image stay together
        - 45 unique pictures with fold_size=15 -> 3 folds

    If you want session-level isolation instead, pass groups instead of image_names.
    """
    if fold_size < 1:
        raise ValueError("fold_size must be >= 1")

    fold_keys = np.asarray(fold_keys).astype(str)
    all_indices = np.arange(len(fold_keys))

    key_to_indices: dict[str, list[int]] = {}
    for idx, key in enumerate(fold_keys):
        key_to_indices.setdefault(key, []).append(idx)

    unique_keys = np.asarray(list(key_to_indices.keys()), dtype=object)
    n_keys = len(unique_keys)
    n_folds = math.ceil(n_keys / fold_size)

    key_labels = np.asarray(
        [int(pd.Series(y[key_to_indices[key]]).mode().iloc[0]) for key in unique_keys],
        dtype=np.int32,
    )

    rng = np.random.default_rng(seed)
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    fold_counts = [0] * n_folds

    for cls in rng.permutation(np.unique(key_labels)):
        cls_keys = unique_keys[key_labels == cls].copy()
        rng.shuffle(cls_keys)

        for key in cls_keys:
            available = [i for i in range(n_folds) if fold_counts[i] < fold_size]
            if not available:
                available = list(range(n_folds))

            target_fold = min(available, key=lambda i: fold_counts[i])
            folds[target_fold].append(str(key))
            fold_counts[target_fold] += 1

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_test_keys in folds:
        if not fold_test_keys:
            continue

        test_mask = np.isin(fold_keys, np.asarray(fold_test_keys, dtype=object))
        test_idx = all_indices[test_mask]
        train_idx = all_indices[~test_mask]

        if len(test_idx) == 0 or len(train_idx) == 0:
            continue

        splits.append((train_idx, test_idx))

    return splits


# =========================
# MODEL
# =========================
def build_bilstm_classifier(max_len: int, n_features: int, hp: dict) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(max_len, n_features), name="sequence_input")
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            hp["lstm_units"],
            return_sequences=False,
            dropout=hp["dropout"] * 0.5,
            recurrent_dropout=0.0,
        )
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(hp["dropout"])(x)
    x = tf.keras.layers.Dense(hp["dense_units"], activation="relu")(x)
    x = tf.keras.layers.Dropout(hp["dropout"])(x)
    outputs = tf.keras.layers.Dense(len(CLASS_ORDER), activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp["learning_rate"]),
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
    scaler = fit_feature_scaler(X_train_seq)

    X_train_scaled = transform_sequence_list(X_train_seq, scaler)
    X_test_scaled = transform_sequence_list(X_test_seq, scaler)

    X_train_pad = pad_sequence_list(X_train_scaled, max_len=max_len, n_features=n_features)
    X_test_pad = pad_sequence_list(X_test_scaled, max_len=max_len, n_features=n_features)

    model = build_bilstm_classifier(max_len=max_len, n_features=n_features, hp=hp)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=1e-4,
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train_pad,
        y_train,
        epochs=MAX_EPOCHS,
        batch_size=hp["batch_size"],
        verbose=VERBOSE_FIT,
        callbacks=callbacks,
        class_weight=compute_class_weights(y_train),
        shuffle=True,
    )

    probs = model.predict(X_test_pad, verbose=0)
    preds = probs.argmax(axis=1)

    train_losses = history.history["loss"]
    best_epoch = int(np.argmin(train_losses) + 1)
    best_train_loss = float(np.min(train_losses))
    final_train_loss = float(train_losses[-1])
    final_train_accuracy = (
        float(history.history["accuracy"][-1])
        if "accuracy" in history.history else float("nan")
    )

    return {
        "best_train_loss": best_train_loss,
        "final_train_loss": final_train_loss,
        "final_train_accuracy": final_train_accuracy,
        "best_epoch": best_epoch,
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "test_f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
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
    scaler = fit_feature_scaler(X_list)
    X_all_scaled = transform_sequence_list(X_list, scaler)
    X_all_pad = pad_sequence_list(X_all_scaled, max_len=max_len, n_features=n_features)

    model = build_bilstm_classifier(max_len=max_len, n_features=n_features, hp=hp)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=1e-4,
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_all_pad,
        y,
        epochs=final_epochs,
        batch_size=hp["batch_size"],
        verbose=VERBOSE_FIT,
        callbacks=callbacks,
        class_weight=compute_class_weights(y),
        shuffle=True,
    )

    probs = model.predict(X_all_pad, verbose=0)
    preds = probs.argmax(axis=1)

    return {
        "model": model,
        "scaler": scaler,
        "history": history,
        "preds": preds,
        "probs": probs,
        "train_accuracy": float(accuracy_score(y, preds)),
        "train_f1_macro": float(f1_score(y, preds, average="macro", zero_division=0)),
    }


def save_final_model_bundle(
    model: tf.keras.Model,
    scaler: StandardScaler,
    feature_cols: list[str],
    entropy_columns: list[str],
    asymmetry_columns: list[str],
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
            "entropy_columns": entropy_columns,
            "asymmetry_columns": asymmetry_columns,
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
    df, feature_cols, entropy_columns, asymmetry_columns = build_feature_table(raw_df)
    X_list, y, groups, trial_ids, image_names, global_max_len = build_trial_sequences(df, feature_cols)

    n_trials = len(X_list)
    if LEAVE_K_OUT < 1:
        raise ValueError("LEAVE_K_OUT must be >= 1.")

    n_features = len(feature_cols)

    # Build non-overlapping folds by image_name
    # 45 unique images with LEAVE_K_OUT=15 -> 3 folds
    split_indices = make_non_overlapping_folds(
        fold_keys=image_names,
        y=y,
        fold_size=LEAVE_K_OUT,
        seed=SEED,
    )
    total_folds = len(split_indices)

    if total_folds == 0:
        raise ValueError("No valid folds were created.")

    log_lines: list[str] = []

    def log(msg: str):
        print(msg)
        log_lines.append(str(msg))

    log(f"\nOutput directory: {output_dir.resolve()}")
    log(f"Trials: {n_trials}")
    log(f"Sessions: {len(np.unique(groups))}")
    log(f"Unique images: {len(np.unique(image_names))}")
    log(f"Feature dimension: {n_features}")
    log(f"Global max sequence length: {global_max_len}")
    log(f"Target column: {TARGET_COLUMN}")
    log(f"Fold size (held-out images): {LEAVE_K_OUT}")
    log(f"Total folds: {total_folds}")
    log("Classification target order: " + ", ".join(CLASS_ORDER))

    fold_results: list[dict] = []
    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    all_prediction_rows: list[dict] = []
    best_epochs: list[int] = []

    for fold_idx, (train_idx, test_idx) in enumerate(split_indices, start=1):
        log("\n" + "=" * 70)
        log(f"FOLD {fold_idx}/{total_folds}")

        X_train = subset_list_by_indices(X_list, train_idx)
        y_train = y[train_idx]
        groups_train = groups[train_idx]
        images_train = image_names[train_idx].astype(str)

        X_test = subset_list_by_indices(X_list, test_idx)
        y_test = y[test_idx]
        groups_test = groups[test_idx]
        trial_ids_test = trial_ids[test_idx]
        image_names_test = image_names[test_idx].astype(str)

        # Hard leakage guard on image names
        overlap = set(images_train).intersection(set(image_names_test))
        if overlap:
            raise RuntimeError(f"Image leakage detected across train/test: {sorted(overlap)[:10]}")

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
    log("FINAL RESULTS")
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
            entropy_columns=entropy_columns,
            asymmetry_columns=asymmetry_columns,
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
        "n_unique_images": int(len(np.unique(image_names))),
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
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(json.dumps(summary_json, indent=2))
    print("\nClassification report:\n")
    print(class_report_text)

    return results_df


if __name__ == "__main__":
    run_lkocv()