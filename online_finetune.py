from __future__ import annotations

import argparse
import csv
import json
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from model import (
    ASYMMETRY_COLUMNS,
    CHANNELS,
    DEFAULT_FEATURE_COLUMNS,
    ENTROPY_COLUMNS,
    POW_COLUMNS,
    QuadrantModel,
)

CLASS_ORDER = ["LALV", "LAHV", "HAHV", "HALV"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(CLASS_ORDER)}
SAMPLE_TIME_COLUMN = "sample_timestamp"
SEED = 42
MAX_EPOCHS_CAP = 14
BASE_DIR = Path(__file__).resolve().parent
ONLINE_DIR = BASE_DIR / "online_state"
ONLINE_DIR.mkdir(parents=True, exist_ok=True)
STATUS_PATH = ONLINE_DIR / "training_status.json"
LOG_PATH = ONLINE_DIR / "finetune_log.csv"
ONLINE_MODEL_PATH = ONLINE_DIR / "online_model.keras"
ONLINE_BUNDLE_PATH = ONLINE_DIR / "online_model_bundle.joblib"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def write_status(payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_log(row: dict[str, Any]) -> None:
    file_exists = LOG_PATH.exists() and LOG_PATH.stat().st_size > 0
    with LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def resolve_base_bundle_path() -> Path:
    user_bundle = ONLINE_BUNDLE_PATH
    if user_bundle.exists():
        return user_bundle
    candidates = [
        BASE_DIR / "models" / "quadrant_bilstm_lkocv_bundle.joblib",
        BASE_DIR / "models" / "quadrant_bilstm_classifier_bundle.joblib",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find a base BiLSTM bundle to fine-tune.")


def load_current_session(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if SAMPLE_TIME_COLUMN in df.columns:
        df[SAMPLE_TIME_COLUMN] = pd.to_datetime(df[SAMPLE_TIME_COLUMN], errors="coerce", utc=True)
    else:
        raise ValueError("current_session.csv must include sample_timestamp for online fine-tuning.")
    df["trial_index"] = pd.to_numeric(df["trial_index"], errors="coerce")
    df["trial_id"] = df["session_id"].astype(str) + "__" + df["trial_index"].astype("Int64").astype(str)
    df["user_predicted_code"] = df["user_predicted_code"].astype(str).str.upper().str.strip()
    df["__row_order__"] = np.arange(len(df))
    sort_keys = ["trial_index", SAMPLE_TIME_COLUMN, "__row_order__"]
    return df.sort_values(sort_keys, kind="stable").reset_index(drop=True)


def build_sequences(df: pd.DataFrame) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, int, list[str]]:
    qm = QuadrantModel(resolve_base_bundle_path())
    feature_names = list(qm.feature_names)
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    trial_ids: list[str] = []
    for trial_id, g in df.groupby("trial_id", sort=False):
        label = str(g["user_predicted_code"].iloc[0]).upper().strip()
        if label not in LABEL_TO_INDEX:
            continue
        seq_df = qm.build_windowed_feature_sequence(g)
        if seq_df.empty:
            continue
        sequences.append(seq_df[feature_names].to_numpy(dtype=np.float32))
        labels.append(LABEL_TO_INDEX[label])
        trial_ids.append(str(trial_id))
    if not sequences:
        raise ValueError("No valid trial sequences were built from current_session.csv")
    max_len = max(len(seq) for seq in sequences)
    return sequences, np.asarray(labels, dtype=np.int32), np.asarray(trial_ids), max_len, feature_names


def fit_scaler(seqs: Sequence[np.ndarray]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(np.vstack(seqs))
    return scaler


def transform_sequences(seqs: Sequence[np.ndarray], scaler: StandardScaler) -> list[np.ndarray]:
    return [scaler.transform(seq).astype(np.float32) for seq in seqs]


def pad_sequence_list(seqs: Sequence[np.ndarray], max_len: int, n_features: int) -> np.ndarray:
    x = np.zeros((len(seqs), max_len, n_features), dtype=np.float32)
    for i, seq in enumerate(seqs):
        seq_len = min(len(seq), max_len)
        x[i, :seq_len, :] = seq[:seq_len]
    return x


def choose_validation_index(labels: np.ndarray) -> int:
    if len(labels) <= 1:
        return 0
    counts = pd.Series(labels).value_counts()
    candidate_classes = [cls for cls, count in counts.items() if count > 1]
    if candidate_classes:
        chosen_class = int(counts.loc[candidate_classes].idxmax())
        return int(np.where(labels == chosen_class)[0][-1])
    return len(labels) - 1


def hyperparameter_schedule(completed_rounds: int) -> list[dict[str, Any]]:
    stage = max(1, completed_rounds // 5)
    configs = [
        {"learning_rate": 5e-4, "batch_size": 4, "epochs": 4},
    ]
    if stage >= 2:
        configs.extend([
            {"learning_rate": 1e-4, "batch_size": 4, "epochs": 6},
            {"learning_rate": 5e-4, "batch_size": 8, "epochs": 6},
        ])
    if stage >= 3:
        configs.extend([
            {"learning_rate": 2e-4, "batch_size": 4, "epochs": 8},
            {"learning_rate": 1e-4, "batch_size": 8, "epochs": 8},
        ])
    if stage >= 4:
        configs.extend([
            {"learning_rate": 5e-5, "batch_size": 4, "epochs": 10},
            {"learning_rate": 2e-4, "batch_size": 8, "epochs": 10},
        ])
    for cfg in configs:
        cfg["epochs"] = min(int(cfg["epochs"]), MAX_EPOCHS_CAP)
    return configs


def clone_frozen_dense_tunable_model(bundle_path: Path) -> tf.keras.Model:
    bundle = joblib.load(bundle_path)
    keras_model_path = bundle.get("keras_model_path")
    if not keras_model_path:
        raise RuntimeError("Bundle missing keras_model_path.")
    keras_model_path = Path(keras_model_path)
    if not keras_model_path.is_absolute():
        keras_model_path = (bundle_path.parent / keras_model_path).resolve()
    model = tf.keras.models.load_model(keras_model_path)
    for layer in model.layers:
        layer.trainable = False
        if isinstance(layer, tf.keras.layers.Dense):
            layer.trainable = True
    return model


def train_one_config(
    bundle_path: Path,
    train_seqs: list[np.ndarray],
    train_y: np.ndarray,
    val_seqs: list[np.ndarray],
    val_y: np.ndarray,
    max_len: int,
    n_features: int,
    config: dict[str, Any],
    seed: int,
) -> tuple[float, dict[str, Any], StandardScaler, tf.keras.Model]:
    set_all_seeds(seed)
    scaler = fit_scaler(train_seqs)
    x_train = pad_sequence_list(transform_sequences(train_seqs, scaler), max_len, n_features)
    x_val = pad_sequence_list(transform_sequences(val_seqs, scaler), max_len, n_features)

    model = clone_frozen_dense_tunable_model(bundle_path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(config["learning_rate"])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=0)
    ]
    history = model.fit(
        x_train,
        train_y,
        validation_data=(x_val, val_y),
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        verbose=0,
        callbacks=callbacks,
    )
    best_val = float(np.min(history.history["val_loss"]))
    info = dict(config)
    info["best_val_loss"] = best_val
    info["best_epoch"] = int(np.argmin(history.history["val_loss"]) + 1)
    return best_val, info, scaler, model


def train_final_model(
    source_bundle_path: Path,
    sequences: list[np.ndarray],
    labels: np.ndarray,
    max_len: int,
    n_features: int,
    config: dict[str, Any],
) -> tuple[tf.keras.Model, StandardScaler]:
    set_all_seeds(SEED)
    scaler = fit_scaler(sequences)
    x_all = pad_sequence_list(transform_sequences(sequences, scaler), max_len, n_features)
    model = clone_frozen_dense_tunable_model(source_bundle_path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(config["learning_rate"])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        x_all,
        labels,
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        verbose=0,
    )
    return model, scaler


def save_online_bundle(model: tf.keras.Model, scaler: StandardScaler, feature_names: list[str], max_len: int, config: dict[str, Any], rounds: int, n_trials: int) -> None:
    model.save(ONLINE_MODEL_PATH)
    payload = {
        "model_type": "keras_bilstm_classifier",
        "keras_model_path": str(ONLINE_MODEL_PATH.resolve()),
        "feature_names": feature_names,
        "pow_columns": POW_COLUMNS,
        "entropy_columns": ENTROPY_COLUMNS,
        "asymmetry_columns": ASYMMETRY_COLUMNS,
        "class_order": CLASS_ORDER,
        "label_to_index": LABEL_TO_INDEX,
        "index_to_label": {idx: label for label, idx in LABEL_TO_INDEX.items()},
        "max_len": int(max_len),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "model_kind": "online_dense_finetune",
        "completed_rounds": int(rounds),
        "n_trials_seen": int(n_trials),
        "hyperparameters": config,
    }
    joblib.dump(payload, ONLINE_BUNDLE_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-count", type=int, required=True)
    parser.add_argument("--current-session", type=str, default=str(BASE_DIR / "current_session.csv"))
    args = parser.parse_args()

    current_session_path = Path(args.current_session)
    write_status({"status": "running", "round_count": args.round_count, "message": "Fine-tuning started."})

    df = load_current_session(current_session_path)
    sequences, labels, trial_ids, max_len, feature_names = build_sequences(df)
    n_trials = len(sequences)
    if n_trials < 2:
        raise ValueError("Need at least 2 labeled trials in current_session.csv before fine-tuning.")

    val_idx = choose_validation_index(labels)
    train_seqs = [seq for i, seq in enumerate(sequences) if i != val_idx]
    train_y = labels[np.arange(len(labels)) != val_idx]
    val_seqs = [sequences[val_idx]]
    val_y = np.asarray([labels[val_idx]], dtype=np.int32)
    if len(train_seqs) == 0:
        raise ValueError("Fine-tuning needs at least one training sequence after validation split.")

    bundle_path = resolve_base_bundle_path()
    configs = hyperparameter_schedule(args.round_count)
    best_val = float("inf")
    best_info: dict[str, Any] = {}
    best_source_bundle = bundle_path

    for idx, config in enumerate(configs, start=1):
        val_loss, info, _scaler, _model = train_one_config(
            bundle_path=bundle_path,
            train_seqs=train_seqs,
            train_y=train_y,
            val_seqs=val_seqs,
            val_y=val_y,
            max_len=max_len,
            n_features=len(feature_names),
            config=config,
            seed=SEED + idx,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_info = info

    final_model, final_scaler = train_final_model(
        source_bundle_path=bundle_path,
        sequences=sequences,
        labels=labels,
        max_len=max_len,
        n_features=len(feature_names),
        config=best_info,
    )
    save_online_bundle(final_model, final_scaler, feature_names, max_len, best_info, args.round_count, n_trials)

    status = {
        "status": "completed",
        "round_count": args.round_count,
        "n_trials": int(n_trials),
        "best_config": best_info,
        "bundle_path": str(ONLINE_BUNDLE_PATH.resolve()),
        "message": "Fine-tuning completed.",
    }
    write_status(status)
    append_log({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "round_count": args.round_count,
        "n_trials": n_trials,
        "search_size": len(configs),
        "best_val_loss": best_info.get("best_val_loss", float("nan")),
        "best_epoch": best_info.get("best_epoch", ""),
        "learning_rate": best_info.get("learning_rate", ""),
        "batch_size": best_info.get("batch_size", ""),
        "epochs": best_info.get("epochs", ""),
    })

if __name__ == "__main__":
    main()
