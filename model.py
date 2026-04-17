from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import eegproc as eeg

from ML.utils import HOMOLOGOUS_PAIRS, compute_asymmetry_from_psd

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
POW_COLUMNS = [f"{channel}_{band}" for channel in CHANNELS for band in BANDS]
ENTROPY_COLUMNS = [f"{channel}_{band}_entropy" for channel in CHANNELS for band in BANDS]
ASYMMETRY_COLUMNS = [
    f"{right}_{left}_{band}_{kind}"
    for left, right in HOMOLOGOUS_PAIRS
    for band in BANDS
    for kind in ("da", "ra")
]
DEFAULT_FEATURE_COLUMNS = POW_COLUMNS + ENTROPY_COLUMNS + ASYMMETRY_COLUMNS

WINDOW_SECONDS = 2.0
WINDOW_OVERLAP_SECONDS = 0.5
WINDOW_STEP_SECONDS = WINDOW_SECONDS - WINDOW_OVERLAP_SECONDS

PREDICTION_MAP = {
    "1": "LALV",
    "2": "LAHV",
    "3": "HAHV",
    "4": "HALV",
    "LALV": "LALV",
    "LAHV": "LAHV",
    "HAHV": "HAHV",
    "HALV": "HALV",
    "LOW AROUSAL / LOW VALENCE": "LALV",
    "LOW AROUSAL / HIGH VALENCE": "LAHV",
    "HIGH AROUSAL / HIGH VALENCE": "HAHV",
    "LOW VALENCE / HIGH AROUSAL": "HALV",
}


class QuadrantModel:
    """Strict inference wrapper for the saved quadrant model.

    Behavior:
      - requires a valid saved artifact at construction time
      - requires eegproc for Shannon entropy featurization
      - raises immediately on missing or malformed artifacts
      - supports sklearn bundles and keras BiLSTM bundles
    """

    def __init__(self, artifact_path: str | Path | None = None) -> None:
        self.artifact_path = self._resolve_artifact_path(artifact_path)
        self.bundle = self._load_bundle(self.artifact_path)

        self.model_type = str(self.bundle.get("model_type", "sklearn"))
        self.feature_names = list(self.bundle.get("feature_names", DEFAULT_FEATURE_COLUMNS))
        self.class_order = list(self.bundle.get("class_order", ["LALV", "LAHV", "HAHV", "HALV"]))
        self.max_len = int(self.bundle.get("max_len", 1))

        self.scaler: Any | None = self.bundle.get("scaler")
        self.scaler_mean_: np.ndarray | None = None
        self.scaler_scale_: np.ndarray | None = None

        if self.model_type.startswith("keras_bilstm"):
            keras_model_path = self.bundle.get("keras_model_path")
            if not keras_model_path:
                raise ValueError("Keras bundle is missing 'keras_model_path'.")
            keras_path = Path(keras_model_path)
            if not keras_path.is_absolute():
                keras_path = (self.artifact_path.parent / keras_path).resolve()
            if not keras_path.exists():
                raise FileNotFoundError(f"Keras model file not found: {keras_path}")
            self.model = tf.keras.models.load_model(keras_path)
            self.scaler_mean_ = np.asarray(self.bundle["scaler_mean"], dtype=np.float32)
            self.scaler_scale_ = np.asarray(self.bundle["scaler_scale"], dtype=np.float32)
        else:
            self.model = self.bundle.get("model") or self.bundle.get("estimator")
            if self.model is None:
                raise ValueError("Sklearn bundle is missing 'model' or 'estimator'.")
            if not hasattr(self.model, "predict"):
                raise TypeError("Loaded estimator does not provide predict().")

    def _resolve_artifact_path(self, artifact_path: str | Path | None) -> Path:
        if artifact_path is not None:
            path = Path(artifact_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Model artifact not found: {path}")
            return path

        env_path = os.environ.get("MODEL_ARTIFACT")
        if env_path:
            path = Path(env_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"MODEL_ARTIFACT points to a missing file: {path}")
            return path

        base_dir = Path(__file__).resolve().parent
        candidates = [
            base_dir / "online_state" / "online_model_bundle.joblib",
            base_dir / "models" / "quadrant_bilstm_lkocv_bundle.joblib",
            base_dir / "models" / "quadrant_bilstm_classifier_bundle.joblib",
            base_dir / "models" / "quadrant_model.joblib",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        raise FileNotFoundError(
            "No model artifact found. Expected one of: "
            f"{', '.join(str(path) for path in candidates)}"
        )

    def _load_bundle(self, artifact_path: Path) -> dict[str, Any]:
        artifact = joblib.load(artifact_path)
        if isinstance(artifact, dict):
            return artifact
        return {"model_type": "sklearn", "model": artifact, "feature_names": DEFAULT_FEATURE_COLUMNS}

    def _coerce_psd_frame(
        self,
        psd_input: Mapping[str, float] | Sequence[Mapping[str, float]] | pd.DataFrame,
    ) -> pd.DataFrame:
        if isinstance(psd_input, pd.DataFrame):
            frame = psd_input.copy()
        elif isinstance(psd_input, Mapping):
            frame = pd.DataFrame([dict(psd_input)])
        else:
            frame = pd.DataFrame([dict(row) for row in psd_input])

        if frame.empty:
            raise ValueError("Received empty PSD input.")

        for column in POW_COLUMNS:
            if column not in frame.columns:
                raise ValueError(f"PSD input is missing required column: {column}")

        work = frame[POW_COLUMNS].apply(pd.to_numeric, errors="raise").copy()

        if "timestamp" in frame.columns:
            work["timestamp"] = pd.to_numeric(frame["timestamp"], errors="raise")
        elif "sample_timestamp" in frame.columns:
            sample_ts = frame["sample_timestamp"]
            if pd.api.types.is_numeric_dtype(sample_ts):
                work["timestamp"] = pd.to_numeric(sample_ts, errors="raise")
            else:
                ts = pd.to_datetime(sample_ts, errors="raise", utc=True)
                work["timestamp"] = ts.astype("int64") / 1e9

        return work

    def build_windowed_psd_sequence(
        self,
        psd_input: Mapping[str, float] | Sequence[Mapping[str, float]] | pd.DataFrame,
        window_seconds: float = WINDOW_SECONDS,
        overlap_seconds: float = WINDOW_OVERLAP_SECONDS,
    ) -> pd.DataFrame:
        work = self._coerce_psd_frame(psd_input)

        if "timestamp" not in work.columns or len(work) == 1:
            averaged = work[POW_COLUMNS].mean(axis=0, numeric_only=True).to_frame().T
            averaged["window_start"] = np.nan
            averaged["window_end"] = np.nan
            averaged["n_samples"] = len(work)
            return averaged

        if overlap_seconds >= window_seconds:
            raise ValueError("overlap_seconds must be smaller than window_seconds.")

        times = pd.to_numeric(work["timestamp"], errors="raise").to_numpy(dtype=np.float64)
        start_time = float(times.min())
        end_time = float(times.max())
        step = window_seconds - overlap_seconds

        if end_time - start_time <= window_seconds:
            window_starts = [start_time]
        else:
            last_full_start = end_time - window_seconds
            window_starts = list(np.arange(start_time, last_full_start + 1e-9, step))
            final_start = max(start_time, end_time - window_seconds)
            if not window_starts or abs(window_starts[-1] - final_start) > 1e-6:
                window_starts.append(final_start)

        rows: list[dict[str, float]] = []
        for window_start in window_starts:
            window_end = window_start + window_seconds
            mask = (times >= window_start) & (times <= window_end)
            window = work.loc[mask, POW_COLUMNS]
            if window.empty:
                continue
            means = window.mean(axis=0, numeric_only=True)
            row = {column: float(means[column]) for column in POW_COLUMNS}
            row["window_start"] = float(window_start)
            row["window_end"] = float(window_end)
            row["n_samples"] = int(len(window))
            rows.append(row)

        if not rows:
            raise ValueError("No valid PSD windows were produced from the input.")

        return pd.DataFrame(rows)

    def _align_feature_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        aligned = frame.copy()
        missing_cols = [col for col in self.feature_names if col not in aligned.columns]
        for col in missing_cols:
            aligned[col] = 0.0
        return aligned[self.feature_names].fillna(0.0)

    def featurize_psd(
        self,
        psd_input: Mapping[str, float] | Sequence[Mapping[str, float]] | pd.DataFrame,
    ) -> pd.DataFrame:
        windowed_psd = self.build_windowed_psd_sequence(psd_input)
        psd_only = windowed_psd[POW_COLUMNS]
        entropy_frame = eeg.shannons_entropy(psd_only, fs=128)
        asymmetry_frame = compute_asymmetry_from_psd(psd_only)
        feature_frame = pd.concat([psd_only, entropy_frame, asymmetry_frame], axis=1)
        return self._align_feature_columns(feature_frame)

    def _scale_sequence(self, feature_frame: pd.DataFrame) -> np.ndarray:
        frame = self._align_feature_columns(feature_frame)
        if self.scaler is not None:
            return np.asarray(self.scaler.transform(frame), dtype=np.float32)
        if self.scaler_mean_ is None or self.scaler_scale_ is None:
            return frame.to_numpy(dtype=np.float32)
        scale = np.where(self.scaler_scale_ == 0.0, 1.0, self.scaler_scale_)
        arr = frame.to_numpy(dtype=np.float32)
        return ((arr - self.scaler_mean_) / scale).astype(np.float32)

    def predict_code(
        self,
        psd_input: Mapping[str, float] | Sequence[Mapping[str, float]] | pd.DataFrame,
    ) -> str:
        feature_frame = self.featurize_psd(psd_input)

        if self.model_type.startswith("keras_bilstm"):
            scaled_seq = self._scale_sequence(feature_frame)
            max_len = max(1, int(self.max_len))
            n_features = scaled_seq.shape[1]
            x = np.zeros((1, max_len, n_features), dtype=np.float32)
            seq_len = min(len(scaled_seq), max_len)
            x[0, :seq_len, :] = scaled_seq[:seq_len]
            probs = self.model.predict(x, verbose=0)
            pred_idx = int(np.argmax(probs[0]))
            if pred_idx < 0 or pred_idx >= len(self.class_order):
                raise ValueError(f"Predicted class index out of range: {pred_idx}")
            prediction = self.class_order[pred_idx]
        else:
            aggregated = feature_frame.mean(axis=0, numeric_only=True).to_frame().T
            x: Any = aggregated
            if self.scaler is not None:
                x = self.scaler.transform(aggregated)
            raw_pred = self.model.predict(x)
            prediction = raw_pred[0] if hasattr(raw_pred, "__getitem__") else raw_pred

        normalized = normalize_prediction(prediction)
        if normalized is None:
            raise ValueError(f"Could not normalize prediction: {prediction}")
        return normalized

    def status_text(self) -> str:
        return (
            f"model loaded: {self.artifact_path.name}; "
            f"2.0s windows / 0.5s overlap + eegproc entropy + asymmetry"
        )


def normalize_prediction(prediction: Any) -> str | None:
    if hasattr(prediction, "item"):
        prediction = prediction.item()
    key = str(prediction).strip().upper()
    return PREDICTION_MAP.get(key)