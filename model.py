from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

try:
    import joblib  # type: ignore
except Exception:  # noqa: BLE001
    joblib = None


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
    """
    Lightweight wrapper around a pre-trained quadrant classifier.

    Expected artifacts:
      - MODEL_ARTIFACT environment variable, or
      - ./models/quadrant_model.joblib, or
      - ./models/quadrant_model.pkl

    Supported object shapes:
      1) a raw estimator with `.predict`
      2) a dict bundle with keys such as:
         {"model": estimator, "scaler": scaler, "feature_names": [...]}.
    """

    def __init__(self, artifact_path: Optional[str | Path] = None) -> None:
        self.artifact_path = self._resolve_artifact_path(artifact_path)
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.feature_names: list[str] = POW_COLUMNS.copy()
        self.available = False
        self.load_error: Optional[str] = None

        if self.artifact_path is not None:
            self._load_artifact(self.artifact_path)

    def _resolve_artifact_path(self, artifact_path: Optional[str | Path]) -> Optional[Path]:
        if artifact_path is not None:
            path = Path(artifact_path).expanduser().resolve()
            return path if path.exists() else None

        env_path = os.environ.get("MODEL_ARTIFACT")
        if env_path:
            path = Path(env_path).expanduser().resolve()
            if path.exists():
                return path

        base_dir = Path(__file__).resolve().parent
        candidates = [
            base_dir / "models" / "quadrant_model.joblib",
            base_dir / "models" / "quadrant_model.pkl",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_artifact(self, path: Path) -> None:
        try:
            if path.suffix == ".joblib":
                if joblib is None:
                    raise RuntimeError("joblib is not installed.")
                artifact = joblib.load(path)
            else:
                with path.open("rb") as handle:
                    artifact = pickle.load(handle)

            if isinstance(artifact, dict):
                self.model = artifact.get("model") or artifact.get("estimator")
                self.scaler = artifact.get("scaler")
                feature_names = artifact.get("feature_names")
                if feature_names:
                    self.feature_names = list(feature_names)
            else:
                self.model = artifact
                if hasattr(self.model, "feature_names_in_"):
                    self.feature_names = list(self.model.feature_names_in_)

            self.available = self.model is not None and hasattr(self.model, "predict")
            if not self.available:
                raise RuntimeError("Artifact loaded, but no estimator with a predict() method was found.")
        except Exception as exc:  # noqa: BLE001
            self.available = False
            self.load_error = str(exc)

    def predict_code(self, psd_features: Mapping[str, float]) -> Optional[str]:
        if not self.available or self.model is None:
            return None

        row = {column: float(psd_features.get(column, 0.0)) for column in self.feature_names}
        frame = pd.DataFrame([row], columns=self.feature_names).fillna(0.0)

        x = frame
        if self.scaler is not None:
            x = self.scaler.transform(frame)

        raw_pred = self.model.predict(x)
        if isinstance(raw_pred, (list, tuple)):
            pred = raw_pred[0]
        else:
            try:
                pred = raw_pred[0]
            except Exception:
                pred = raw_pred
        return normalize_prediction(pred)

    def status_text(self) -> str:
        if self.available:
            name = self.artifact_path.name if self.artifact_path is not None else "custom model"
            return f"model loaded: {name}"
        if self.load_error:
            return f"model unavailable: {self.load_error}"
        return "model unavailable: no artifact found"


def normalize_prediction(prediction: Any) -> Optional[str]:
    if prediction is None:
        return None

    try:
        if hasattr(prediction, "item"):
            prediction = prediction.item()
    except Exception:
        pass

    key = str(prediction).strip().upper()
    return PREDICTION_MAP.get(key)
