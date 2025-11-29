from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple, Union, Set
from typing import Iterable, List, Dict
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import chain, combinations, product
import re


_PSD_RE = re.compile(
    r"^(?P<ch>[A-Z0-9]+)_(?P<band>delta|theta|alpha|betaL|betaH|gamma)$"
)
_ENT_RE = re.compile(r"^(?P<ch>[A-Z0-9]+)_entropy$")
_PAIR_RE = re.compile(
    r"^(?P<ch1>[A-Z0-9]+)_(?P<ch2>[A-Z0-9]+)_(?P<band>delta|theta|alpha|betaL|betaH|gamma)_(?P<kind>da|ra)$"
)


def read_table(filename: str = "datasets/features_table.csv") -> pd.DataFrame:
    script_dir = Path(__file__).resolve().parent
    file_path = script_dir.parent / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find file: {file_path}")

    df = pd.read_csv(file_path)
    return df

def important_features_list(
    filename: str = "datasets/USE_feature_importance/dreamer_feature_importance_arousal.csv",
):
    df = pd.read_csv(filename)

    return df["feature"].to_list()


def filter_features(
    features: Iterable[str],
    *,
    remove_channels: Optional[Iterable[str]] = None,
    remove_bands: Optional[Iterable[str]] = None,
    remove_types: Optional[
        Iterable[str]
    ] = None,  # any of {"psd","entropy","da","ra","pair","pair_da","pair_ra"}
    remove_pairs_with_channels: Optional[
        Iterable[str]
    ] = None,
) -> List[str]:
    chans: Set[str] = set(map(str.upper, remove_channels or []))
    pbands: Set[str] = set(remove_bands or [])
    ptypes: Set[str] = set((remove_types or []))
    pair_chans: Set[str] = set(map(str.upper, remove_pairs_with_channels or []))

    def should_drop(feat: str) -> bool:
        f = feat.strip()

        m = _PAIR_RE.match(f)
        if m:
            ch1, ch2 = m["ch1"].upper(), m["ch2"].upper()
            band, kind = m["band"], m["kind"]  # kind in {"da","ra"}

            if ch1 in chans or ch2 in chans:
                return True
            if pair_chans and (ch1 in pair_chans or ch2 in pair_chans):
                return True
            if band in pbands:
                return True
            if "pair" in ptypes:
                return True
            if kind in ptypes or f"pair_{kind}" in ptypes:
                return True
            return False

        m = _ENT_RE.match(f)
        if m:
            ch = m["ch"].upper()
            if ch in chans:
                return True
            if "entropy" in ptypes:
                return True
            return False

        m = _PSD_RE.match(f)
        if m:
            ch, band = m["ch"].upper(), m["band"]
            if ch in chans:
                return True
            if band in pbands:
                return True
            if "psd" in ptypes:
                return True
            return False

        return False

    return [f for f in features if not should_drop(f)]


# Homologous left–right pairs for Emotiv/DREAMER (14ch)
HOMOLOGOUS_PAIRS = [
    ("AF3", "AF4"),
    ("F3", "F4"),
    ("F7", "F8"),
    ("FC5", "FC6"),
    ("T7", "T8"),
    ("P7", "P8"),
    ("O1", "O2"),
]


AS_BANDS = {"delta", "theta", "alpha", "beta", "gamma"}


def compute_asymmetry_from_psd(
    psd: pd.DataFrame,
    pairs: list[tuple[str, str]] = HOMOLOGOUS_PAIRS,
    eps: float = 1e-12,
    add_log: bool = True,
    prefix_da: str = "da",
    prefix_ra: str = "ra",
) -> pd.DataFrame:
    bands_present = set()
    for col in psd.columns:
        if "_" in col:
            ch, band = col.rsplit("_", 1)
            if band in AS_BANDS:
                bands_present.add(band)

    out_cols = {}

    for L, R in pairs:
        for band in bands_present:
            cL = f"{L}_{band}"
            cR = f"{R}_{band}"
            if cL not in psd.columns or cR not in psd.columns:
                continue

            PL = psd[cL].astype(float)
            PR = psd[cR].astype(float)

            if add_log:
                da = np.log(PR + eps) - np.log(PL + eps)
            else:
                da = (PR + eps) - (PL + eps)

            ra = (PR - PL) / (PR + PL + eps)

            out_cols[f"{R}_{L}_{band}_{prefix_da}"] = da
            out_cols[f"{R}_{L}_{band}_{prefix_ra}"] = ra

    return pd.DataFrame(out_cols, index=psd.index)


def plot_regressor_accuracy(y_true, y_pred, size_increment=5, title=None):

    fig, ax = plt.subplots(figsize=(5, 5))

    size: dict[tuple, int] = {}
    for i in range(len(y_true)):
        if (y_pred[i], y_true[i]) in size:
            size[(y_pred[i], y_true[i])] += size_increment
        else:
            size[(y_pred[i], y_true[i])] = size_increment

    s = []
    for i in range(len(y_true)):
        s.append(size[(y_pred[i], y_true[i])])

    ax.scatter(y_true, y_pred, s=s, alpha=0.7, edgecolors="none")
    ax.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], c="blue", linestyle="--", linewidth=2)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    if title:
        ax.set_title(title)

    return ax
