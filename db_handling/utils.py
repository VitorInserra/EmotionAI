import pandas as pd
import numpy as np

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
