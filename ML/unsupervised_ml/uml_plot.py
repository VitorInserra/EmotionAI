from ML.data_proc import relevant_sensors, relevant_bands, df_relevant

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

TARGET_COL = "performance_metric"  # adjust if needed


# ---------------------------
# Column utilities (use precomputed *_mean / *_std)
# ---------------------------
def sensor_band_mean_cols(sensor: str, bands: list[str]) -> list[str]:
    s = sensor.lower()
    return [f"{s}_{b}_mean" for b in bands if f"{s}_{b}_mean" in df_relevant.columns]

def sensor_band_std_cols(sensor: str, bands: list[str]) -> list[str]:
    s = sensor.lower()
    return [f"{s}_{b}_std" for b in bands if f"{s}_{b}_std" in df_relevant.columns]


# ---------------------------
# Label utilities
# ---------------------------
def normalize_test_version(series: pd.Series) -> pd.Series:
    def to_ab(v):
        if pd.isna(v): return np.nan
        s = str(v).strip().lower()
        if s in {"1", "a", "false"}: return "A"
        if s in {"2", "b", "true"}:  return "B"
        try:
            n = float(s)
            if n == 1: return "A"
            if n == 2: return "B"
        except Exception:
            pass
        return np.nan
    out = series.map(to_ab)
    return pd.Categorical(out, categories=["A", "B"], ordered=True)


# ---------------------------
# Filtering utilities
# ---------------------------
def _apply_sigma_filter_on_std(df: pd.DataFrame, sensor: str, *, sigma_k: float = 1.5,
                               robust: bool = False, verbose: bool = True) -> pd.DataFrame:
    """
    Tight outlier filter on the 'std' column per sensor.
    robust=False -> keep within mean(std) ± k*std(std)
    robust=True  -> keep within median(std) ± k*(1.4826*MAD(std))
    """
    if df.empty: return df

    s = df["std"].values
    if robust:
        center = np.nanmedian(s)
        mad = np.nanmedian(np.abs(s - center))
        scale = 1.4826 * mad
    else:
        center = np.nanmean(s)
        scale  = np.nanstd(s, ddof=1)

    if not np.isfinite(scale) or scale == 0:
        if verbose:
            print(f"[{sensor}] sigma filter skipped (scale≈0).")
        return df

    lo, hi = center - sigma_k*scale, center + sigma_k*scale
    mask = (df["std"] >= lo) & (df["std"] <= hi)
    removed = int((~mask).sum())
    if verbose and removed > 0:
        mode = "MAD" if robust else "mean/std"
        print(f"[{sensor}] {mode} filter (k={sigma_k}) removed {removed} / {len(df)} rows.")
    return df.loc[mask]


# ---------------------------
# Core aggregation (use existing *_mean and *_std)
# ---------------------------
def _aggregate_sensor_rows(sensor: str) -> pd.DataFrame:
    """
    For a given sensor, build a DataFrame with:
      - avg: mean across that sensor's *_mean columns (selected bands)
      - std: mean across that sensor's *_std columns (selected bands)
      - perf: performance metric
      - test_version (normalized to A/B if present)
    """
    mean_cols = sensor_band_mean_cols(sensor, relevant_bands)
    std_cols  = sensor_band_std_cols(sensor, relevant_bands)

    if not mean_cols and not std_cols:
        return pd.DataFrame()

    # Fill missing groups with NaN-safe behavior
    avg_vals = df_relevant[mean_cols].mean(axis=1, skipna=True) if mean_cols else pd.Series(np.nan, index=df_relevant.index)
    std_vals = df_relevant[std_cols].mean(axis=1, skipna=True) if std_cols  else pd.Series(np.nan, index=df_relevant.index)

    out = pd.DataFrame({
        "sensor": sensor,
        "avg": avg_vals.replace([np.inf, -np.inf], np.nan),
        "std": std_vals.replace([np.inf, -np.inf], np.nan),
        "perf": df_relevant[TARGET_COL].values
    }).dropna(subset=["avg", "std", "perf"])

    if "test_version" in df_relevant.columns:
        out["test_version"] = normalize_test_version(df_relevant["test_version"])
    return out


# ---------------------------
# Plots
# ---------------------------
def plot_avg_v_std(*, sigma_k: float = 1.5, robust: bool = False):
    """
    2D per-sensor plot: x = avg (mean of *_mean across bands), y = std (mean of *_std across bands).
    Applies tight sigma/MAD filter on 'std'.
    """
    out_dir = Path("kmeans_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    any_plotted = False

    for sensor in relevant_sensors:
        tmp = _aggregate_sensor_rows(sensor)
        if tmp.empty:
            print(f"[skip] No columns matched for {sensor}")
            continue

        tmp = _apply_sigma_filter_on_std(tmp, sensor, sigma_k=sigma_k, robust=robust)
        if tmp.empty:
            print(f"[{sensor}] No data left after filtering; skipping.")
            continue

        plt.figure()
        # Color by A/B if available
        if "test_version" in tmp.columns:
            a_pts, b_pts = tmp[tmp["test_version"]=="A"], tmp[tmp["test_version"]=="B"]
            if len(a_pts): plt.scatter(a_pts["avg"], a_pts["std"], label="A", s=18, alpha=0.9)
            if len(b_pts): plt.scatter(b_pts["avg"], b_pts["std"], label="B", s=18, alpha=0.9)
            if len(a_pts) or len(b_pts):
                plt.legend(title="test_version")
            else:
                plt.scatter(tmp["avg"], tmp["std"], s=18, alpha=0.9)
        else:
            plt.scatter(tmp["avg"], tmp["std"], s=18, alpha=0.9)

        plt.xlabel("Sensor avg across bands (from *_mean)")
        plt.ylabel("Sensor std across bands (from *_std)")
        plt.title(f"{sensor}: avg vs std (σ-filter k={sigma_k}, robust={robust})")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.savefig(out_dir / f"{sensor}_avg_vs_std_sig{sigma_k}_{'rob' if robust else 'std'}.png",
                    bbox_inches="tight", dpi=150)
        plt.show()
        any_plotted = True

    if not any_plotted:
        raise RuntimeError("Nothing plotted. Loosen filtering or check column names.")
    print(f"Saved 2D avg-vs-std PNGs to: {out_dir.resolve()}")


def plot_avg_std_perf(*, sigma_k: float = 1.5, robust: bool = False):
    """
    3D per-sensor plot: x = avg (mean of *_mean), y = std (mean of *_std), z = performance_metric.
    Applies tight sigma/MAD filter on 'std'.
    """
    out_dir = Path("kmeans_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    any_plotted = False

    for sensor in relevant_sensors:
        tmp = _aggregate_sensor_rows(sensor)
        if tmp.empty:
            print(f"[skip] No columns matched for {sensor}")
            continue

        tmp = _apply_sigma_filter_on_std(tmp, sensor, sigma_k=sigma_k, robust=robust)
        if tmp.empty:
            print(f"[{sensor}] No data left after filtering; skipping.")
            continue

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        if "test_version" in tmp.columns:
            a_pts, b_pts = tmp[tmp["test_version"]=="A"], tmp[tmp["test_version"]=="B"]
            if len(a_pts): ax.scatter(a_pts["avg"], a_pts["std"], a_pts["perf"], label="A", s=18, alpha=0.9)
            if len(b_pts): ax.scatter(b_pts["avg"], b_pts["std"], b_pts["perf"], label="B", s=18, alpha=0.9)
            if len(a_pts) or len(b_pts):
                ax.legend(title="test_version")
            else:
                ax.scatter(tmp["avg"], tmp["std"], tmp["perf"], s=18, alpha=0.9)
        else:
            ax.scatter(tmp["avg"], tmp["std"], tmp["perf"], s=18, alpha=0.9)

        ax.set_xlabel("Sensor avg across bands (from *_mean)")
        ax.set_ylabel("Sensor std across bands (from *_std)")
        ax.set_zlabel(TARGET_COL)
        ax.set_title(f"{sensor}: avg vs std vs {TARGET_COL} (σ-filter k={sigma_k}, robust={robust})")
        ax.grid(True)
        ax.view_init(elev=18, azim=35)

        plt.savefig(out_dir / f"{sensor}_avg_std_{TARGET_COL}_3d_sig{sigma_k}_{'rob' if robust else 'std'}.png",
                    bbox_inches="tight", dpi=150)
        plt.show()
        any_plotted = True

    if not any_plotted:
        raise RuntimeError("Nothing plotted. Loosen filtering or check columns.")
    print(f"Saved 3D PNGs to: {out_dir.resolve()}")


def plot_std_vs_perf(*, sigma_k: float = 1.5, robust: bool = False):
    """
    2D per-sensor plot: x = std (mean of *_std across bands), y = performance_metric.
    Applies tight sigma/MAD filter on 'std'.
    """
    out_dir = Path("kmeans_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    any_plotted = False

    for sensor in relevant_sensors:
        tmp = _aggregate_sensor_rows(sensor)
        if tmp.empty:
            print(f"[skip] No columns matched for {sensor}")
            continue

        tmp = _apply_sigma_filter_on_std(tmp, sensor, sigma_k=sigma_k, robust=robust)
        if tmp.empty:
            print(f"[{sensor}] No data left after filtering; skipping.")
            continue

        plt.figure()
        if "test_version" in tmp.columns:
            a_pts, b_pts = tmp[tmp["test_version"]=="A"], tmp[tmp["test_version"]=="B"]
            if len(a_pts): plt.scatter(a_pts["std"], a_pts["perf"], label="A", s=18, alpha=0.9)
            if len(b_pts): plt.scatter(b_pts["std"], b_pts["perf"], label="B", s=18, alpha=0.9)
            if len(a_pts) or len(b_pts):
                plt.legend(title="test_version")
            else:
                plt.scatter(tmp["std"], tmp["perf"], s=18, alpha=0.9)
        else:
            plt.scatter(tmp["std"], tmp["perf"], s=18, alpha=0.9)

        plt.xlabel("Sensor std across bands (from *_std)")
        plt.ylabel(TARGET_COL)
        plt.title(f"{sensor}: std vs {TARGET_COL} (σ-filter k={sigma_k}, robust={robust})")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.savefig(out_dir / f"{sensor}_std_vs_{TARGET_COL}_sig{sigma_k}_{'rob' if robust else 'std'}.png",
                    bbox_inches="tight", dpi=150)
        plt.show()
        any_plotted = True

    if not any_plotted:
        raise RuntimeError("Nothing plotted. Loosen filtering or check columns.")
    print(f"Saved 2D std-vs-performance PNGs to: {out_dir.resolve()}")
