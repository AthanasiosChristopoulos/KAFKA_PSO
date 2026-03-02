from pathlib import Path
import os

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv("java/.env")


def compute_pbest_bytes(df: pd.DataFrame) -> pd.Series:
    """
    Estimate pBest bytes when CSV only has TOTAL_BYTES_SENT overall.
    Assumes average bytes/message applies to pBest too.
    """
    total_bytes = df["TOTAL_BYTES_SENT"].astype(float)
    total_msgs = df["TOTAL_MESSAGES_SENT"].astype(float).replace(0, pd.NA)
    pbest_msgs = df["TOTAL_MESSAGES_SENT_PBEST"].astype(float)

    bytes_per_msg = (total_bytes / total_msgs).fillna(0.0)
    pbest_bytes = pbest_msgs * bytes_per_msg
    return pbest_bytes


def main():
    mode = os.getenv("EXPERIMENTATION_MODE", "").strip()
    experimentation_dir = os.getenv("EXPERIMENTATION_DIR", "").strip()

    if not experimentation_dir:
        raise RuntimeError("EXPERIMENTATION_DIR is not set in java/.env")

    # Choose CSV + x-axis + plots by mode
    if mode == "THRESHOLD":
        csv_path = Path(f"java/{experimentation_dir}/results_threshold.csv")
        xcol = "LOSS_THRESHOLD_DIFF"
        xlabel = "LOSS_THRESHOLD_DIFF (T)"
        suffix = "threshold"
        plots = [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs Threshold (T)", "accuracy"),
            ("PBEST_BYTES_EST", "Estimated pBest bytes", "pBest Bytes vs Threshold (T)", "bytes"),
        ]

    elif mode == "FILTER_ENABLED":
        csv_path = Path(f"java/{experimentation_dir}/results_filter_enabled.csv")
        xcol = "FILTER_ENABLED"
        xlabel = "FILTER_ENABLED"
        suffix = "filter_enabled"
        plots = [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs FILTER_ENABLED", "accuracy"),
            ("TOTAL_ELAPSED", "TOTAL_ELAPSED (sec)", "Time vs FILTER_ENABLED", "time"),
            ("TOTAL_BYTES_SENT", "TOTAL_BYTES_SENT", "Bytes vs FILTER_ENABLED", "bytes"),
        ]

    else:
        # default: N_WORKERS experiments
        csv_path = Path(f"java/{experimentation_dir}/results_n_workers.csv")
        xcol = "N_WORKERS"
        xlabel = "N_WORKERS"
        suffix = "workers"
        plots = [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs N_WORKERS", "accuracy"),
            ("TOTAL_ELAPSED", "TOTAL_ELAPSED (sec)", "Time vs N_WORKERS", "time"),
            ("TOTAL_BYTES_SENT", "TOTAL_BYTES_SENT", "Bytes vs N_WORKERS", "bytes"),
        ]

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    outdir = csv_path.parent
    df = pd.read_csv(csv_path)

    if xcol not in df.columns:
        raise KeyError(f"CSV missing x column '{xcol}'. Columns: {list(df.columns)}")

    df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
    df = df.sort_values(xcol)

    if "TOTAL_MESSAGES_SENT_PBEST" in df.columns and "TOTAL_MESSAGES_SENT" in df.columns and "TOTAL_BYTES_SENT" in df.columns:
        df["PBEST_BYTES_EST"] = compute_pbest_bytes(df)

    xs = df[xcol].tolist()

    saved = []
    for ycol, ylabel, title, tag in plots:
        if ycol not in df.columns:
            raise KeyError(f"CSV missing y column '{ycol}' needed for plot '{title}'. Columns: {list(df.columns)}")

        ys = pd.to_numeric(df[ycol], errors="coerce").fillna(0.0).tolist()

        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.ylim(bottom=0)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(xs)  # ok for small number of discrete x values
        plt.grid(True)

        outpath = outdir / f"{csv_path.stem}_{tag}_vs_{suffix}.png"
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close()
        saved.append(outpath)

    print("Mode:", mode or "(default N_WORKERS)")
    print("CSV :", csv_path)
    print("Saved:")
    for p in saved:
        print(" -", p)


if __name__ == "__main__":
    main()