from pathlib import Path
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import numpy as np

load_dotenv("../java/.env")

# ============================================================================================
# Filename -> plotting config

CSV_CONFIGS = {
    "results_threshold.csv": {
        "mode": "THRESHOLD",
        "xcol": "LOSS_THRESHOLD_DIFF",
        "xlabel": "LOSS_THRESHOLD_DIFF",
        "suffix": "threshold",
        "plots": [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs Threshold", "accuracy"),
            ("PBEST_BYTES_EST", "Estimated pBest bytes", "pBest Bytes vs Threshold", "bytes"),
            ("TOTAL_BYTES_SENT", "TOTAL_BYTES_SENT", "Bytes vs Threshold", "bytes"),
            ("TOTAL_MESSAGES_SENT", "TOTAL_MESSAGES_SENT", "Messages vs Threshold", "messages"),
        ],
    },
    "results_filter_enabled.csv": {
        "mode": "FILTER_ENABLED",
        "xcol": "FILTER_ENABLED",
        "xlabel": "FILTER_ENABLED",
        "suffix": "filter_enabled",
        "plots": [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs FILTER_ENABLED", "accuracy"),
            ("TOTAL_ELAPSED", "TOTAL_ELAPSED (sec)", "Time vs FILTER_ENABLED", "time"),
            ("TOTAL_BYTES_SENT", "TOTAL_BYTES_SENT", "Bytes vs FILTER_ENABLED", "bytes"),
            ("TOTAL_MESSAGES_SENT", "TOTAL_MESSAGES_SENT", "Messages vs FILTER_ENABLED", "messages"),
        ],
    },
    "results_strength.csv": {
        "mode": "FILTER_STRENGTH",
        "xcol": "STRENGTH_CODE",
        "xlabel": "FILTER_STRENGTH_INDEX",
        "suffix": "strength",
        "plots": [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs FILTER_STRENGTH", "accuracy"),
            ("TOTAL_ELAPSED", "TOTAL_ELAPSED (sec)", "Time vs FILTER_STRENGTH", "time"),
            ("TOTAL_BYTES_SENT", "TOTAL_BYTES_SENT", "Bytes vs FILTER_STRENGTH", "bytes"),
            ("TOTAL_MESSAGES_SENT", "TOTAL_MESSAGES_SENT", "Messages vs FILTER_STRENGTH", "messages"),
        ],
    },
    "results_fully_informed_vs_classical.csv": {
        "mode": "FULLY_INFORMED_VS_CLASSICAL",
        "xcol": "FULLY_INFORMED",
        "xlabel": "FULLY_INFORMED",
        "suffix": "fully_informed_vs_classical",
        "plots": [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs FULLY_INFORMED_VS_CLASSICAL", "accuracy"),
            ("TOTAL_ELAPSED", "TOTAL_ELAPSED (sec)", "Time vs FULLY_INFORMED_VS_CLASSICAL", "time"),
            ("TOTAL_BYTES_SENT", "TOTAL_BYTES_SENT", "Bytes vs FULLY_INFORMED_VS_CLASSICAL", "bytes"),
            ("TOTAL_MESSAGES_SENT", "TOTAL_MESSAGES_SENT", "Messages vs FULLY_INFORMED_VS_CLASSICAL", "messages"),
        ],
    },
    # "results_monitoring_iterations.csv": {
    #     "mode": "MONITORING_ITERATIONS",
    #     "xcol": "MONITORING_ITER",
    #     "xlabel": "MONITORING_ITER",
    #     "suffix": "monitoring",
    #     "plots": [
    #         ("ACCURACY", "ACCURACY", "Accuracy vs Monitoring Iteration", "accuracy"),
    #     ],
    # },
    
    "results_monitoring_iterations.csv": {
        "mode": "MONITORING_ITERATIONS",
        "xcol": "MONITORING_ITER",
        "xlabel": "MONITORING_ITER",
        "suffix": "monitoring",
        "plots": [
            ("ACCURACY", "ACCURACY", "Accuracy vs Monitoring Iteration", "accuracy", "MONITORING_ITER", "MONITORING_ITER"),
            ("ACCURACY", "ACCURACY", "Accuracy vs Time", "accuracy_vs_time", "TIME_SEC", "TIME_SEC"),
        ],
    },
    "results_n_workers.csv": {
        "mode": "N_WORKERS",
        "xcol": "N_WORKERS",
        "xlabel": "N_WORKERS",
        "suffix": "workers",
        "plots": [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs N_WORKERS", "accuracy"),
            ("TOTAL_ELAPSED", "TOTAL_ELAPSED (sec)", "Time vs N_WORKERS", "time"),
            ("TOTAL_BYTES_SENT", "TOTAL_BYTES_SENT", "Bytes vs N_WORKERS", "bytes"),
            ("TOTAL_MESSAGES_SENT", "TOTAL_MESSAGES_SENT", "Messages vs N_WORKERS", "messages"),
        ],
    },
    "results_dimensionality.csv": {
        "mode": "DIMENSIONALITY",
        "xcol": "DIMENSIONALITY",
        "xlabel": "DIMENSIONALITY",
        "suffix": "dimensionality",
        "plots": [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs DIMENSIONALITY", "accuracy"),
            ("TOTAL_ELAPSED", "TOTAL_ELAPSED (sec)", "Time vs DIMENSIONALITY", "time"),
            ("TOTAL_BYTES_SENT", "TOTAL_BYTES_SENT", "Bytes vs DIMENSIONALITY", "bytes"),
            ("TOTAL_MESSAGES_SENT", "TOTAL_MESSAGES_SENT", "Messages vs DIMENSIONALITY", "messages"),
        ],
    },
    "results_topology.csv": {
        "mode": "TOPOLOGY",
        "xcol": "NEIGHBORHOOD_TOPOLOGY",
        "xlabel": "TOPOLOGY",
        "suffix": "topology",
        "plots": [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs TOPOLOGY", "accuracy"),
            ("TOTAL_ELAPSED", "TOTAL_ELAPSED (sec)", "Time vs TOPOLOGY", "time"),
            ("TOTAL_BYTES_SENT", "TOTAL_BYTES_SENT", "Bytes vs TOPOLOGY", "bytes"),
            ("TOTAL_MESSAGES_SENT", "TOTAL_MESSAGES_SENT", "Messages vs TOPOLOGY", "messages"),
        ],
    },
}

MODE_TO_FILENAME = {
    cfg["mode"]: filename
    for filename, cfg in CSV_CONFIGS.items()
}

# ============================================================================================

def compute_pbest_bytes(df: pd.DataFrame) -> pd.Series:
    total_bytes = df["TOTAL_BYTES_SENT"].astype(float)
    total_msgs = df["TOTAL_MESSAGES_SENT"].astype(float).replace(0, pd.NA)
    pbest_msgs = df["TOTAL_MESSAGES_SENT_PBEST"].astype(float)

    bytes_per_msg = (total_bytes / total_msgs).fillna(0.0)
    pbest_bytes = pbest_msgs * bytes_per_msg
    return pbest_bytes

# ============================================================================================

def downsample_df(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    n = len(df)
    if n <= max_rows:
        return df
    idx = np.linspace(0, n - 1, num=max_rows, dtype=int)
    return df.iloc[idx].copy()

# ============================================================================================

def get_csv_config_from_mode(mode: str) -> dict:
    mode = (mode or "").strip()
    if mode not in MODE_TO_FILENAME:
        raise ValueError(f"Unsupported EXPERIMENTATION_MODE: {mode}")

    filename = MODE_TO_FILENAME[mode]
    return CSV_CONFIGS[filename]

# ============================================================================================

def get_csv_config_from_filename(csv_path: Path) -> dict:
    name = csv_path.name
    if name not in CSV_CONFIGS:
        raise ValueError(f"Unsupported CSV filename: {name}")
    return CSV_CONFIGS[name]

# ============================================================================================

def process_one_csv(csv_path: Path):
    config = get_csv_config_from_filename(csv_path)

    mode = config["mode"]
    xcol = config["xcol"]
    xlabel = config["xlabel"]
    suffix = config["suffix"]
    plots = config["plots"]

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    outdir = csv_path.parent

    if(mode != "MONITORING_ITERATIONS"):
        df = pd.read_csv(csv_path)

    else:
        lines = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("FILTER_ENABLED,"):
                    break
                lines.append(line)

        from io import StringIO
        df = pd.read_csv(StringIO("".join(lines)))
        

    if xcol not in df.columns:
        raise KeyError(f"CSV missing x column '{xcol}'. Columns: {list(df.columns)}")

    if mode in {"N_WORKERS", "MODEL_VERSION", "RING_RADIUS", "DIMENSIONALITY", "MONITORING_ITERATIONS"}:
        df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
        df = df.dropna(subset=[xcol]).sort_values(xcol)
    elif mode == "TOPOLOGY":
        df[xcol] = df[xcol].astype(str)
    else:
        return
    
    if (
        "TOTAL_MESSAGES_SENT_PBEST" in df.columns
        and "TOTAL_MESSAGES_SENT" in df.columns
        and "TOTAL_BYTES_SENT" in df.columns
    ):
        df["PBEST_BYTES_EST"] = compute_pbest_bytes(df)

    saved = []

    for plot in plots:
        if len(plot) == 4:
            ycol, ylabel, title, tag = plot
            plot_xcol = xcol
            plot_xlabel = xlabel
        elif len(plot) == 6:
            ycol, ylabel, title, tag, plot_xcol, plot_xlabel = plot
        else:
            raise ValueError(f"Invalid plot config: {plot}")        
            
        if ycol not in df.columns:
            raise KeyError(
                f"CSV missing y column '{ycol}' needed for plot '{title}'. "
                f"Columns: {list(df.columns)}"
            )

        if plot_xcol not in df.columns:
            raise KeyError(
                f"CSV missing x column '{plot_xcol}' needed for plot '{title}'. "
                f"Columns: {list(df.columns)}"
            )

        ys = pd.to_numeric(df[ycol], errors="coerce")
        plot_xs = pd.to_numeric(df[plot_xcol], errors="coerce") if mode == "MONITORING_ITERATIONS" else df[plot_xcol]
        mask = ys.notna() & plot_xs.notna()
        xs_plot = plot_xs.loc[mask].tolist()
        ys_plot = ys.loc[mask].tolist()

        plot_xs = pd.to_numeric(df[plot_xcol], errors="coerce") if mode == "MONITORING_ITERATIONS" else df[plot_xcol]
        mask = ys.notna() & plot_xs.notna()
        xs_plot = plot_xs.loc[mask].tolist()
        ys_plot = ys.loc[mask].tolist()

        if mode == "MONITORING_ITERATIONS":
            max_points = int(os.getenv("MAX_PLOT_POINTS", "1000"))
            plot_df = df.loc[mask, [plot_xcol, ycol]].copy()
            plot_df[plot_xcol] = pd.to_numeric(plot_df[plot_xcol], errors="coerce")
            plot_df[ycol] = pd.to_numeric(plot_df[ycol], errors="coerce")
            plot_df = plot_df.sort_values(plot_xcol)
            plot_df = downsample_df(plot_df, max_points)

            xs_plot = plot_df[plot_xcol].tolist()
            ys_plot = plot_df[ycol].tolist()

        plt.figure()

        if mode == "MONITORING_ITERATIONS":
            plt.plot(xs_plot, ys_plot, marker="o", markersize=3, markeredgewidth=0.3)

        elif mode in {"DIMENSIONALITY", "TOPOLOGY"}:
            positions = np.arange(len(xs_plot))
            plt.plot(positions, ys_plot, marker="o")

        else:
            plt.plot(xs_plot, ys_plot, marker="o")
        
        # plt.margins(y=0.25)
       
        if ycol in {"GBEST_ACC", "ACCURACY"}:
            plt.ylim(0, 1)
            plt.yticks(np.linspace(0, 1, 11))
            
        else:
            ymin, ymax = 0, max(ys_plot)
            plt.ylim(ymin, ymax * 1.15)

        plt.xlabel(plot_xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)

        if mode == "MONITORING_ITERATIONS":
            tick_count = 10

            if len(xs_plot) > tick_count:
                tick_idx = np.linspace(0, len(xs_plot) - 1, tick_count, dtype=int)
            else:
                tick_idx = range(len(xs_plot))

            tick_positions = [xs_plot[i] for i in tick_idx]
            tick_labels = [f"{xs_plot[i]:.1f}" for i in tick_idx]  # <-- rounding

            plt.xticks(tick_positions, tick_labels, ha="right")

        elif mode in {"DIMENSIONALITY", "TOPOLOGY"}:
            plt.xticks(positions, xs_plot)

        else:
            plt.xticks(xs_plot)

        plt.tight_layout()

        outpath = outdir / f"{csv_path.stem}_{tag}.png"
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close()
        saved.append(outpath)

    print(f"[OK] Mode: {mode}")
    print(f"     CSV : {csv_path}")
    print("     Saved:")
    for p in saved:
        print(f"      - {p}")

# ============================================================================================

def process_multi(csv_dir: Path):
    if not csv_dir.exists():
        raise FileNotFoundError(f"Directory not found: {csv_dir}")

    csv_files = sorted(csv_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {csv_dir}")

    supported = 0
    skipped = []

    for csv_path in csv_files:
        try:
            process_one_csv(csv_path)
            supported += 1
        except ValueError:
            skipped.append(csv_path.name)

    print()
    print(f"Processed {supported} supported CSV file(s) in {csv_dir}")
    if skipped:
        print("Skipped unsupported CSV file(s):")
        for name in skipped:
            print(f" - {name}")

# ============================================================================================

def delete_pngs(root_dir: Path):

    png_files = list(root_dir.rglob("*.png"))

    if not png_files:
        print("No PNG files found.")
        return

    for p in png_files:
        try:
            p.unlink()
            print("Deleted:", p)
        except Exception as e:
            print("Failed:", p, e)

    print(f"\nDeleted {len(png_files)} PNG files.")
    
# ============================================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete all PNG plots recursively and exit"
    )

    return parser.parse_args()

# ============================================================================================

def main():
    args = parse_args()

    experimentation_dir = os.getenv("EXPERIMENTATION_DIR", "").strip()
    mode = os.getenv("EXPERIMENTATION_MODE", "").strip()

    default_csv_dir = Path(f"../java/{experimentation_dir}") if experimentation_dir else None
    # csv_dir = Path(f"../java/exp_dataset/fully_informed_vs_classical/mnist/help")
    # csv_dir = Path(f"../java/exp_dataset/dimensionality/pendigits/help")
    csv_dir = Path(f"../java/exp_dataset")
    
    if args.delete:
        delete_pngs(csv_dir)
        return
    
    process_multi(csv_dir)

    if not mode:
        raise RuntimeError("EXPERIMENTATION_MODE is not set in ../java/.env")

    # config = get_csv_config_from_mode(mode)
    # csv_path = csv_dir / MODE_TO_FILENAME[config["mode"]]
    # process_one_csv(csv_path)

# ============================================================================================

if __name__ == "__main__":
    main()