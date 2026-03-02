
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os

from dotenv import load_dotenv
loaded = load_dotenv("java/.env")

# ==========================================================================================================

def main():

    mode = os.getenv("EXPERIMENTATION_MODE")
    experimentation_dir = os.getenv("EXPERIMENTATION_DIR")

    # =================================================================================

    if mode == "THRESHOLD":
        csv_path = Path(f"java/{experimentation_dir}/results_threshold.csv")
        xcol = "LOSS_THRESHOLD_DIFF"
        xlabel = "LOSS_THRESHOLD_DIFF (T)"
        suffix = "threshold"
        plots = [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs Threshold (T)", "accuracy"),
            ("TOTAL_BYTES_SENT", "TOTAL_BYTES_SENT", "Bytes vs Threshold (T)", "bytes"),
        ]

    # =================================================================================

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

    # =================================================================================

    else:

        csv_path = Path(f"java/{experimentation_dir}/results_n_workers.csv")
        xcol = "N_WORKERS"  # use for x in the plot the field of the .csv N_WORKERS
        xlabel = "N_WORKERS"
        suffix = "workers"
        plots = [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs N_WORKERS", "accuracy"),
            ("TOTAL_ELAPSED", "TOTAL_ELAPSED (sec)", "Time vs N_WORKERS", "time"),
            ("TOTAL_BYTES_SENT", "TOTAL_BYTES_SENT", "Bytes vs N_WORKERS", "bytes"),
        ]

    # =================================================================================

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    outdir = csv_path.parent
    df = pd.read_csv(csv_path).sort_values(xcol)

    xs = df[xcol].tolist()

    saved = []
    for ycol, ylabel, title, tag in plots:
        plt.figure()
        plt.plot(xs, df[ycol].tolist(), marker="o")
        plt.ylim(bottom=0)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(xs)
        plt.grid(True)
        outpath = outdir / f"{csv_path.stem}_{tag}_vs_{suffix}.png"
        plt.savefig(outpath, dpi=200)
        plt.close()
        saved.append(outpath)

    print("Mode:", mode)
    print("CSV :", csv_path)
    print("Saved:")

    for p in saved:
        print(" -", p)

# ==========================================================================================================

if __name__ == "__main__":
    main()