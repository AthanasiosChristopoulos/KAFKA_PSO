import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

## ======================================================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to results_*.csv")
    ap.add_argument("--outdir", default=None, help="Output directory for images (default: same as CSV)")

    csv_path = Path("java/experimental_results_v1/results_n_workers.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    outdir = Path("java/experimental_results_v1")
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Ensure sorted by N_WORKERS for nice lines
    df = df.sort_values("N_WORKERS")

    xs = df["N_WORKERS"].tolist()

    # 1) Accuracy vs N_WORKERS  (GBEST_ACC)
    plt.figure()
    plt.plot(xs, df["GBEST_ACC"].tolist(), marker="o")
    plt.ylim(bottom=0)
    plt.xlabel("N_WORKERS")
    plt.ylabel("GBEST_ACC")
    plt.title("Accuracy vs N_WORKERS")
    plt.xticks(xs)
    plt.grid(True)
    p1 = outdir / f"{csv_path.stem}_accuracy_vs_workers.png"
    plt.savefig(p1, dpi=200)
    plt.close()

    # 2) Time vs N_WORKERS  (TOTAL_ELAPSED)
    plt.figure()
    plt.plot(xs, df["TOTAL_ELAPSED"].tolist(), marker="o")
    plt.ylim(bottom=0)
    plt.xlabel("N_WORKERS")
    plt.ylabel("TOTAL_ELAPSED (sec)")
    plt.title("Training Time vs N_WORKERS")
    plt.xticks(xs)
    plt.grid(True)
    p2 = outdir / f"{csv_path.stem}_time_vs_workers.png"
    plt.savefig(p2, dpi=200)
    plt.close()

    # 3) Bytes vs N_WORKERS  (ignore messages as you asked)
    plt.figure()
    plt.plot(xs, df["TOTAL_BYTES_SENT"].tolist(), marker="o")
    plt.ylim(bottom=0)
    plt.xlabel("N_WORKERS")
    plt.ylabel("TOTAL_BYTES_SENT")
    plt.title("Total Bytes Sent vs N_WORKERS")
    plt.xticks(xs)
    plt.grid(True)
    p3 = outdir / f"{csv_path.stem}_bytes_vs_workers.png"
    plt.savefig(p3, dpi=200)
    plt.close()

    print("Saved:")
    print(" -", p1)
    print(" -", p2)
    print(" -", p3)

## ======================================================================================================

if __name__ == "__main__":
    main()