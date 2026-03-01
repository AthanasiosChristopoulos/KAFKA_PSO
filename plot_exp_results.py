# import argparse
# from pathlib import Path

# import pandas as pd
# import matplotlib.pyplot as plt

# ## ======================================================================================================

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--csv", required=True, help="Path to results_*.csv")
#     ap.add_argument("--outdir", default=None, help="Output directory for images (default: same as CSV)")

#     csv_path = Path("java/experimental_results_v1/results_n_workers.csv")
#     if not csv_path.exists():
#         raise FileNotFoundError(f"CSV not found: {csv_path}")

#     outdir = Path("java/experimental_results_v1")
#     outdir.mkdir(parents=True, exist_ok=True)

#     df = pd.read_csv(csv_path)

#     # Ensure sorted by N_WORKERS for nice lines
#     df = df.sort_values("N_WORKERS")

#     xs = df["N_WORKERS"].tolist()

#     # 1) Accuracy vs N_WORKERS  (GBEST_ACC)
#     plt.figure()
#     plt.plot(xs, df["GBEST_ACC"].tolist(), marker="o")
#     plt.ylim(bottom=0)
#     plt.xlabel("N_WORKERS")
#     plt.ylabel("GBEST_ACC")
#     plt.title("Accuracy vs N_WORKERS")
#     plt.xticks(xs)
#     plt.grid(True)
#     p1 = outdir / f"{csv_path.stem}_accuracy_vs_workers.png"
#     plt.savefig(p1, dpi=200)
#     plt.close()

#     # 2) Time vs N_WORKERS  (TOTAL_ELAPSED)
#     plt.figure()
#     plt.plot(xs, df["TOTAL_ELAPSED"].tolist(), marker="o")
#     plt.ylim(bottom=0)
#     plt.xlabel("N_WORKERS")
#     plt.ylabel("TOTAL_ELAPSED (sec)")
#     plt.title("Training Time vs N_WORKERS")
#     plt.xticks(xs)
#     plt.grid(True)
#     p2 = outdir / f"{csv_path.stem}_time_vs_workers.png"
#     plt.savefig(p2, dpi=200)
#     plt.close()

#     # 3) Bytes vs N_WORKERS  (ignore messages as you asked)
#     plt.figure()
#     plt.plot(xs, df["TOTAL_BYTES_SENT"].tolist(), marker="o")
#     plt.ylim(bottom=0)
#     plt.xlabel("N_WORKERS")
#     plt.ylabel("TOTAL_BYTES_SENT")
#     plt.title("Total Bytes Sent vs N_WORKERS")
#     plt.xticks(xs)
#     plt.grid(True)
#     p3 = outdir / f"{csv_path.stem}_bytes_vs_workers.png"
#     plt.savefig(p3, dpi=200)
#     plt.close()

#     print("Saved:")
#     print(" -", p1)
#     print(" -", p2)
#     print(" -", p3)

# ## ======================================================================================================

# if __name__ == "__main__":
#     main()


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os

from dotenv import load_dotenv
loaded = load_dotenv("java/.env")

# ==========================================================================================================

def main():
    mode = os.getenv("EXPERIMENTATION_MODE")

    if mode == "THRESHOLD":
        csv_path = Path("java/experimental_results_server/results_threshold.csv")
        xcol = "LOSS_THRESHOLD_DIFF"
        xlabel = "LOSS_THRESHOLD_DIFF (T)"
        suffix = "threshold"
        plots = [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs Threshold (T)", "accuracy"),
            ("TOTAL_BYTES_SENT", "TOTAL_BYTES_SENT", "Bytes vs Threshold (T)", "bytes"),
        ]
    else:
        csv_path = Path("java/experimental_results_server/results_n_workers.csv")
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