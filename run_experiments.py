import re
import time
import csv
import signal
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

WORKERS_LIST = [5, 10, 15, 20]
# BASE_DIR = Path(__file__).resolve().parent
# RUN_SCRIPT = BASE_DIR.parent / "run_streams.sh"
RUN_SCRIPT = "./run_streams.sh"

ACC_RE = re.compile(r"Global bestAccuracy:\s*([0-9]*\.?[0-9]+)")

END_PATTERNS = [
    re.compile(r"Training is over\.", re.IGNORECASE),
    re.compile(r"Test Records run out\.", re.IGNORECASE),
]

# ========================================================================================

def run_once(n_workers: int, logs_dir: Path):
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"run_{n_workers}.log"

    cmd = [RUN_SCRIPT, str(n_workers)]

    # Start process
    start = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True,
    )

    best_acc = None
    finished = False

    with log_path.open("w", encoding="utf-8") as f:
        for line in p.stdout:
            print(line, end="")   # keep live output
            f.write(line)

            m = ACC_RE.search(line)
            if m:
                best_acc = float(m.group(1))

            if any(r.search(line) for r in END_PATTERNS):
                finished = True
                break

    if finished:    # If we detected end-of-training
        try:
            p.send_signal(signal.SIGINT)
        except Exception:
            pass

    # Wait for a clean shutdown
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()

    end = time.time()
    elapsed = end - start

    return best_acc, elapsed, str(log_path)

# ========================================================================================

def main():
    # logs_dir = BASE_DIR / "experiment_logs"
    logs_dir = Path("./experiment_logs")

    results = []

    for n in WORKERS_LIST:
        print("\n" + "=" * 80)
        print(f"RUNNING EXPERIMENT: N_WORKERS={n}")
        print("=" * 80)
        acc, secs, log_path = run_once(n, logs_dir)

        if acc is None:
            print(f"[WARN] No accuracy parsed for N_WORKERS={n}. Check log: {log_path}")

        results.append({"N_WORKERS": n, "ACCURACY": acc, "TRAIN_TIME_SEC": secs, "LOG": log_path})

    # Save CSV
    csv_path = Path("results.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["N_WORKERS", "ACCURACY", "TRAIN_TIME_SEC", "LOG"])
        w.writeheader()
        w.writerows(results)


    xs = [r["N_WORKERS"] for r in results]

    # Plot 1: Accuracy vs N_WORKERS ==================================================================

    accs = [r["ACCURACY"] for r in results]
    plt.figure()
    plt.plot(xs, accs, marker="o")
    plt.xlabel("N_WORKERS")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs N_WORKERS")
    plt.xticks(xs)
    plt.grid(True)
    plt.savefig("accuracy_vs_workers.png", dpi=200)
    plt.close()

    # Plot 2: Training time vs N_WORKERS ===========================================================

    times = [r["TRAIN_TIME_SEC"] for r in results]
    plt.figure()
    plt.plot(xs, times, marker="o")
    plt.xlabel("N_WORKERS")
    plt.ylabel("Training Time (seconds)")
    plt.title("Training Time vs N_WORKERS")
    plt.xticks(xs)
    plt.grid(True)
    plt.savefig("time_vs_workers.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()