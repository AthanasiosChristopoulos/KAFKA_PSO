import re
import time
import csv
import signal
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

WORKERS_LIST = [2, 4, 6, 8]
RUN_SCRIPT = "./run_streams.sh"

ACC_RE = re.compile(r"Global bestAccuracy:\s*([0-9]*\.?[0-9]+)")

END_PATTERNS = [
    re.compile(r"Training is over\.", re.IGNORECASE),
    re.compile(r"Test Records run out\.", re.IGNORECASE),
]

# NEW: parse times printed by Java
WORKER_TIME_RE = re.compile(r"\[Worker\s+(\d+)\]\s+Wall time:\s*([0-9]*\.?[0-9]+)\s*seconds", re.IGNORECASE)
COORD_TIME_RE  = re.compile(r"\[Coordinator\]\s+Wall time:\s*([0-9]*\.?[0-9]+)\s*seconds", re.IGNORECASE)

# ========================================================================================

def run_once(n_workers, logs_dir):
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"run_{n_workers}.log"

    cmd = [RUN_SCRIPT, str(n_workers)]

    start = time.time()
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    best_acc = None
    finished = False

    # NEW: collect wall times
    coordinator_time_sec = None
    worker_times = {}  # workerId -> seconds

    with log_path.open("w", encoding="utf-8") as f:
        for line in p.stdout:
            print(line, end="")
            f.write(line)

            m = ACC_RE.search(line)
            if m:
                best_acc = float(m.group(1))

            # NEW: parse worker wall time
            wm = WORKER_TIME_RE.search(line)
            if wm:
                wid = int(wm.group(1))
                wsec = float(wm.group(2))
                worker_times[wid] = wsec

            # NEW: parse coordinator wall time
            cm = COORD_TIME_RE.search(line)
            if cm:
                coordinator_time_sec = float(cm.group(1))

            if any(r.search(line) for r in END_PATTERNS):
                finished = True
                break

    if finished:
        try:
            p.send_signal(signal.SIGINT)
        except Exception:
            pass

    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()

    elapsed = time.time() - start

    # NEW: last worker elapsed time = max(worker times) if any
    last_worker_time_sec = max(worker_times.values()) if worker_times else None

    print(
        f"Results for N_WORKERS = {n_workers} : "
        f"best accuracy: {best_acc} - "
        f"training time: {elapsed:.3f} - "
        f"coordinator wall time: {coordinator_time_sec} - "
        f"last worker wall time: {last_worker_time_sec}"
    )

    return best_acc, elapsed, str(log_path), coordinator_time_sec, last_worker_time_sec

# ========================================================================================

def main():
    logs_dir = Path("./experimental_results/experiment_logs")
    Path("experimental_results").mkdir(parents=True, exist_ok=True)

    results = []

    for n in WORKERS_LIST:
        print("\n" + "=" * 80)
        print(f"RUNNING EXPERIMENT: N_WORKERS={n}")
        print("=" * 80)

        acc, secs, log_path, coord_secs, last_worker_secs = run_once(n, logs_dir)

        if acc is None:
            print(f"[WARN] No accuracy parsed for N_WORKERS={n}. Check log: {log_path}")

        results.append({
            "N_WORKERS": n,
            "ACCURACY": acc,
            "TRAIN_TIME_SEC": secs,
            "COORD_WALL_TIME_SEC": coord_secs,
            "LAST_WORKER_WALL_TIME_SEC": last_worker_secs,
            "LOG": log_path
        })

    csv_path = Path("experimental_results/results.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "N_WORKERS",
                "ACCURACY",
                "TRAIN_TIME_SEC",
                "COORD_WALL_TIME_SEC",
                "LAST_WORKER_WALL_TIME_SEC",
                "LOG",
            ],
        )
        w.writeheader()
        w.writerows(results)

    xs = [r["N_WORKERS"] for r in results]

    # Plot 1: Accuracy vs N_WORKERS
    accs = [r["ACCURACY"] for r in results]
    plt.figure()
    plt.plot(xs, accs, marker="o")
    plt.xlabel("N_WORKERS")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs N_WORKERS")
    plt.xticks(xs)
    plt.grid(True)
    plt.savefig("experimental_results/accuracy_vs_workers.png", dpi=200)
    plt.close()

    # Plot 2: Training time vs N_WORKERS
    times = [r["TRAIN_TIME_SEC"] for r in results]
    plt.figure()
    plt.plot(xs, times, marker="o")
    plt.xlabel("N_WORKERS")
    plt.ylabel("Training Time (seconds)")
    plt.title("Training Time vs N_WORKERS")
    plt.xticks(xs)
    plt.grid(True)
    plt.savefig("experimental_results/time_vs_workers.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()