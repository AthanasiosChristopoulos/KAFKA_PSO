import re
import time
import csv
import signal
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

# WORKERS_LIST = [3, 5, 8]
WORKERS_LIST = [2, 4, 6]
# WORKERS_LIST = [2, 3, 4, 5, 6]
# WORKERS_LIST = [2, 4, 6, 8]

RUN_STREAMS_SCRIPT = "./run_streams.sh"

ACCURACY_REGEX = re.compile(    # REGEX == Regular Expression
    r",\s*bestAccuracy:\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE
)

END_PATTERNS = [
    re.compile(r"\s*Training is over\s*", re.IGNORECASE),
]

TRAINING_TIME_REGEX = re.compile(
    r"Training\s+is\s+over,\s*ElapsedTime:\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE
)

# NEW: parse times printed by Java
WORKER_TIME_REGEX = re.compile(
    r"\[Worker\s+(\d+)\s*\]\s+(?:Wall time|Elapsed time):\s*([0-9]*\.?[0-9]+)\s*seconds",
    re.IGNORECASE
)
COORD_TIME_REGEX = re.compile(
    r"\[Coordinator\]\s+(?:Wall time|Elapsed time):\s*([0-9]*\.?[0-9]+)\s*seconds",
    re.IGNORECASE
)

# ========================================================================================

def workers_tag(workers):
    return "-".join(map(str, workers))

# ========================================================================================

def run_once(n_workers, log_path):

    cmd = [RUN_STREAMS_SCRIPT, str(n_workers)]

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

    coordinator_time_sec = None  
    worker_times = {}               # workerId -> seconds

    with log_path.open("a", encoding="utf-8") as experiment_log_file:

        for line in p.stdout:  # write from the current stdout / terminal, to the experiment_log_file

            print(line, end="")
            experiment_log_file.write(line)

            m = ACCURACY_REGEX.search(line)
            if m:
                best_acc = float(m.group(1))

            wm = WORKER_TIME_REGEX.search(line)
            if wm:
                wid = int(wm.group(1))
                wsec = float(wm.group(2))
                worker_times[wid] = wsec

            cm = COORD_TIME_REGEX.search(line)
            if cm:
                coordinator_time_sec = float(cm.group(1))

            tm = TRAINING_TIME_REGEX.search(line)
            if tm:
                elapsed = float(tm.group(1))
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

    last_worker_time_sec = max(worker_times.values()) if worker_times else None     # for worker elapsed time, only consider last worker elapsed time

    result_line = (
        f"Results for N_WORKERS = {n_workers} : "
        f"best accuracy: {best_acc} - "
        f"training time: {elapsed:.3f} - "
        f"coordinator elapsed time: {coordinator_time_sec} - "
        f"last worker elapsed time: {last_worker_time_sec}\n"
    )
    print(result_line, end="")

    with log_path.open("a", encoding="utf-8") as f:
        f.write(result_line)


    return best_acc, elapsed, coordinator_time_sec, last_worker_time_sec

# ========================================================================================

def main():

    logs_dir = Path("./experimental_results")
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / f"experiment_log_workers_{workers_tag(WORKERS_LIST)}.log"

    log_path.write_text("", encoding="utf-8")   # Reset log file

    Path("experimental_results").mkdir(parents=True, exist_ok=True)

    results = []

    for n in WORKERS_LIST:

        header = "\n" + "=" * 80 + "\n" + f"RUNNING EXPERIMENT: N_WORKERS={n}\n" + "=" * 80 + "\n"
        print(header, end="")

        with log_path.open("a", encoding="utf-8") as f:
            f.write(header)

        acc, secs, coord_secs, last_worker_secs = run_once(n, log_path)

        if acc is None:
            print(f"[WARN] No accuracy parsed for N_WORKERS={n}. Check log: {log_path}")

        results.append({
            "N_WORKERS": n,
            "ACCURACY": acc,
            "TRAIN_TIME_SEC": secs,
            "COORDINATOR_ELAPSED_TIME_SEC": coord_secs,
            "LAST_WORKER_ELAPSED_TIME_SEC": last_worker_secs,
        })

    csv_path = Path("experimental_results/results.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "N_WORKERS",
                "ACCURACY",
                "TRAIN_TIME_SEC",
                "COORDINATOR_ELAPSED_TIME_SEC",
                "LAST_WORKER_ELAPSED_TIME_SEC",
            ],
        )
        w.writeheader()
        w.writerows(results)

    xs = [r["N_WORKERS"] for r in results]

    # Plot 1: Accuracy vs N_WORKERS
    accs = [r["ACCURACY"] for r in results]
    plt.figure()
    plt.plot(xs, accs, marker="o")
    plt.ylim(bottom=0)
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
    plt.ylim(bottom=0)
    plt.xlabel("N_WORKERS")
    plt.ylabel("Training Time (seconds)")
    plt.title("Training Time vs N_WORKERS")
    plt.xticks(xs)
    plt.grid(True)
    plt.savefig("experimental_results/time_vs_workers.png", dpi=200)
    plt.close()

# ========================================================================================

if __name__ == "__main__":
    main()