from pathlib import Path
import os

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import numpy as np
from io import StringIO

load_dotenv("../java/.env")
MAX_POINTS = 200

PLOT_STYLE = "bar"     # options: "line", "bar" 

# ============================================================================================

BAR_COMPATIBLE_MODES = {
    "N_WORKERS",
    "FILTER_ENABLED",
    "THRESHOLD",
    "FILTER_STRENGTH",
    "FULLY_INFORMED_VS_CLASSICAL",
    "DIMENSIONALITY",
    "TOPOLOGY",
}

# ============================================================================================

MODE_COLORS = {
    "N_WORKERS": "tab:orange",
    "FILTER_STRENGTH": "firebrick",
    "TOPOLOGY": "khaki",
    "LOSS_FUNCTIONS": "tab:green"
}

# Color options:
    # "gold" 
    # "goldenrod" 
    # "darkgoldenrod"
    # "tab:orange",
    # "tab:green",
    # "tab:blue",
    # "coral"      
    # "tomato"    
    # "darkorange" 
    # "khaki"  

# ============================================================================================
def get_plot_color(mode: str) -> str | None:
    return MODE_COLORS.get(mode)

# ============================================================================================

def should_use_bar_plot(mode: str) -> bool:
    return PLOT_STYLE == "bar" and mode in BAR_COMPATIBLE_MODES

# ============================================================================================

def parse_loss_functions_csv(csv_path: Path) -> list[tuple[pd.DataFrame, dict]]:
    experiments = []

    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    current_block = []
    current_meta = None

    for line in lines:
        if line.startswith("MONITORING_ITER,"):
            if current_block:
                df = pd.read_csv(StringIO("\n".join(current_block)))
                experiments.append((df, current_meta or {}))
                current_block = []
                current_meta = None

            current_block = [line]

        elif line.startswith("LOSS_FUNCTION,"):
            current_meta = {}
            parts = line.split(",")
            for i in range(0, len(parts) - 1, 2):
                key = parts[i].strip()
                value = parts[i + 1].strip()
                current_meta[key] = value

            if current_block:
                df = pd.read_csv(StringIO("\n".join(current_block)))
                experiments.append((df, current_meta))
                current_block = []
                current_meta = None

        else:
            current_block.append(line)

    if current_block:
        df = pd.read_csv(StringIO("\n".join(current_block)))
        experiments.append((df, current_meta or {}))

    return experiments

# ============================================================================================

def plot_loss_functions_experiments(csv_path: Path, outdir: Path, mode):
    experiments = parse_loss_functions_csv(csv_path)
    saved = []

    for df, meta in experiments:
        if "MONITORING_ITER" not in df.columns or "ACCURACY" not in df.columns:
            continue

        df["MONITORING_ITER"] = pd.to_numeric(df["MONITORING_ITER"], errors="coerce")
        df["ACCURACY"] = pd.to_numeric(df["ACCURACY"], errors="coerce")
        df = df.dropna(subset=["MONITORING_ITER", "ACCURACY"]).sort_values("MONITORING_ITER")
        df = downsample_df(df, MAX_POINTS)

        xs_plot = df["MONITORING_ITER"].tolist()
        ys_plot = df["ACCURACY"].tolist()

        loss_function = meta.get("LOSS_FUNCTION", "UNKNOWN")
        combine_loss = meta.get("COMBINE_LOSS", "UNKNOWN")
        regularizer = meta.get("REGULARIZER", "UNKNOWN")
        safe_loss = loss_function.lower()
        safe_combine = combine_loss.lower()
        safe_regularizer = regularizer.lower()
        
        plot_color = get_plot_color(mode)

        plt.figure()
        plt.plot(xs_plot, ys_plot, marker="o", markersize=3, markeredgewidth=0.3, color=plot_color)
        plt.ylim(0, 1)
        plt.yticks(np.linspace(0, 1, 11))
        plt.xlabel("MONITORING_ITER")
        plt.ylabel("ACCURACY")
        plt.title(f"{loss_function} - {combine_loss} - {regularizer}")
        plt.grid(True)
        plt.tight_layout()
        
        outpath = outdir / (
            f"{csv_path.stem}_accuracy_{safe_loss}_{safe_combine}_{safe_regularizer}.png"
        )
        
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close()
        saved.append(outpath)
        
        print(
            f"Parsed block: LOSS_FUNCTION={loss_function}, "
            f"COMBINE_LOSS={combine_loss}, REGULARIZER={regularizer}"
        )

    return saved

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

def main():
        
    mode = os.getenv("EXPERIMENTATION_MODE", "").strip()
    print(mode)
    experimentation_dir = os.getenv("EXPERIMENTATION_DIR", "").strip()
    csv_dir = f"../java/{experimentation_dir}"
    
    if not experimentation_dir:
        raise RuntimeError("EXPERIMENTATION_DIR is not set in ../java/.env")

    # =================================================================================================

    if mode == "THRESHOLD":
        csv_path = Path(f"{csv_dir}/results_threshold.csv")
        xcol = "LOSS_THRESHOLD_DIFF"
        xlabel = "LOSS_THRESHOLD_DIFF (T)"
        suffix = "threshold"
        plots = [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs Threshold (T)", "accuracy"),
            ("PBEST_BYTES_EST", "Estimated pBest bytes", "pBest Bytes vs Threshold (T)", "bytes"),
            ("TOTAL_BYTES_SENT", "TOTAL_BYTES_SENT", "Bytes vs FILTER_ENABLED", "bytes"),
            ("TOTAL_MESSAGES_SENT", "TOTAL_MESSAGES_SENT", "Messages vs FILTER_ENABLED", "messages"),
        ]
    
    # =================================================================================================
    # if its loss functions, the .csv that needs to be read is very different. So we make a custom function for it, to read_csv line by line

    elif mode == "LOSS_FUNCTIONS":
        csv_path = Path(f"{csv_dir}/results_loss_functions.csv")
        outdir = csv_path.parent

        suffix = "loss_functions"
        saved = plot_loss_functions_experiments(csv_path, outdir, mode)

        print("Mode:", mode)
        print("CSV :", csv_path)
        print("Saved:")
        for p in saved:
            print(" -", p)
        return
    
    # =================================================================================================
    
    elif mode == "FILTER_ENABLED":
        csv_path = Path(f"{csv_dir}/results_filter_enabled.csv")
        xcol = "FILTER_ENABLED"
        xlabel = "FILTER_ENABLED"
        suffix = "filter_enabled"
        plots = [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs FILTER_ENABLED", "accuracy"),
            ("TOTAL_ELAPSED", "TOTAL_ELAPSED (sec)", "Time vs FILTER_ENABLED", "time"),
            ("TOTAL_BYTES_SENT", "TOTAL_BYTES_SENT", "Bytes vs FILTER_ENABLED", "bytes"),
            ("TOTAL_MESSAGES_SENT", "TOTAL_MESSAGES_SENT", "Messages vs FILTER_ENABLED", "messages"),
        ]
        
    # =================================================================================================
    
    elif mode == "FILTER_STRENGTH":
        csv_path = Path(f"{csv_dir}/results_strength.csv")
        xcol = "STRENGTH_CODE"  
        xlabel = "FILTER_STRENGTH_INDEX"
        suffix = "strength"
        plots = [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs FILTER_STRENGTH", "accuracy"),
            ("TOTAL_ELAPSED", "TOTAL_ELAPSED (sec)", "Time vs FILTER_STRENGTH", "time"),
            ("TOTAL_BYTES_SENT", "TOTAL_BYTES_SENT", "Bytes vs FILTER_STRENGTH", "bytes"),
            ("TOTAL_MESSAGES_SENT", "TOTAL_MESSAGES_SENT", "Messages vs FILTER_STRENGTH", "messages"),
        ]
        
    # =================================================================================================
    
    elif mode == "FULLY_INFORMED_VS_CLASSICAL":
        csv_path = Path(f"{csv_dir}/results_fully_informed_vs_classical.csv")
        xcol = "FULLY_INFORMED" 
        xlabel = "FULLY_INFORMED"
        suffix = "fully_informed_vs_classical"
        plots = [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs FULLY_INFORMED_VS_CLASSICAL", "accuracy"),
            ("TOTAL_ELAPSED", "TOTAL_ELAPSED (sec)", "Time vs FULLY_INFORMED_VS_CLASSICAL", "time"),
            ("TOTAL_BYTES_SENT", "TOTAL_BYTES_SENT", "Bytes vs FULLY_INFORMED_VS_CLASSICAL", "bytes"),
            ("TOTAL_MESSAGES_SENT", "TOTAL_MESSAGES_SENT", "Messages vs FULLY_INFORMED_VS_CLASSICAL", "messages"),
        ]
        
    # =================================================================================================

    elif mode == "MONITORING_ITERATIONS":
        csv_path = Path(f"{csv_dir}/results_monitoring_iterations.csv")
        xcol = "MONITORING_ITER"
        xlabel = "MONITORING_ITER"
        suffix = "monitoring"
        plots = [
            ("ACCURACY", "ACCURACY", "Accuracy vs Monitoring Iteration", "accuracy"),
        ]

    # =================================================================================================

    elif mode == "N_WORKERS":
        csv_path = Path(f"{csv_dir}/results_n_workers.csv")
        xcol = "N_WORKERS"
        xlabel = "N_WORKERS"
        suffix = "workers"
        plots = [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs N_WORKERS", "accuracy"),
            ("TOTAL_ELAPSED", "TOTAL_ELAPSED (sec)", "Time vs N_WORKERS", "time"),
            ("TOTAL_BYTES_SENT", "TOTAL_BYTES_SENT", "Bytes vs N_WORKERS", "bytes"),
            ("TOTAL_MESSAGES_SENT", "TOTAL_MESSAGES_SENT", "Messages vs N_WORKERS", "messages"),
        ]
        
    # =================================================================================================

    elif mode == "DIMENSIONALITY":
        csv_path = Path(f"{csv_dir}/results_dimensionality.csv")
        xcol = "DIMENSIONALITY"
        xlabel = "DIMENSIONALITY"
        suffix = "dimensionality"
        plots = [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs DIMENSIONALITY", "accuracy"),
            ("TOTAL_ELAPSED", "TOTAL_ELAPSED (sec)", "Time vs DIMENSIONALITY", "time"),
            ("TOTAL_BYTES_SENT", "TOTAL_BYTES_SENT", "Bytes vs DIMENSIONALITY", "bytes"),
            ("TOTAL_MESSAGES_SENT", "TOTAL_MESSAGES_SENT", "Messages vs DIMENSIONALITY", "messages"),
        ]
            
    # =================================================================================================
    
    elif mode == "TOPOLOGY":
        csv_path = Path(f"{csv_dir}/results_topology.csv")
        xcol = "NEIGHBORHOOD_TOPOLOGY"
        xlabel = "TOPOLOGY"
        suffix = "topology"
        plots = [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs TOPOLOGY", "accuracy"),
            ("TOTAL_ELAPSED", "TOTAL_ELAPSED (sec)", "Time vs TOPOLOGY", "time"),
            ("TOTAL_BYTES_SENT", "TOTAL_BYTES_SENT", "Bytes vs TOPOLOGY", "bytes"),
            ("TOTAL_MESSAGES_SENT", "TOTAL_MESSAGES_SENT", "Messages vs TOPOLOGY", "messages"),
        ]
            
    # =================================================================================================

    else: 
        print("Wrong experimentation mode selected, exiting")
        exit(-1)
        
    # =================================================================================================
    
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

    if "TOTAL_MESSAGES_SENT_PBEST" in df.columns and "TOTAL_MESSAGES_SENT" in df.columns and "TOTAL_BYTES_SENT" in df.columns:
        df["PBEST_BYTES_EST"] = compute_pbest_bytes(df)

    xs = df[xcol].tolist()

    saved = []
    
    # ===========================================================================================
    
    for ycol, ylabel, title, tag in plots:  
        
        if ycol not in df.columns:
            raise KeyError(f"CSV missing y column '{ycol}' needed for plot '{title}'. Columns: {list(df.columns)}")

        ys = pd.to_numeric(df[ycol], errors="coerce")
        mask = ys.notna()
        xs_plot = df.loc[mask, xcol].tolist()
        ys_plot = ys.loc[mask].tolist()

        # ===========================================================================================
    
        if mode == "MONITORING_ITERATIONS":

            plot_df = df.loc[mask, [xcol, ycol]].copy()
            plot_df[xcol] = pd.to_numeric(plot_df[xcol], errors="coerce")
            plot_df[ycol] = pd.to_numeric(plot_df[ycol], errors="coerce")
            plot_df = plot_df.dropna(subset=[xcol, ycol]).sort_values(xcol)
            plot_df = downsample_df(plot_df, MAX_POINTS)

            xs_plot = plot_df[xcol].tolist()
            ys_plot = pd.to_numeric(plot_df[ycol], errors="coerce").tolist()
       
        plot_color = get_plot_color(mode)
                    
        # ===========================================================================================
        # Actually plot:

        plt.figure()

        use_bar = should_use_bar_plot(mode)

        if mode == "MONITORING_ITERATIONS":
            plt.plot(xs_plot, ys_plot, marker="o", markersize=3, markeredgewidth=0.3, color=plot_color)

        elif use_bar:
            positions = np.arange(len(xs_plot))
            plt.bar(positions, ys_plot, width=0.35, color=plot_color)

        elif mode in {"DIMENSIONALITY", "TOPOLOGY"}:
            positions = np.arange(len(xs_plot))
            plt.plot(positions, ys_plot, marker="o", color=plot_color)

        else:
            plt.plot(xs_plot, ys_plot, marker="o", color=plot_color)
        
        if ycol in {"GBEST_ACC", "ACCURACY"}:
            plt.ylim(0, 1)
            plt.yticks(np.linspace(0, 1, 11))
            
        else:
            ymin, ymax = 0, max(ys_plot)
            plt.ylim(ymin, ymax * 1.15)
            upper = ymax * 1.15 if ymax > 0 else 1
            plt.yticks(np.linspace(0, upper, 11))
            
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        ax = plt.gca()
        ax.set_axisbelow(True)

        if use_bar:
            plt.grid(True)
        else:
            plt.grid(True)  

        # ===========================================================================================
        # xticks:
        
        if mode == "MONITORING_ITERATIONS":
            tick_count = 10

            if len(xs_plot) > tick_count:
                tick_idx = np.linspace(0, len(xs_plot) - 1, tick_count, dtype=int)
            else:
                tick_idx = range(len(xs_plot))

            tick_positions = [xs_plot[i] for i in tick_idx]
            tick_labels = [f"{xs_plot[i]:.1f}" for i in tick_idx]

            plt.xticks(tick_positions, tick_labels, ha="right")

        elif use_bar:
            plt.xticks(positions, [str(x) for x in xs_plot])

        elif mode in {"DIMENSIONALITY", "TOPOLOGY"}:
            plt.xticks(positions, xs_plot)

        else:
            plt.xticks(xs_plot)
            
        # ===========================================================================================

        plt.tight_layout()
        
        outpath = outdir / f"{csv_path.stem}_{tag}.png"
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close()
        saved.append(outpath)
        
        if mode == "MONITORING_ITERATIONS":
            time_xcol = "TIME_SEC"

            if time_xcol not in df.columns:
                raise KeyError(f"CSV missing x column '{time_xcol}'. Columns: {list(df.columns)}")

            df[time_xcol] = pd.to_numeric(df[time_xcol], errors="coerce")

            plot_df = df[[time_xcol, "ACCURACY"]].copy()
            plot_df[time_xcol] = pd.to_numeric(plot_df[time_xcol], errors="coerce")
            plot_df["ACCURACY"] = pd.to_numeric(plot_df["ACCURACY"], errors="coerce")
            plot_df = plot_df.dropna(subset=[time_xcol, "ACCURACY"]).sort_values(time_xcol)

            plot_df = downsample_df(plot_df, MAX_POINTS)

            xs_plot = plot_df[time_xcol].tolist()
            ys_plot = plot_df["ACCURACY"].tolist()
            
            plot_color = get_plot_color(mode)

            plt.figure()
            plt.plot(xs_plot, ys_plot, marker="o", markersize=3, markeredgewidth=0.3, color=plot_color)
            plt.ylim(0, 1)
            plt.yticks(np.linspace(0, 1, 11))
            plt.xlabel("TIME_SEC")
            plt.ylabel("ACCURACY")
            plt.title("Accuracy vs Time")
            plt.grid(True)
            plt.tight_layout()

            outpath = outdir / f"{csv_path.stem}_accuracy_vs_time.png"
            plt.savefig(outpath, dpi=200, bbox_inches="tight")
            plt.close()
            saved.append(outpath)

    print("Mode:", mode or "(default N_WORKERS)")
    print("CSV :", csv_path)
    print("Saved:")
    for p in saved:
        print(" -", p)

# ===========================================================================================

if __name__ == "__main__":
    main()