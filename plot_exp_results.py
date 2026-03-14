from pathlib import Path
import os

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import numpy as np

load_dotenv("java/.env")

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
    experimentation_dir = os.getenv("EXPERIMENTATION_DIR", "").strip()

    if not experimentation_dir:
        raise RuntimeError("EXPERIMENTATION_DIR is not set in java/.env")

    # =================================================================================================

    if mode == "THRESHOLD":
        csv_path = Path(f"java/{experimentation_dir}/results_threshold.csv")
        xcol = "LOSS_THRESHOLD_DIFF"
        xlabel = "LOSS_THRESHOLD_DIFF (T)"
        suffix = "threshold"
        plots = [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs Threshold (T)", "accuracy"),
            ("PBEST_BYTES_EST", "Estimated pBest bytes", "pBest Bytes vs Threshold (T)", "bytes"),
        ]
    
    # =================================================================================================
    
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
        
    # =================================================================================================
    
    elif mode == "SEVERITY_OF_FILTER":
        csv_path = Path(f"java/{experimentation_dir}/results_severity_of_filter.csv")
        xcol = "SEVERITY_CODE"  # This is what is needed for the pandas to find the correct code
        xlabel = "SEVERITY_CODE"
        suffix = "severity_of_filter"
        plots = [
            ("GBEST_ACC", "GBEST_ACC", "Accuracy vs FILTER_ENABLED", "accuracy"),
            ("TOTAL_ELAPSED", "TOTAL_ELAPSED (sec)", "Time vs FILTER_ENABLED", "time"),
            ("TOTAL_BYTES_SENT", "TOTAL_BYTES_SENT", "Bytes vs FILTER_ENABLED", "bytes"),
        ]
        
    # =================================================================================================
    
    elif mode == "FULLY_INFORMED_VS_CLASSICAL":
        csv_path = Path(f"java/{experimentation_dir}/results_fully_informed_vs_classical.csv")
        xcol = "FULLY_INFORMED"  # This is what is needed for the pandas to find the correct code
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
        # put your monitoring csv here, e.g. java/<dir>/monitoring_accuracy.csv
        csv_path = Path(f"java/{experimentation_dir}/results_monitoring_iterations.csv")
        xcol = "MONITORING_ITER"
        xlabel = "MONITORING_ITER"
        suffix = "monitoring"
        plots = [
            ("ACCURACY", "ACCURACY", "Accuracy vs Monitoring Iteration", "accuracy"),
        ]

    # =================================================================================================

    elif mode == "N_WORKERS":
        # default: N_WORKERS experiments
        csv_path = Path(f"java/{experimentation_dir}/results_n_workers.csv")
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
        csv_path = Path(f"java/{experimentation_dir}/results_dimensionality.csv")
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
        csv_path = Path(f"java/{experimentation_dir}/results_topology.csv")
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
    # data frame processing 
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    outdir = csv_path.parent
    df = pd.read_csv(csv_path)

    if xcol not in df.columns:
        raise KeyError(f"CSV missing x column '{xcol}'. Columns: {list(df.columns)}")

    if mode in {"N_WORKERS", "MODEL_VERSION", "RING_RADIUS", "DIMENSIONALITY", "MONITORING_ITERATIONS"}:
        df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
        df = df.dropna(subset=[xcol]).sort_values(xcol)
    elif mode == "TOPOLOGY":
        df[xcol] = df[xcol].astype(str)

    # pbest estimation only when those columns exist (won't run for MONITORING)
    if "TOTAL_MESSAGES_SENT_PBEST" in df.columns and "TOTAL_MESSAGES_SENT" in df.columns and "TOTAL_BYTES_SENT" in df.columns:
        df["PBEST_BYTES_EST"] = compute_pbest_bytes(df)

    xs = df[xcol].tolist()

    saved = []
    
    # ===========================================================================================
    
    for ycol, ylabel, title, tag in plots:  
        
        if ycol not in df.columns:
            raise KeyError(f"CSV missing y column '{ycol}' needed for plot '{title}'. Columns: {list(df.columns)}")

        ys = pd.to_numeric(df[ycol], errors="coerce")
        # for monitoring, keep NaNs out rather than forcing to 0
        mask = ys.notna()
        xs_plot = df.loc[mask, xcol].tolist()
        ys_plot = ys.loc[mask].tolist()

        # ===========================================================================================
    
        if mode == "MONITORING_ITERATIONS":
            
            MAX_POINTS = int(os.getenv("MAX_PLOT_POINTS", "1000"))

            plot_df = df.loc[mask, [xcol, ycol]].copy()
            plot_df = plot_df.sort_values(xcol)
            plot_df = downsample_df(plot_df, MAX_POINTS)

            xs_plot = plot_df[xcol].tolist()
            ys_plot = pd.to_numeric(plot_df[ycol], errors="coerce").tolist()
       
            
        plt.figure()
        
        # ===========================================================================================
        # plot points:
        
        if mode == "MONITORING_ITERATIONS":
            plt.plot(xs_plot, ys_plot)
            
        elif mode in {"DIMENSIONALITY", "TOPOLOGY"}:
            positions = np.arange(len(xs_plot))
            plt.plot(positions, ys_plot, marker="o")
            
        else:
            plt.plot(xs_plot, ys_plot, marker="o")
            
        if ycol == "GBEST_ACC":
            plt.ylim(0, 1)
            plt.yticks(np.linspace(0, 1, 11))
            
        else:
            plt.ylim(bottom=0)
            
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        
        # ===========================================================================================
        # xticks:
        
        if mode == "MONITORING_ITERATIONS":
            tick_count = 10
            if len(xs_plot) > tick_count:
                tick_idx = np.linspace(0, len(xs_plot) - 1, tick_count, dtype=int)
                plt.xticks([xs_plot[i] for i in tick_idx])
                
        elif mode in {"DIMENSIONALITY", "TOPOLOGY"}:
            plt.xticks(positions, xs_plot)
            
        else:
            plt.xticks(xs_plot)
            # plt.xticks(xs_plot, rotation=30, ha="right")
        
        # ===========================================================================================

        plt.tight_layout()
        
        outpath = outdir / f"{csv_path.stem}_{tag}_vs_{suffix}.png"
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