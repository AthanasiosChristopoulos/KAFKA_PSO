## Data Streaming / Data Pipeline:

- “We just keep feeding data and doing backprop forever.” => This almost never happens in practice
    => catastrophic forgetting of data already processed
    => Deep networks with SGD do not behave well in infinite online mode unless heavily constrained.

- Real streaming systems:
    - Pattern A — Windowed retraining (MOST COMMON)
        - Data stream is split into windows (time-based or count-based)
            For each window:
                - train 
                - converge (Loss stops improving / Weights stabilize)
        - old data fades out
    - Pattern B - Periodic Reset

