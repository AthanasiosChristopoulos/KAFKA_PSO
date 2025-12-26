import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =======================================================================================
# wine quality:

LABEL_COL = "quality"   # change if your CSV uses a different name
N_CLASSES = 7           # you asked minlength=7 in bincount


def evaluate_dataset(X_train, y_train, X_test, y_test, *, n_classes: int = N_CLASSES):
    print("Train shape:", X_train.shape, "y range:", (int(y_train.min()), int(y_train.max())),
          "counts:", np.bincount(y_train, minlength=n_classes))
    print("Test  shape:", X_test.shape,  "y range:", (int(y_test.min()), int(y_test.max())),
          "counts:", np.bincount(y_test, minlength=n_classes))


def load_wine_quality_data(
    path: str = "./winequality.csv",
    *,
    nrows: int | None = 220_000,
    test_size: float = 0.2,
    random_state: int = 123,
):

    print(f"Loading from: {path}")
    df = pd.read_csv(path, sep=",", nrows=nrows)
    print("Raw df shape:", df.shape)
    print(df.head())

    if LABEL_COL not in df.columns:
        raise ValueError(f"Expected label column '{LABEL_COL}' in CSV. Found columns: {list(df.columns)}")

    y = df[LABEL_COL].astype(int).values

    # Typical wine quality labels are 3..9 (or similar). Map to 0..6 for 7 classes.
    y_min, y_max = int(y.min()), int(y.max())
    if (y_max - y_min + 1) != N_CLASSES:
        # If it's not exactly 7 distinct consecutive values, we still map to 0..(k-1)
        # BUT your evaluate_dataset expects minlength=7, so we enforce 7 bins.
        # Strategy: clip into 7 bins using quantiles -> 0..6, balanced-ish.
        # This makes your pipeline stable even if the CSV is merged/modified.
        print(f"Label range is {y_min}..{y_max} (not exactly 7 consecutive classes).")
        print("Binning quality into 7 classes via quantiles to enforce 7-class softmax task.")

        y_float = df[LABEL_COL].astype(float).values
        qs = np.quantile(y_float, np.linspace(0, 1, N_CLASSES + 1))
        # digitize into 0..6
        y = np.digitize(y
