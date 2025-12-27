import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/bank-additional-full.csv", sep=";")

df["y"] = (df["y"] == "yes").astype(int)
y = df["y"]
X = df.drop(columns=["y"])
X_onehot = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_onehot)

X_scaled_df = pd.DataFrame(X_scaled, columns=X_onehot.columns)

processed_df = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)

output_path = "../data/processed_bank.csv"
processed_df.to_csv(output_path, index=False)

print("Saved processed dataset to:", output_path)
print("Shape:", processed_df.shape)
