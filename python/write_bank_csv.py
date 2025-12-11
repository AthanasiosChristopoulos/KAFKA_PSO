import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Load original CSV
df = pd.read_csv("../data/bank-additional-full.csv", sep=";")

# 2. Convert label y (yes/no) → integer (1/0)
df["y"] = (df["y"] == "yes").astype(int)

# 3. Separate features and labels
y = df["y"]
X = df.drop(columns=["y"])

# 4. One-hot encode categorical columns
X_onehot = pd.get_dummies(X, drop_first=True)

# 5. Scale all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_onehot)

# Convert back to DataFrame with column names
X_scaled_df = pd.DataFrame(X_scaled, columns=X_onehot.columns)

# 6. Add label column back as last column
processed_df = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)

# 7. Save to CSV
output_path = "../data/processed_bank.csv"
processed_df.to_csv(output_path, index=False)

print("Saved processed dataset to:", output_path)
print("Shape:", processed_df.shape)
