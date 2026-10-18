import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def standardize_benchmark_features(
    train_path: str = "data/processed/train_features.csv",
    test_path: str = "data/processed/test_features.csv",
    output_dir: str = "data/Standard",
):
    """Fit StandardScaler strictly on training features, transform both splits,

    and save standardized CSVs preserving metadata columns.
    """
    # 1. Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"[+] Loading training set from: {train_path}")
    train_df = pd.read_csv(train_path)

    print(f"[+] Loading independent test set from: {test_path}")
    test_df = pd.read_csv(test_path)

    # 2. Identify metadata vs. numerical feature columns
    meta_cols = [
        col for col in ["id", "sequence", "label"] if col in train_df.columns
    ]
    feature_cols = [col for col in train_df.columns if col not in meta_cols]

    print(
        f"[+] Detected {len(meta_cols)} metadata columns: {meta_cols}"
    )
    print(f"[+] Detected {len(feature_cols)} numerical biophysical features")

    # 3. Fit scaler ONLY on training data to prevent data leakage
    scaler = StandardScaler()

    print("[+] Fitting StandardScaler on training set and transforming...")
    train_scaled_features = scaler.fit_transform(train_df[feature_cols].values)
    test_scaled_features = scaler.transform(test_df[feature_cols].values)

    # 4. Reconstruct DataFrames with original metadata preserved
    train_standardized_df = pd.DataFrame(
        train_scaled_features, columns=feature_cols
    )
    test_standardized_df = pd.DataFrame(
        test_scaled_features, columns=feature_cols
    )

    # Prepend metadata columns
    for col in reversed(meta_cols):
        train_standardized_df.insert(0, col, train_df[col].values)
        test_standardized_df.insert(0, col, test_df[col].values)

    # 5. Define target output paths
    train_out_path = os.path.join(output_dir, "train_features_standardized.csv")
    test_out_path = os.path.join(output_dir, "test_features_standardized.csv")

    # 6. Save to CSV
    train_standardized_df.to_csv(train_out_path, index=False)
    test_standardized_df.to_csv(test_out_path, index=False)

    print("\n" + "=" * 65)
    print("PHASE 2 COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print(f"Saved: {train_out_path} ({train_standardized_df.shape})")
    print(f"Saved: {test_out_path} ({test_standardized_df.shape})")

    # Sanity verification: training mean should be ~0, std should be ~1
    sample_feat = feature_cols[0]
    print("\n[Sanity Check on Training Feature:", sample_feat + "]")
    print(f"  Mean (target ~0.0): {train_standardized_df[sample_feat].mean():.6f}")
    print(f"  Std  (target ~1.0): {train_standardized_df[sample_feat].std():.6f}")


if __name__ == "__main__":
    standardize_benchmark_features()
