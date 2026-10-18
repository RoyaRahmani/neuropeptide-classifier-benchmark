import itertools
import os
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
import pandas as pd
from tqdm import tqdm  # Optional: for a visual progress bar

# 20 canonical amino acids
STANDARD_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# All 400 possible dipeptide pairs ('AA', 'AC', ..., 'YY')
ALL_DIPEPTIDES = [
    f"{a}{b}" for a, b in itertools.product(STANDARD_AMINO_ACIDS, repeat=2)
]


def clean_sequence(seq: str) -> str:
    """Uppercase and filter out non-canonical or ambiguous residues."""
    seq = str(seq).upper().strip()
    valid_aas = set(STANDARD_AMINO_ACIDS)
    return "".join([aa for aa in seq if aa in valid_aas])


def extract_features_from_sequence(seq_str: str) -> dict:
    """Calculate 429 numerical descriptors (AAC, DPC, and ProtParam metrics)."""
    features = {}
    length = len(seq_str)

    if length < 2:
        return features

    # 1. Amino Acid Composition (AAC: 20 features)
    analysis = ProteinAnalysis(seq_str)
    raw_aac = analysis.amino_acids_percent
    for aa in STANDARD_AMINO_ACIDS:
        features[f"AAC_{aa}"] = raw_aac.get(aa, 0.0)

    # 2. Dipeptide Composition (DPC: 400 features)
    dpc_counts = {dp: 0 for dp in ALL_DIPEPTIDES}
    for i in range(length - 1):
        pair = seq_str[i : i + 2]
        if pair in dpc_counts:
            dpc_counts[pair] += 1

    total_dipeptides = length - 1
    for dp, count in dpc_counts.items():
        features[f"DPC_{dp}"] = count / total_dipeptides

    # 3. Physicochemical Properties via ProtParam (9 features)
    features["length"] = length
    features["molecular_weight"] = analysis.molecular_weight()
    features["aromaticity"] = analysis.aromaticity()
    features["instability_index"] = analysis.instability_index()
    features["isoelectric_point"] = analysis.isoelectric_point()
    features["gravy"] = analysis.gravy()
    features["charge_pH7"] = analysis.charge_at_pH(7.0)

    helix, turn, sheet = analysis.secondary_structure_fraction()
    features["sec_struct_helix"] = helix
    features["sec_struct_turn"] = turn
    features["sec_struct_sheet"] = sheet

    return features


def process_benchmark_csv(input_csv_path: str, output_csv_path: str):
    """Load benchmark CSV, compute features for each sequence, and save the feature matrix."""
    print(f"Reading {input_csv_path}...")
    df = pd.read_csv(input_csv_path)

    # Ensure expected column names exist
    assert "seq" in df.columns and "label" in df.columns, (
        "CSV must contain 'seq' and 'label' columns."
    )

    records = []
    skipped = 0

    # Iterate through rows and extract features
    for idx, row in df.iterrows():
        clean_seq = clean_sequence(row["seq"])

        # Filter peptides that are too short to calculate secondary structures
        if len(clean_seq) < 3:
            skipped += 1
            continue

        feat_dict = extract_features_from_sequence(clean_seq)
        feat_dict["sequence"] = clean_seq
        feat_dict["label"] = int(row["label"])
        records.append(feat_dict)

    feature_df = pd.DataFrame(records)

    # Shuffle rows within the set
    feature_df = feature_df.sample(frac=1.0, random_state=42).reset_index(
        drop=True
    )

    # Save to disk
    feature_df.to_csv(output_csv_path, index=False)
    print(
        f"Saved {len(feature_df)} samples to {output_csv_path} (Skipped {skipped} invalid sequences).\n"
    )


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    # Convert training set
    process_benchmark_csv(
        input_csv_path="E:/neuropeptide-classifier-benchmark/train.csv",
        output_csv_path="data/processed/train_features.csv",
    )

    # Convert independent test set
    process_benchmark_csv(
        input_csv_path="E:/neuropeptide-classifier-benchmark/test.csv",
        output_csv_path="data/processed/test_features.csv",
    )
