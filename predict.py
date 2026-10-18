"""Command-Line Interface (CLI) for Neuropeptide Prediction and Attention Extraction.

Usage:
    python predict.py --seq "RPKPQQFFGLMGKRDAG"
    python predict.py --fasta "examples.fasta" --output "predictions.csv"
"""

import argparse
import os
import sys
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from src.models import PurePLMNet

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DEFAULT_CHECKPOINT = "checkpoints/best_pure_plm_650M.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict neuropeptide bioactivity and extract attention profiles."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seq", type=str, help="Single amino acid sequence")
    group.add_argument(
        "--fasta", type=str, help="Path to input FASTA / text file"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save prediction results CSV",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision probability threshold (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_pipeline(checkpoint_path: str, device: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. Download it to checkpoints/ first."
        )

    print(f"[+] Loading ESM-2 tokenizer and backbone ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    esm_backbone = AutoModel.from_pretrained(MODEL_NAME).to(device)
    esm_backbone.eval()
    for p in esm_backbone.parameters():
        p.requires_grad = False

    print(f"[+] Loading PurePLMNet classifier from {checkpoint_path}...")
    model = PurePLMNet(esm_dim=1280, lstm_hidden=128).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    return tokenizer, esm_backbone, model


def predict_sequence(
    sequence: str, tokenizer, esm_backbone, model, device: str
):
    clean_seq = sequence.strip().upper()
    encoded = tokenizer(
        clean_seq, return_tensors="pt", add_special_tokens=True
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        esm_out = esm_backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        tokens = esm_out.last_hidden_state
        logits, attn_weights = model(tokens=tokens, mask=attention_mask.bool())
        prob = torch.sigmoid(logits).item()

        # Slice out real residue attention (drop <cls> and <eos>)
        attn = attn_weights[0, 1:-1].cpu().numpy()
        attn = attn / (attn.sum() + 1e-9)

    return prob, attn


def main():
    args = parse_args()
    tokenizer, esm_backbone, model = load_pipeline(args.checkpoint, args.device)

    sequences = []
    identifiers = []

    if args.seq:
        sequences.append(args.seq)
        identifiers.append("User_Query")
    else:
        # Simple FASTA/text parser
        with open(args.fasta, "r") as f:
            current_id = "seq_1"
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_seq:
                        sequences.append("".join(current_seq))
                        identifiers.append(current_id)
                        current_seq = []
                    current_id = line[1:].split()[0]
                elif line:
                    current_seq.append(line)
            if current_seq:
                sequences.append("".join(current_seq))
                identifiers.append(current_id)

    print(f"\n[+] Running inference on {len(sequences)} sequence(s)...\n")
    results = []

    for seq_id, seq in zip(identifiers, sequences):
        prob, attn = predict_sequence(
            seq, tokenizer, esm_backbone, model, args.device
        )
        prediction = (
            "Neuropeptide" if prob >= args.threshold else "Non-Neuropeptide"
        )

        results.append(
            {
                "ID": seq_id,
                "Sequence": seq,
                "Length": len(seq),
                "Probability": round(prob, 4),
                "Prediction": prediction,
            }
        )

        print(
            f"ID: {seq_id:<15} | Length: {len(seq):<4} | Prob: {prob:.4f} | Prediction: {prediction}"
        )

    res_df = pd.DataFrame(results)
    if args.output:
        res_df.to_csv(args.output, index=False)
        print(f"\n[+] Saved complete predictions to: {args.output}")


if __name__ == "__main__":
    main()
