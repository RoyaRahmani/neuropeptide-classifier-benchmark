# Dual-Stream PLM Benchmark for Neuropeptide Identification and Mechanistic Explainability

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace ESM-2](https://img.shields.io/badge/Model-ESM--2%20650M-yellow.svg)](https://huggingface.co/facebook/esm2_t33_650M_UR50D)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository provides a deep learning framework for bioactive neuropeptide identification from sequence data. The framework leverages **Meta's ESM-2 (650M parameter)** representations paired with **Bidirectional Long Short-Term Memory (BiLSTM)** and **Attention Pooling**, complemented by dense biophysical priors (amino acid composition, charge, hydropathy, secondary structure).

### Tripartite Ablation Benchmark (Independent Test Set: N = 888)

| Architecture | Representation Paradigm | Accuracy | Sensitivity | Specificity | MCC | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Model A** | Handcrafted Biophysics (SVM RBF) | 0.9009 | 0.9122 | 0.8896 | 0.8020 | 0.9547 |
| **Model B** | **Pure PLM (ESM-2 650M + BiLSTM)** | **0.9234** | **0.9144** | **0.9324** | **0.8470** | **0.9692** |
| **Model C** | Dual-Stream Gated (650M + Biophysics) | 0.9178 | 0.9302 | 0.9054 | 0.8358 | 0.9682 |

## Mechanistic Explainability (XAI)
Using residue-level attention attribution across 150 independent test neuropeptides, the model demonstrated statistically validated biological focus:
* **Receptor-Binding Pharmacophores (F, Y, W, L, M):** **2.67x Enrichment** ($p = 8.04 \times 10^{-50}$)
* **Amidation Glycine Donors (G):** **1.80x Enrichment** ($p = 1.54 \times 10^{-12}$)
* **Dibasic Cleavage Sites (K, R):** **1.37x Enrichment** ($p = 8.32 \times 10^{-3}$)

## Quickstart

### 1. Installation
```bash
git clone [https://github.com/your-username/neuropeptide-classifier-benchmark.git](https://github.com/your-username/neuropeptide-classifier-benchmark.git)
cd neuropeptide-classifier-benchmark
pip install -r requirements.txt