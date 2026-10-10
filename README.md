# SentiMatrix: LoRA Fine-Tuning for Multidimensional Sentiment Analysis (Replication)

This repository replicates the main model of the paper **"SentiMatrix: Parameter-Efficient Fine-Tuning of Encoder-Based Transformers for Multidimensional Sentiment Analysis"**. The main model is **LoRA (Low-Rank Adaptation)** applied to encoder-based transformers. Only LoRA is reproduced here. The full-fine-tuning and AdaLoRA comparisons are out of scope.

LoRA is run on four of the paper's datasets (SST-2, CARER Emotion, Amazon E-Commerce 5-class, SemEval-2014 ABSA), each with 3 random seeds, on a Kaggle Tesla T4 GPU. Results are reported as mean ± std and compared with the paper's Table 3b settings and published numbers.

**Original paper:** Arafat, Akmal, Abosinnee, Orosz. *Machine Learning* (2026) 115:229. https://doi.org/10.1007/s10994-026-07161-4
**Authors' code:** https://github.com/ELTE-DSED/senti-matrix

---

## Repository Structure

```
SentiMatrix-LoRA-Replication/
│
├── Documents/
│  
│   └── s10994-026-07161-4.pdf          # original paper
│
├── Results/
│   ├── Graphs/
│   │   ├── confusion_<dataset>.png
│   │   ├── curves_<dataset>.png
│   │   └── ours_vs_paper_<dataset>.png
│   └── Tables/
│       ├── all_results.csv
│       └── results_<dataset>.csv
│
├── Src/
│   └── nlp-replication.ipynb           # complete Kaggle notebook
│
├── requirements.txt
└── README.md
```

> **Datasets are not stored in this repo.** The notebook clones the authors' repository (`ELTE-DSED/senti-matrix`) and reads the CSVs from its `Dataset/` folder.

---

## Problem Statement

Fine-tuning every parameter of a transformer for sentiment analysis is costly in memory, time and storage. This is a problem when many task-specific models are needed or when the data is small.

The paper shows that **LoRA** trains only small low-rank adapter matrices plus the classifier head. It reports up to 99.8% fewer trainable parameters while staying close to full fine-tuning in accuracy.

This project asks whether those results can be reproduced independently, using the paper's per-dataset LoRA configurations and a shared evaluation protocol.

---

## Datasets Used

| Dataset | Task | Classes | Rows used | Train / Val / Test | Model |
|---|---|---|---|---|---|
| SST-2 | Sentiment | 2 | 9,613 | 7,690 / 961 / 962 | DistilBERT, RoBERTa-base |
| CARER Emotion | Emotion detection | 6 | 25,000 (subset) | 20,000 / 2,500 / 2,500 | DistilBERT |
| Amazon E-Commerce | Review rating | 5 | 22,641 | 18,112 / 2,264 / 2,265 | BERT-base-multilingual |
| SemEval-2014 ABSA | Aspect-based sentiment | 3 | 5,914 | 4,731 / 591 / 592 | DeBERTa-v3-base |

All CSVs come from the authors' repository: https://github.com/ELTE-DSED/senti-matrix/tree/main/Dataset

---

## Methodology

**Stage 1: Data preparation.** Each CSV is loaded and split into 80% train, 10% validation and 10% test. The split is stratified with `random_state=42`, which gives the same test set as the authors' code. Text is tokenised with dynamic padding.

**Stage 2: LoRA configuration.** LoRA adapters are injected into the attention projections with the `peft` library, and the classification head is trained fully (`modules_to_save=["classifier"]`). Rank, alpha, dropout, target modules, learning rate, batch size, epochs and weight decay follow the paper's **Table 3b** for each dataset (see Hyperparameters below).

**Stage 3: Training.** The setup is fp16 mixed precision with a cosine learning-rate schedule, 10% warmup (3% for Emotion) and early stopping on validation loss (patience 3, or 1 for Emotion). Each experiment runs with seeds **42, 43 and 44**.

**Stage 4: Evaluation.** Metrics are computed on the held-out test set:

- Accuracy, Precision, Recall and F1 (macro-averaged; weighted for ABSA)
- Similarity: cosine similarity between the predicted probability vector and the one-hot label
- Confidence: mean of the maximum softmax probability
- ECE: Expected Calibration Error, 10 equal-width bins
- Brier score
- Trainable parameters, training time and peak GPU memory

**Stage 5: Reporting.** The notebook produces comparison tables against the paper, confusion matrices, loss and validation curves, and an "ours vs paper" plot with the paper's ±std.

---

## Results

### Summary (test set, mean ± std over 3 seeds)

| Dataset | Model | Paper Acc | **Ours Acc** | Paper F1 | **Ours F1** | Trainable params (paper → ours) |
|---|---|---|---|---|---|---|
| SST-2 | DistilBERT | 89.02 ± 0.95 | **88.98 ± 0.31** | 89.22 ± 1.18 | **88.96 ± 0.30** | 758K → 758,018 |
| SST-2 | RoBERTa-base | 93.28 ± 0.91 | **94.07 ± 0.63** | 93.51 ± 0.91 | **94.06 ± 0.63** | 1.20M → 1,200,386 |
| Emotion | DistilBERT | 93.22 ± 0.03 | **92.85 ± 0.18** | 90.20 ± 0.03 | **89.66 ± 0.16** | 742.66K → 742,662 |
| E-Commerce (5-class) | mBERT | 67.63 ± 0.26 | **67.26 ± 0.16** | 47.37 ± 1.48 | **49.31 ± 0.87** | 298.76K → 298,757 |
| ABSA | DeBERTa-v3 | 80.74 (single run) | **79.73 ± 0.45** | 79.84 | **78.66 ± 0.78** | 334.08K → 334,083 |

The trainable-parameter counts match the paper for all five models, which confirms the LoRA configuration is the same.

### SST-2, DistilBERT

| Metric | Paper | Ours |
|---|---|---|
| Accuracy (%) | 89.02 ± 0.95 | 88.98 ± 0.31 |
| Precision (%) | 90.53 ± 2.70 | 89.02 ± 0.39 |
| Recall (%) | 88.13 ± 3.96 | 88.95 ± 0.28 |
| F1 (%) | 89.22 ± 1.18 | 88.96 ± 0.30 |
| Similarity (%) | 90.76 ± 0.71 | 90.82 ± 0.09 |
| Confidence (%) | 92.06 ± 2.74 | 93.92 ± 0.78 |
| ECE (%) | 3.86 ± 1.71 | 5.07 ± 0.78 |
| Brier | 0.170 ± 0.010 | 0.168 ± 0.002 |
| Training time (s) | 195 | 159 ± 14 |
| GPU memory (GB) | 0.46 | 0.45 |

![SST-2 DistilBERT confusion matrix](Results/Graphs/confusion_sst2.png)
![SST-2 curves](Results/Graphs/curves_sst2.png)
![SST-2 ours vs paper](Results/Graphs/ours_vs_paper_sst2.png)

### SST-2, RoBERTa-base

| Metric | Paper | Ours |
|---|---|---|
| Accuracy (%) | 93.28 ± 0.91 | 94.07 ± 0.63 |
| Precision (%) | 93.42 ± 1.60 | 94.12 ± 0.65 |
| Recall (%) | 93.64 ± 2.14 | 94.03 ± 0.63 |
| F1 (%) | 93.51 ± 0.91 | 94.06 ± 0.63 |
| Similarity (%) | 94.28 ± 0.61 | 94.76 ± 0.56 |
| Confidence (%) | 95.67 ± 2.68 | 97.99 ± 0.26 |
| ECE (%) | 3.15 ± 1.04 | 3.97 ± 0.29 |
| Brier | 0.110 ± 0.010 | 0.099 ± 0.011 |
| Training time (s) | 392 | 337 ± 42 |
| GPU memory (GB) | 0.92 | 0.90 |

### CARER Emotion, DistilBERT (25k subset)

| Metric | Paper | Ours |
|---|---|---|
| Accuracy (%) | 93.22 ± 0.03 | 92.85 ± 0.18 |
| Precision (%) | 89.85 ± 0.03 | 88.92 ± 0.35 |
| Recall (%) | 90.60 ± 0.03 | 90.56 ± 0.24 |
| F1 (%) | 90.20 ± 0.03 | 89.66 ± 0.16 |
| Similarity (%) | 92.45 ± 0.01 | 94.84 ± 0.23 |
| Confidence (%) | 94.70 ± 0.03 | 94.89 ± 0.04 |
| ECE (%) | 1.73 ± 0.12 | 2.07 ± 0.26 |
| Brier | 0.083 | 0.089 ± 0.004 |
| Training time (s) | 364 | 105 ± 13 |
| GPU memory (GB) | 1.15 | 0.74 |

### Amazon E-Commerce (5-class), multilingual BERT

| Metric | Paper | Ours |
|---|---|---|
| Accuracy (%) | 67.63 ± 0.26 | 67.26 ± 0.16 |
| Precision (%) | 53.88 ± 2.34 | 54.55 ± 1.28 |
| Recall (%) | 46.05 ± 1.07 | 47.76 ± 0.83 |
| F1 (%) | 47.37 ± 1.48 | 49.31 ± 0.87 |
| Similarity (%) | 74.06 ± 0.03 | 73.92 ± 0.31 |
| Confidence (%) | 71.03 ± 0.37 | 69.62 ± 2.13 |
| ECE (%) | 3.67 ± 0.35 | 3.70 ± 1.06 |
| Brier | 0.430 | 0.431 ± 0.004 |
| Training time (s) | 413 | 369 ± 3 |
| GPU memory (GB) | 1.22 | 1.22 |

### SemEval-2014 ABSA, DeBERTa-v3-base

| Metric | Paper (single run) | Ours |
|---|---|---|
| Accuracy (%) | 80.74 | 79.73 ± 0.45 |
| Precision (%) | 79.57 | 79.67 ± 0.49 |
| Recall (%) | 80.74 | 79.73 ± 0.45 |
| F1 (%) | 79.84 | 78.66 ± 0.78 |
| Similarity (%) | 84.38 | 82.51 ± 0.17 |
| Confidence (%) | 83.92 | 84.31 ± 2.96 |
| ECE (%) | 4.34 | 6.18 ± 2.18 |
| Brier | 0.270 | 0.309 ± 0.005 |
| Training time (s) | 2510 | 351 ± 62 |
| GPU memory (GB) | 2.5 | 1.59 |

---

## Key Observations

- **The parameter counts are identical to the paper** on all five models. Only 0.3M to 1.2M parameters are trained, a small fraction of each backbone.
- **Accuracy and F1 are within about 1 to 2 points of the paper** on every dataset. Most differences are inside the paper's own run-to-run variation.
- **RoBERTa on SST-2 scored higher than the paper** (94.07 vs 93.28), and DistilBERT matched it (88.98 vs 89.02).
- **E-Commerce 5-class has high accuracy but low macro-F1.** The class imbalance hurts the minority rating classes, and this shows in both the paper and our run.
- **Our seed-to-seed variance is smaller** than the paper's on SST-2, probably because the paper's runs used different seeds or data order.
- **Training times are not comparable** to the paper's because the hardware differs. Ours were measured on a Kaggle T4.

### Deviations and limitations (stated honestly)

- **Emotion** uses the 25,000-row subset in the authors' repo. The paper reports 75k rows, which explains part of the gap. Our Similarity (94.84) is 2.4 points above the paper's, and I could not explain that.
- **ABSA** results in the paper are a single run. Ours are a 3-seed mean ± std, and our ECE and Brier are somewhat worse.
- **ECE and Brier** follow my implementation (10 equal-width bins on max probability; Brier summed over classes). The paper does not fully specify them.
- **Confidence** is the mean max-softmax probability. This matches the paper's table values, though the paper's text describes it differently.
- Twitter, IMDb, Yelp, E-Commerce 3-class, full fine-tuning and AdaLoRA are **not** reproduced.

---

## Hyperparameters (paper Table 3b)

| Setting | SST-2 | Emotion | E-Commerce 5-class | ABSA |
|---|---|---|---|---|
| Backbone | DistilBERT / RoBERTa-base | DistilBERT | BERT-base-multilingual-uncased | DeBERTa-v3-base |
| LoRA target modules | q_lin, v_lin, out_lin / query, value, output.dense | q_lin, v_lin | query, value | query_proj, key_proj, value_proj |
| Rank r | 6 | 8 | 8 | 6 |
| Alpha | 12 | 16 | 16 | 12 |
| LoRA dropout | 0.2 | 0.1 | 0.05 | 0.1 |
| Learning rate | 3e-4 | see notebook | see notebook | see notebook |
| Batch size (train / eval) | 8 / 16 | 32 / 64 | 8 / 16 | 8 / 16 |
| Epochs (max) | 10 | 5 | 3 | 10 |
| Weight decay | 0.05 | 0.01 | 0.01 | 0.2 |
| Warmup ratio | 0.1 | 0.03 | 0.1 | 0.1 |
| Early-stopping patience | 3 | 1 | 3 | 3 |
| Max sequence length | see notebook | see notebook | 256 | see notebook |
| Seeds | 42, 43, 44 | 42, 43, 44 | 42, 43, 44 | 42, 43, 44 |

Common to all runs: AdamW, cosine LR schedule, fp16, dynamic padding, `modules_to_save=["classifier"]`, 80/10/10 stratified split with `random_state=42`.

---

## How to Run

1. Open https://www.kaggle.com and create a new notebook.
2. Upload `Src/nlp-replication.ipynb` (File → Import Notebook).
3. In **Settings**, set **Accelerator = GPU T4** and **Internet = On**.
4. Run **Cell 1** (installs `sentencepiece`, removes the incompatible `torchao`), then **restart the session**.
5. Run the remaining cells in order. The notebook:
   - clones `ELTE-DSED/senti-matrix` for the datasets,
   - trains LoRA on all four datasets × 3 seeds (about 75 minutes in total),
   - writes tables and figures to `/kaggle/working/outputs/`,
   - zips everything to `/kaggle/working/lora_replication_outputs.zip`.
6. To run a single dataset, edit `DATASETS = ["sst2", "emotion", "ecom5", "absa"]` in Cell 2.

Notes: the notebook sets `CUDA_VISIBLE_DEVICES=0` so that the two-GPU DataParallel does not silently double the batch size. It also casts DeBERTa-v3 to fp32 after loading, because newer `transformers` versions load it in fp16.

---

## Conclusion

The LoRA results of SentiMatrix were reproduced on four datasets and five model/dataset pairs. The trainable-parameter counts match the paper exactly, and accuracy and F1 land within about 1 to 2 points of the published values. LoRA trains well under 1.5M parameters per model and still gives competitive sentiment classification.

The remaining differences come from the smaller Emotion subset, the single-run ABSA numbers in the paper, unspecified calibration-metric details and different hardware.

---

## Reference

Md. Easin Arafat, Muhammad Usman Akmal, Ali S. Abosinnee, Tamás Orosz. *SentiMatrix: Parameter-Efficient Fine-Tuning of Encoder-Based Transformers for Multidimensional Sentiment Analysis.* Machine Learning, 115, 229 (2026). https://doi.org/10.1007/s10994-026-07161-4

Authors' code: https://github.com/ELTE-DSED/senti-matrix

---

## Author

**<Your Name>**
<Your degree, college / university>
GitHub: <your-username>
Email: <your-email>
