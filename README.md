# Assignment 2: Responsible & Explainable AI
## Auditing Content Moderation AI for Bias, Adversarial Robustness & Safety
### FAST-NUCES

---

## Environment

- **Python version**: 3.10
- **GPU used**: NVIDIA T4 (Google Colab free tier) — ~25–35 min per DistilBERT training run
- **Framework**: PyTorch 2.1 + HuggingFace Transformers 4.35

---

## Repository Structure

```
.
├── part1.ipynb          # Baseline DistilBERT classifier
├── part2.ipynb          # Bias audit: cohort analysis with AIF360
├── part3.ipynb          # Adversarial attacks: evasion + poisoning
├── part4.ipynb          # Bias mitigation: reweighing, threshold opt, oversampling
├── part5.ipynb          # Guardrail pipeline demonstration
├── pipeline.py          # ModerationPipeline class (three-layer pipeline)
├── requirements.txt     # Pinned dependencies
└── README.md            # This file
```

---

## Dataset Setup

1. Create a free Kaggle account at https://kaggle.com
2. Go to: https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data
3. Accept the competition rules and download only these two files:
   - `jigsaw-unintended-bias-train.csv`
   - `validation.csv` (optional)
4. Place the CSV files in the **same directory** as the notebooks.

> **Note**: Do NOT commit these CSV files. They are in `.gitignore`.

---

## Reproduction Steps

### Option A: Google Colab (Recommended)

1. Upload all notebooks and `pipeline.py` to a Colab session
2. Upload the dataset CSV to the session storage
3. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
4. Run each notebook in order: `part1` → `part2` → `part3` → `part4` → `part5`
5. Each notebook saves intermediate files (CSVs, `.npy` probability arrays, model checkpoints) that downstream notebooks depend on

### Option B: Local environment with CUDA GPU

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run notebooks
jupyter notebook
```

---

## Inter-notebook Dependencies

The notebooks must be run in order because each saves artifacts used by later notebooks:

| Notebook | Saves | Used by |
|----------|-------|---------|
| part1.ipynb | `train_subset.csv`, `eval_subset.csv`, `eval_probs_part1.npy`, `eval_labels.npy`, `./model_checkpoint_part1/` | All later parts |
| part2.ipynb | Analysis outputs only | Part 4 (conceptually) |
| part3.ipynb | Poisoned model (optional checkpoint) | Standalone |
| part4.ipynb | `eval_probs_reweigh.npy`, `eval_probs_oversample.npy`, `./model_checkpoint_oversample/` | Part 5 |
| part5.ipynb | Demonstration outputs | Standalone |

---

## Key Design Decisions

### Threshold (Part 1)
Operating threshold set at **0.5** as a neutral baseline. Justified via threshold sweep across 0.3–0.7.

### Cohort Construction (Part 2)
- High-black cohort: `black >= 0.5`
- Reference cohort: `black < 0.1 AND white >= 0.5`
- Follows methodology from Sachdeva et al. (2022) and the original Jigsaw paper

### Mitigation (Part 4)
Best mitigated model: **Oversampling** — provides the best FPR reduction for the high-black cohort with minimal F1 degradation, and does not require post-hoc threshold adjustment that can mask underlying model bias.

### Pipeline Thresholds (Part 5)
- Block: confidence >= 0.6
- Allow: confidence <= 0.4
- Review queue: 0.4–0.6 (human review)

---

## .gitignore

```
*.csv
*.pt
*.bin
saved_model/
model_checkpoint*/
results_*/
__pycache__/
*.npy
logs/
```

---

## References

- Sachdeva, R. et al. (2022). *Measuring and Mitigating Unintended Bias in Text Classification*
- Chouldechova, A. (2017). *Fair Prediction with Disparate Impact*
- Dwork, C. et al. (2012). *Fairness Through Awareness*
- Jigsaw/Conversation AI. (2019). *Unintended Bias in Toxicity Classification* (Kaggle)
- Davidson, T. et al. (2019). *Racial Bias in Hate Speech and Abusive Language Detection Datasets*
