# 🎬 IMDB Sentiment Analysis — LSTM vs BERT

Binary sentiment classification on the [IMDB movie reviews dataset](https://ai.stanford.edu/~amaas/data/sentiment/) (50k reviews), comparing a from-scratch **bidirectional LSTM** against a fine-tuned **BERT** model.

## What it does
1. **Data pipeline** — downloads the aclImdb archive, strips HTML, builds a clean CSV (`review`, `sentiment`, `split`)
2. **LSTM model** — custom vocabulary (30k tokens), 128-dim embeddings, 2-layer bidirectional LSTM, trained with BCE loss and packed sequences
3. **BERT model** — fine-tunes `bert-base-uncased` with AdamW, linear warmup scheduling, gradient accumulation, and mixed precision
4. **Evaluation** — accuracy, precision, recall, F1 on the held-out test set, confusion-matrix plots, and a `model_comparison.csv` summarizing both models

## Tech
Python · PyTorch · Hugging Face Transformers · scikit-learn · pandas · matplotlib

## Run it
```bash
pip install -r requirements.txt
python imdb_sentiment.py --model both
```

Options:
- `--model lstm|bert|both` — which model(s) to train (default: `both`)
- `--max-train-samples N` / `--max-test-samples N` — subsample for quick experiments
- `--lstm-epochs`, `--bert-epochs`, `--bert-batch-size`, etc. — full hyperparameter control via CLI (see `--help`)

Artifacts (checkpoints, confusion matrices, comparison CSV) are written to `artifacts/`.

## Notes
- The dataset (~84 MB) downloads automatically on first run into `data/`.
- BERT fine-tuning benefits strongly from a GPU; the script auto-detects CUDA.
