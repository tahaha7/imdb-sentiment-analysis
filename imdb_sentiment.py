import argparse
import logging
import random
import re
import sys
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "aclImdb"
ARCHIVE_PATH = DATA_DIR / "aclImdb_v1.tar.gz"
CSV_PATH = DATA_DIR / "imdb_dataset.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

HTML_TAG_RE = re.compile(r"<[^>]+>")
TOKEN_RE = re.compile(r"[a-z0-9']+")

LOGGER = logging.getLogger("imdb_sentiment")

def configure_logging():
    LOGGER.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(handler)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def strip_html(text):
    return HTML_TAG_RE.sub(" ", text)

def tokenize(text):
    return TOKEN_RE.findall(text.lower())

def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DIR.exists():
        return
    if not ARCHIVE_PATH.exists():
        LOGGER.info("downloading IMDB dataset from %s", DATASET_URL)
        urllib.request.urlretrieve(DATASET_URL, ARCHIVE_PATH)
    LOGGER.info("extracting dataset archive")
    with tarfile.open(ARCHIVE_PATH) as archive:
        archive.extractall(DATA_DIR, filter="data")
    ARCHIVE_PATH.unlink()

def build_dataset_csv():
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH, keep_default_na=False, na_filter=False)
    records = []
    for split in ("train", "test"):
        for sentiment in ("pos", "neg"):
            folder = RAW_DIR / split / sentiment
            for file_path in sorted(folder.glob("*.txt")):
                text = strip_html(file_path.read_text(encoding="utf-8"))
                records.append({"review": text, "sentiment": sentiment, "split": split})
    df = pd.DataFrame.from_records(records)
    df.to_csv(CSV_PATH, index=False)
    return df

def prepare_data(args):
    download_dataset()
    df = build_dataset_csv()
    df["label"] = (df["sentiment"] == "pos").astype(int)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    if args.max_train_samples:
        train_df = train_df.sample(n=min(args.max_train_samples, len(train_df)), random_state=args.seed)
        train_df = train_df.reset_index(drop=True)
    if args.max_test_samples:
        test_df = test_df.sample(n=min(args.max_test_samples, len(test_df)), random_state=args.seed)
        test_df = test_df.reset_index(drop=True)
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=args.seed, stratify=train_df["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

class Vocabulary:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, tokenized_texts, max_size=30000, min_freq=2):
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)
        itos = [self.PAD, self.UNK]
        for token, freq in counter.most_common():
            if freq < min_freq:
                continue
            if len(itos) >= max_size:
                break
            itos.append(token)
        self.itos = itos
        self.stoi = {token: idx for idx, token in enumerate(itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, max_len):
        ids = [self.stoi.get(token, 1) for token in tokens[:max_len]]
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        return ids

class LSTMReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        tokens = tokenize(self.texts[index])
        length = min(len(tokens), self.max_len) or 1
        ids = self.vocab.encode(tokens, self.max_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(self.labels[index], dtype=torch.float),
        }

class BertReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
        }

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        combined = torch.cat([hidden[-2], hidden[-1]], dim=1)
        combined = self.dropout(combined)
        return self.classifier(combined).squeeze(1)

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

def save_confusion_matrix(y_true, y_pred, title, output_path):
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["negative", "positive"])
    ax.set_yticklabels(["negative", "positive"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i, j], ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def run_lstm_eval(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"]
            labels = batch["label"]
            logits = model(input_ids, lengths)
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.long().tolist())
    return compute_metrics(all_labels, all_preds), all_labels, all_preds

def train_lstm(train_df, val_df, test_df, args, device):
    train_tokens = [tokenize(text) for text in train_df["review"]]
    vocab = Vocabulary(train_tokens, max_size=args.vocab_size, min_freq=args.min_freq)
    LOGGER.info("lstm vocabulary size: %d", len(vocab))

    train_dataset = LSTMReviewDataset(train_df["review"].tolist(), train_df["label"].tolist(), vocab, args.max_len)
    val_dataset = LSTMReviewDataset(val_df["review"].tolist(), val_df["label"].tolist(), vocab, args.max_len)
    test_dataset = LSTMReviewDataset(test_df["review"].tolist(), test_df["label"].tolist(), vocab, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.lstm_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.lstm_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.lstm_batch_size)

    model = LSTMClassifier(len(vocab), args.embed_dim, args.hidden_dim, args.lstm_layers, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lstm_lr)
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_path = ARTIFACTS_DIR / "lstm_model.pt"
    best_val_f1 = 0.0
    start_time = time.time()
    for epoch in range(1, args.lstm_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"lstm epoch {epoch}/{args.lstm_epochs}"):
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"]
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * input_ids.size(0)
        train_loss = running_loss / len(train_dataset)
        val_metrics, _, _ = run_lstm_eval(model, val_loader, device)
        LOGGER.info(
            "lstm epoch %d/%d train_loss=%.4f val_acc=%.4f val_f1=%.4f",
            epoch, args.lstm_epochs, train_loss, val_metrics["accuracy"], val_metrics["f1"],
        )
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save({"model_state": model.state_dict(), "vocab_itos": vocab.itos}, checkpoint_path)
    train_time = time.time() - start_time

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, y_true, y_pred = run_lstm_eval(model, test_loader, device)
    save_confusion_matrix(y_true, y_pred, "LSTM Confusion Matrix", ARTIFACTS_DIR / "lstm_confusion_matrix.png")
    test_metrics["train_time_sec"] = round(train_time, 2)
    test_metrics["model"] = "LSTM"
    LOGGER.info("lstm test metrics: %s", test_metrics)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test_metrics

def run_bert_eval(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    return compute_metrics(all_labels, all_preds), all_labels, all_preds

def train_bert(train_df, val_df, test_df, args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.bert_model_name, num_labels=2).to(device)

    train_dataset = BertReviewDataset(train_df["review"].tolist(), train_df["label"].tolist(), tokenizer, args.bert_max_len)
    val_dataset = BertReviewDataset(val_df["review"].tolist(), val_df["label"].tolist(), tokenizer, args.bert_max_len)
    test_dataset = BertReviewDataset(test_df["review"].tolist(), test_df["label"].tolist(), tokenizer, args.bert_max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.bert_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bert_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.bert_batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.bert_lr)
    total_steps = max(1, (len(train_loader) // args.grad_accum_steps) * args.bert_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    checkpoint_dir = ARTIFACTS_DIR / "bert_model"
    best_val_f1 = 0.0
    start_time = time.time()
    for epoch in range(1, args.bert_epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc=f"bert epoch {epoch}/{args.bert_epochs}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / args.grad_accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            running_loss += loss.item() * args.grad_accum_steps * input_ids.size(0)
        train_loss = running_loss / len(train_dataset)
        val_metrics, _, _ = run_bert_eval(model, val_loader, device)
        LOGGER.info(
            "bert epoch %d/%d train_loss=%.4f val_acc=%.4f val_f1=%.4f",
            epoch, args.bert_epochs, train_loss, val_metrics["accuracy"], val_metrics["f1"],
        )
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
    train_time = time.time() - start_time

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
    test_metrics, y_true, y_pred = run_bert_eval(model, test_loader, device)
    save_confusion_matrix(y_true, y_pred, "BERT Confusion Matrix", ARTIFACTS_DIR / "bert_confusion_matrix.png")
    test_metrics["train_time_sec"] = round(train_time, 2)
    test_metrics["model"] = "BERT"
    LOGGER.info("bert test metrics: %s", test_metrics)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test_metrics

def build_arg_parser():
    parser = argparse.ArgumentParser(description="IMDB review sentiment classification with LSTM and BERT")
    parser.add_argument("--model", choices=["lstm", "bert", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lstm-lr", type=float, default=1e-3)
    parser.add_argument("--lstm-batch-size", type=int, default=64)
    parser.add_argument("--lstm-epochs", type=int, default=5)
    parser.add_argument("--bert-model-name", default="bert-base-uncased")
    parser.add_argument("--bert-max-len", type=int, default=256)
    parser.add_argument("--bert-batch-size", type=int, default=4)
    parser.add_argument("--bert-lr", type=float, default=2e-5)
    parser.add_argument("--bert-epochs", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    return parser

def main():
    configure_logging()
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    LOGGER.info("using device: %s", device)

    train_df, val_df, test_df = prepare_data(args)
    LOGGER.info("train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    results = []
    if args.model in ("lstm", "both"):
        results.append(train_lstm(train_df, val_df, test_df, args, device))
    if args.model in ("bert", "both"):
        results.append(train_bert(train_df, val_df, test_df, args, device))

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(ARTIFACTS_DIR / "model_comparison.csv", index=False)
        LOGGER.info("\n%s", results_df.to_string(index=False))

if __name__ == "__main__":
    main()
