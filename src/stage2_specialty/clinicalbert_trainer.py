import torch
import numpy as np
import pandas as pd
import mlflow
import json
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import (
    f1_score, accuracy_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from src.stage1_doctype.label_consolidation import get_class_weights
from datetime import datetime


# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LENGTH   = 512
CHUNK_STRIDE = 256      # overlap between chunks
MAX_CHUNKS   = 4        # max chunks per document (4 * 512 = 2048 tokens max)
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────
class ClinicalNotesDataset(Dataset):
    """
    Chunks long documents into overlapping 512-token windows.
    Each sample is stored as a list of chunks.
    The model pools across chunks to get one document representation.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int = MAX_LENGTH,
        stride: int = CHUNK_STRIDE,
        max_chunks: int = MAX_CHUNKS,
    ):
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length  = max_length
        self.stride      = stride
        self.max_chunks  = max_chunks
        self.samples     = []

        over_limit = 0
        for text, label in zip(texts, labels):
            chunks = self._chunk_text(text)
            if len(chunks["input_ids"]) > max_chunks:
                over_limit += 1
            self.samples.append({
                "input_ids":      chunks["input_ids"][:max_chunks],
                "attention_mask": chunks["attention_mask"][:max_chunks],
                "label":          label,
            })

        logger.info(
            f"Dataset built: {len(self.samples)} samples. "
            f"{over_limit} exceeded {max_chunks} chunks and were truncated."
        )

    def _chunk_text(self, text: str) -> dict:
        """Tokenize and split into overlapping chunks."""
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"][0]
        attention  = encoding["attention_mask"][0]

        chunks_ids  = []
        chunks_mask = []
        step = self.max_length - self.stride  # non-overlapping step size

        for start in range(0, max(1, len(input_ids) - 1), step):
            end = start + self.max_length - 2  # reserve 2 tokens for [CLS] [SEP]

            chunk_ids = input_ids[start:end]
            chunk_msk = attention[start:end]

            # Add [CLS] and [SEP]
            cls = torch.tensor([self.tokenizer.cls_token_id])
            sep = torch.tensor([self.tokenizer.sep_token_id])
            chunk_ids = torch.cat([cls, chunk_ids, sep])
            chunk_msk = torch.cat([torch.ones(1, dtype=torch.long),
                                   chunk_msk,
                                   torch.ones(1, dtype=torch.long)])

            # Pad to max_length
            pad_len = self.max_length - len(chunk_ids)
            if pad_len > 0:
                chunk_ids = torch.cat([chunk_ids,
                    torch.full((pad_len,), self.tokenizer.pad_token_id)])
                chunk_msk = torch.cat([chunk_msk,
                    torch.zeros(pad_len, dtype=torch.long)])

            chunks_ids.append(chunk_ids[:self.max_length])
            chunks_mask.append(chunk_msk[:self.max_length])

            if end >= len(input_ids):
                break

        return {
            "input_ids":      chunks_ids,
            "attention_mask": chunks_mask,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """
    Pad batches so all samples have the same number of chunks.
    Shape after collation: (batch_size, n_chunks, seq_len)
    """
    max_chunks = max(len(s["input_ids"]) for s in batch)
    seq_len    = batch[0]["input_ids"][0].shape[0]

    padded_ids  = []
    padded_mask = []
    labels      = []

    for sample in batch:
        n = len(sample["input_ids"])
        pad_needed = max_chunks - n

        ids  = torch.stack(sample["input_ids"])
        mask = torch.stack(sample["attention_mask"])

        if pad_needed > 0:
            pad_ids  = torch.zeros(pad_needed, seq_len, dtype=torch.long)
            pad_mask = torch.zeros(pad_needed, seq_len, dtype=torch.long)
            ids  = torch.cat([ids,  pad_ids],  dim=0)
            mask = torch.cat([mask, pad_mask], dim=0)

        padded_ids.append(ids)
        padded_mask.append(mask)
        labels.append(sample["label"])

    return {
        "input_ids":      torch.stack(padded_ids),
        "attention_mask": torch.stack(padded_mask),
        "labels":         torch.tensor(labels, dtype=torch.long),
    }


# ── Model ─────────────────────────────────────────────────────────────────────
class ChunkPoolClinicalBERT(nn.Module):
    """
    ClinicalBERT with mean-pool across chunks for long-document classification.

    Forward pass:
      1. Reshape (batch, chunks, seq) → (batch*chunks, seq)
      2. Run all chunks through BERT in one forward pass
      3. Extract [CLS] token embedding per chunk
      4. Mean-pool across chunks → one vector per document
      5. Dropout → linear classification head
    """

    def __init__(self, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.bert      = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size    = self.bert.config.hidden_size  # 768 for BERT-base
        self.dropout   = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        """
        input_ids:      (batch, n_chunks, seq_len)
        attention_mask: (batch, n_chunks, seq_len)
        """
        batch, n_chunks, seq_len = input_ids.shape

        # Flatten chunks into batch dimension
        ids_flat  = input_ids.view(batch * n_chunks, seq_len)
        mask_flat = attention_mask.view(batch * n_chunks, seq_len)

        # BERT forward — only pass non-padding chunks
        # (chunks with all-zero attention mask still run but contribute 0 after pool)
        outputs = self.bert(input_ids=ids_flat, attention_mask=mask_flat)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Reshape back to (batch, n_chunks, hidden)
        cls_embeddings = cls_embeddings.view(batch, n_chunks, -1)

        # Mean pool across chunks (ignoring all-padding chunks)
        # Build chunk mask: a chunk is valid if its attention mask has any 1s
        chunk_valid = attention_mask.sum(dim=-1) > 0  # (batch, n_chunks)
        chunk_valid = chunk_valid.unsqueeze(-1).float()  # (batch, n_chunks, 1)

        pooled = (cls_embeddings * chunk_valid).sum(dim=1)
        pooled = pooled / chunk_valid.sum(dim=1).clamp(min=1)  # (batch, hidden)

        return self.classifier(self.dropout(pooled))


# ── Training utilities ────────────────────────────────────────────────────────
def build_label_encoder(labels: pd.Series) -> tuple[dict, dict]:
    """Returns (label2id, id2label) dicts."""
    unique = sorted(labels.unique())
    label2id = {l: i for i, l in enumerate(unique)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def evaluate(model, loader, device, id2label) -> dict:
    """Run evaluation loop, return metrics dict."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)

            logits = model(ids, mask)
            preds  = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())

    label_names = [id2label[i] for i in sorted(id2label)]
    macro_f1    = f1_score(all_labels, all_preds, average="macro")
    accuracy    = accuracy_score(all_labels, all_preds)
    report      = classification_report(
        all_labels, all_preds,
        target_names=label_names,
        output_dict=True,
    )
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "report":   report,
        "cm":       cm,
        "preds":    all_preds,
        "labels":   all_labels,
    }


def plot_confusion_matrix(cm, labels, path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ── Main training function ────────────────────────────────────────────────────
def train_clinicalbert(
    stage: int = 2,
    epochs: int = 4,
    batch_size: int = 8,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    dropout: float = 0.1,
):
    logger.info(f"Training Stage {stage} ClinicalBERT on {DEVICE}")
    logger.info(f"Model: {MODEL_NAME}")

    # ── Load data ──────────────────────────────────────────────────────────
    prefix    = f"data/processed/stage{stage}"
    label_col = "stage1_label" if stage == 1 else "stage2_label"

    train_df = pd.read_csv(f"{prefix}_train.csv", index_col=0).dropna(
        subset=[label_col, "transcription"])
    val_df   = pd.read_csv(f"{prefix}_val.csv",   index_col=0).dropna(
        subset=[label_col, "transcription"])
    test_df  = pd.read_csv(f"{prefix}_test.csv",  index_col=0).dropna(
        subset=[label_col, "transcription"])

    label2id, id2label = build_label_encoder(train_df[label_col])
    n_classes = len(label2id)
    logger.info(f"Classes ({n_classes}): {list(label2id.keys())}")

    # ── Tokenizer & datasets ───────────────────────────────────────────────
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Token limit warning
    lengths = train_df["transcription"].apply(
        lambda t: len(tokenizer.encode(t, add_special_tokens=False))
    )
    over_512 = (lengths > 512).sum()
    logger.info(
        f"Token length check: {over_512}/{len(train_df)} "
        f"({over_512/len(train_df)*100:.1f}%) exceed 512 tokens → "
        f"will be chunked into up to {MAX_CHUNKS} chunks"
    )

    def make_dataset(df):
        texts  = df["transcription"].tolist()
        labels = [label2id[l] for l in df[label_col]]
        return ClinicalNotesDataset(texts, labels, tokenizer)

    logger.info("Building datasets...")
    train_ds = make_dataset(train_df)
    val_ds   = make_dataset(val_df)
    test_ds  = make_dataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)

    # ── Model ──────────────────────────────────────────────────────────────
    logger.info("Loading ClinicalBERT...")
    model = ChunkPoolClinicalBERT(n_classes=n_classes, dropout=dropout).to(DEVICE)

    # ── Loss with class weights ────────────────────────────────────────────
    weights_dict = get_class_weights(train_df[label_col])
    weight_tensor = torch.tensor(
        [weights_dict[id2label[i]] for i in range(n_classes)],
        dtype=torch.float
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # ── Optimizer & scheduler ──────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps   = len(train_loader) * epochs
    warmup_steps  = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow.set_experiment(f"stage{stage}_clinicalbert")
    Path("outputs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    

    with mlflow.start_run(run_name=f"clinicalbert_stage{stage}_ep{epochs}_{timestamp}"):
        mlflow.log_params({
            "model":        MODEL_NAME,
            "stage":        stage,
            "epochs":       epochs,
            "batch_size":   batch_size,
            "lr":           lr,
            "max_length":   MAX_LENGTH,
            "chunk_stride": CHUNK_STRIDE,
            "max_chunks":   MAX_CHUNKS,
            "dropout":      dropout,
            "device":       str(DEVICE),
        })

        best_val_f1   = 0.0
        best_model_path = f"models/stage{stage}_clinicalbert_{timestamp}.pt"

        # ── Training loop ─────────────────────────────────────────────
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0

            for step, batch in enumerate(train_loader):
                ids  = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                lbls = batch["labels"].to(DEVICE)

                optimizer.zero_grad()
                logits = model(ids, mask)
                loss   = criterion(logits, lbls)
                loss.backward()

                # Gradient clipping — stabilizes BERT fine-tuning
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

                if (step + 1) % 50 == 0:
                    avg = total_loss / (step + 1)
                    logger.info(
                        f"Epoch {epoch}/{epochs} | "
                        f"Step {step+1}/{len(train_loader)} | "
                        f"Loss: {avg:.4f}"
                    )

            avg_loss = total_loss / len(train_loader)

            # ── Validation ────────────────────────────────────────────
            val_metrics = evaluate(model, val_loader, DEVICE, id2label)
            val_f1      = val_metrics["macro_f1"]
            val_acc     = val_metrics["accuracy"]

            logger.info(
                f"Epoch {epoch} complete | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Val Macro F1: {val_f1:.4f} | "
                f"Val Accuracy: {val_acc:.4f}"
            )

            mlflow.log_metrics({
                f"train_loss":  avg_loss,
                f"val_macro_f1": val_f1,
                f"val_accuracy": val_acc,
            }, step=epoch)

            # Save best model checkpoint
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    "epoch":      epoch,
                    "model_state_dict": model.state_dict(),
                    "val_macro_f1":     val_f1,
                    "label2id":   label2id,
                    "id2label":   id2label,
                }, best_model_path)
                logger.info(f"  ✓ New best model saved (Val F1: {val_f1:.4f})")

        # ── Final evaluation on test set ──────────────────────────────
        logger.info("Loading best checkpoint for test evaluation...")
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics = evaluate(model, test_loader, DEVICE, id2label)
        test_f1      = test_metrics["macro_f1"]
        test_acc     = test_metrics["accuracy"]

        label_names = [id2label[i] for i in sorted(id2label)]

        print(f"\n=== FINAL TEST RESULTS — Stage {stage} ClinicalBERT ===")
        print(f"  Macro F1:  {test_f1:.4f}  (baseline: {'0.4940' if stage==1 else '0.6051'})")
        print(f"  Accuracy:  {test_acc:.4f}")
        print(f"\n{classification_report(test_metrics['labels'], test_metrics['preds'], target_names=label_names)}")

        mlflow.log_metrics({
            "test_macro_f1": test_f1,
            "test_accuracy": test_acc,
            "best_val_macro_f1": best_val_f1,
        })

        # Log per-class test metrics
        for label in label_names:
            if label in test_metrics["report"]:
                r = test_metrics["report"][label]
                mlflow.log_metrics({
                    f"test_f1_{label}":        r["f1-score"],
                    f"test_precision_{label}": r["precision"],
                    f"test_recall_{label}":    r["recall"],
                })

        # Confusion matrix
        cm_path = f"outputs/stage{stage}_clinicalbert_confusion_matrix.png"
        plot_confusion_matrix(
            test_metrics["cm"], label_names, cm_path,
            title=f"Stage {stage} ClinicalBERT — Test Set"
        )
        mlflow.log_artifact(cm_path)

        # Save full report
        report_path = f"outputs/stage{stage}_clinicalbert_report.json"
        with open(report_path, "w") as f:
            json.dump(test_metrics["report"], f, indent=2)
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(best_model_path)

        logger.info(f"Training complete. Best val F1: {best_val_f1:.4f}")
        logger.info(f"Test F1: {test_f1:.4f}")

    return {
        "best_val_f1": best_val_f1,
        "test_macro_f1": test_f1,
        "test_accuracy": test_acc,
    }