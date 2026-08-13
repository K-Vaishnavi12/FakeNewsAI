"""Step 4b — fine-tune ``distilbert-base-uncased`` for fake-news detection.

CPU-friendly by design:

* device auto-detected; batch size drops to 8 on CPU
* ``max_len=256``, 3 epochs, lr 2e-5 (per spec)
* ``--max-train-rows`` caps the dataset so a demo run finishes in minutes
* any failure (no torch, OOM, no data) is caught and reported — the caller
  simply keeps using the baseline through the same ``predict()`` interface.

Run::

    python -m src.models.train_transformer --max-train-rows 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import (
    CPU_BATCH_SIZE,
    GPU_BATCH_SIZE,
    SEED,
    TRANSFORMER_MAX_LEN,
    TRANSFORMER_MODEL,
    TRANSFORMER_DIR,
    ensure_dirs,
    get_logger,
    set_seeds,
)

LOG = get_logger("veritruth.models.train_transformer")


def detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def batch_size_for(device: str) -> int:
    return GPU_BATCH_SIZE if device == "cuda" else CPU_BATCH_SIZE


def compute_metrics(eval_pred) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logits, labels = eval_pred
    preds = np.argmax(np.asarray(logits), axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def train(
    epochs: int = 3,
    max_train_rows: int = 0,
    out_dir: Path = TRANSFORMER_DIR,
) -> dict:
    """Fine-tune DistilBERT. Returns a result dict with ``ok`` and ``reason``."""
    set_seeds(SEED)
    ensure_dirs()

    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        LOG.warning("Transformers/torch unavailable (%s); keeping baseline.", exc)
        return {"ok": False, "reason": f"dependencies unavailable: {exc}"}

    from src.data.split import load_splits
    from src.models.calibrate import Calibration, brier, fit_temperature

    device = detect_device()
    bsz = batch_size_for(device)
    LOG.info("Device=%s batch_size=%s epochs=%s", device, bsz, epochs)

    try:
        splits = load_splits()
    except Exception as exc:
        LOG.warning("No training data (%s); keeping baseline.", exc)
        return {"ok": False, "reason": f"no data: {exc}"}

    train_df, val_df = splits["train"], splits["val"]
    if max_train_rows and len(train_df) > max_train_rows:
        train_df = train_df.sample(n=max_train_rows, random_state=SEED).reset_index(drop=True)
    if len(train_df) < 20 or len(val_df) < 4:
        return {"ok": False, "reason": f"dataset too small ({len(train_df)} train rows)"}

    class TextDataset(Dataset):
        def __init__(self, texts: list[str], labels: list[int], tok) -> None:
            self.enc = tok(
                texts,
                truncation=True,
                padding="max_length",
                max_length=TRANSFORMER_MAX_LEN,
            )
            self.labels = labels

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> dict:
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(int(self.labels[idx]))
            return item

    try:
        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            TRANSFORMER_MODEL,
            num_labels=2,
            id2label={0: "FAKE", 1: "REAL"},
            label2id={"FAKE": 0, "REAL": 1},
        )
    except Exception as exc:
        LOG.warning("Could not download %s (%s); keeping baseline.", TRANSFORMER_MODEL, exc)
        return {"ok": False, "reason": f"model download failed: {exc}"}

    ds_train = TextDataset(
        train_df["text"].astype(str).tolist(),
        train_df["label"].astype(int).tolist(),
        tokenizer,
    )
    ds_val = TextDataset(
        val_df["text"].astype(str).tolist(),
        val_df["label"].astype(int).tolist(),
        tokenizer,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=bsz,
        per_device_eval_batch_size=bsz,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=25,
        save_strategy="no",
        report_to=[],
        seed=SEED,
        data_seed=SEED,
        fp16=(device == "cuda"),
        use_cpu=(device == "cpu"),
        disable_tqdm=False,
    )

    try:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        metrics = trainer.evaluate()
    except Exception as exc:
        LOG.warning("Training failed (%s); keeping baseline.", exc)
        return {"ok": False, "reason": f"training failed: {exc}"}

    try:
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
    except Exception as exc:
        LOG.warning("Could not save model (%s).", exc)
        return {"ok": False, "reason": f"save failed: {exc}"}

    # ---- temperature-scale the validation margins -> calibrated trust score
    try:
        raw = trainer.predict(ds_val)
        logits = np.asarray(raw.predictions, dtype=float)
        margin = logits[:, 1] - logits[:, 0]
        y_val = val_df["label"].to_numpy(dtype=int)
        temp = fit_temperature(margin, y_val)
        calib = Calibration(
            method="temperature",
            temperature=temp,
            source_model="distilbert",
            val_brier_before=brier(y_val, 1.0 / (1.0 + np.exp(-margin))),
        )
        calib.val_brier_after = brier(y_val, calib.apply_logit(margin))
        calib.save()
    except Exception as exc:
        LOG.warning("Calibration after training failed (%s); baseline calibration kept.", exc)

    (out_dir / "train_metrics.json").write_text(
        json.dumps({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}, indent=2),
        encoding="utf-8",
    )
    return {"ok": True, "reason": "trained", "metrics": metrics, "device": device, "path": str(out_dir)}


def main() -> dict:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for VeriTruth")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Cap training rows for a fast demo run (0 = use all).",
    )
    ns = parser.parse_args()

    result = train(epochs=ns.epochs, max_train_rows=ns.max_train_rows)

    print("\n--- STEP 4b VERIFICATION (transformer) ------------------------")
    if result["ok"]:
        print(f"Status            : trained on {result['device']}")
        for key, value in result["metrics"].items():
            if isinstance(value, (int, float)):
                print(f"  {key:<24} {value:.4f}")
        print(f"Saved to          : {result['path']}")
        print("Expected          : eval_accuracy printed, model dir contains config.json")
    else:
        print(f"Status            : FELL BACK TO BASELINE ({result['reason']})")
        print("Expected          : this is a valid outcome; predict() still works")
    print("---------------------------------------------------------------\n")
    return result


if __name__ == "__main__":
    main()
