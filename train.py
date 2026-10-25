import json
import os

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import LABELS_FILE, MODEL_NAME, OUTPUT_DIR, SEED, TRAIN_FILE, VALID_FILE


def metrics(prediction):
    logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
    predicted = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(prediction.label_ids, predicted),
        "macro_f1": f1_score(prediction.label_ids, predicted, average="macro", zero_division=0),
    }


def main() -> None:
    if not TRAIN_FILE.exists() or not VALID_FILE.exists():
        raise FileNotFoundError("請先執行 download_data.py 與 prepare_data.py")
    mapping = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    label2id = mapping["label2id"]
    id2label = {index: label for index, label in enumerate(mapping["id2label"])}
    set_seed(SEED)
    dataset = load_dataset(
        "json", data_files={"train": str(TRAIN_FILE), "validation": str(VALID_FILE)}
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )
    cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=float(os.getenv("NLP_EPOCHS", "3")),
        per_device_train_batch_size=int(os.getenv("NLP_TRAIN_BATCH", "8")),
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=int(os.getenv("NLP_GRAD_ACCUM", "4")),
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=cuda,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    print(f"CUDA={cuda}; GPU={torch.cuda.get_device_name(0) if cuda else 'CPU'}")
    trainer.train()
    evaluation = trainer.evaluate()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    (OUTPUT_DIR / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

