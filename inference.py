import json

import torch
from transformers import pipeline

from config import OUTPUT_DIR, RESULTS_DIR

HOMEWORK_TEXTS = [
    "我每天都能跟她一起上學，我好開心！",
    "最好的朋友要離開臺灣了，以後可能不容易再見面...",
    "我覺得我快不行了...",
    "剛剛收到研究所錄取的通知書！",
    "今年的冬天好像比較晚來。",
]


def predict(texts=HOMEWORK_TEXTS):
    if not OUTPUT_DIR.exists():
        raise FileNotFoundError("找不到訓練模型，請先執行 python train.py")
    classifier = pipeline(
        "text-classification",
        model=str(OUTPUT_DIR),
        tokenizer=str(OUTPUT_DIR),
        device=0 if torch.cuda.is_available() else -1,
    )
    raw = classifier(texts, truncation=True, max_length=128)
    results = []
    for text, item in zip(texts, raw):
        row = {"text": text, "label": item["label"], "score": round(item["score"], 4)}
        results.append(row)
        print(f'{text} => {row["label"]} ({row["score"]:.4f})')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "required_predictions.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


if __name__ == "__main__":
    predict()

