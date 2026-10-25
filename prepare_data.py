import json
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

from config import LABELS_FILE, RAW_FILE, SEED, TRAIN_FILE, VALID_FILE


def prepare() -> dict[str, int]:
    if not RAW_FILE.exists():
        raise FileNotFoundError("請先執行 python download_data.py")
    frame = pd.read_csv(RAW_FILE)
    required = {"text", "emotion"}
    if not required.issubset(frame.columns):
        raise ValueError(f"CSV 缺少欄位：{required - set(frame.columns)}")
    frame = frame[["text", "emotion"]].dropna().copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["emotion"] = frame["emotion"].astype(str).str.strip()
    frame = frame[(frame["text"] != "") & (frame["emotion"] != "")]

    # 使用資料首次出現順序建立穩定映射，並把文字標籤轉成整數。
    labels = list(dict.fromkeys(frame["emotion"].tolist()))
    label2id = {label: index for index, label in enumerate(labels)}
    frame["label"] = frame["emotion"].map(label2id)

    train, valid = train_test_split(
        frame, test_size=0.2, random_state=SEED, stratify=frame["label"]
    )
    TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    train[["text", "label"]].to_json(
        TRAIN_FILE, orient="records", lines=True, force_ascii=False
    )
    valid[["text", "label"]].to_json(
        VALID_FILE, orient="records", lines=True, force_ascii=False
    )
    LABELS_FILE.write_text(
        json.dumps({"label2id": label2id, "id2label": labels}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Total={len(frame)}, train={len(train)}, valid={len(valid)}")
    print("Labels:", dict(Counter(frame["emotion"])))
    print("Mapping:", label2id)
    return label2id


if __name__ == "__main__":
    prepare()

