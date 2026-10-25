from pathlib import Path

MODEL_NAME = "google-bert/bert-base-chinese"
DATASET_URL = (
    "https://huggingface.co/datasets/Johnson8187/"
    "Chinese_Multi-Emotion_Dialogue_Dataset/resolve/main/data.csv"
)
ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "data" / "raw" / "data.csv"
TRAIN_FILE = ROOT / "data" / "processed" / "train.jsonl"
VALID_FILE = ROOT / "data" / "processed" / "valid.jsonl"
LABELS_FILE = ROOT / "data" / "processed" / "labels.json"
OUTPUT_DIR = ROOT / "output" / "best_model"
RESULTS_DIR = ROOT / "results"
SEED = 42

