# 作業一：BERT 中文多元情緒分類

使用老師指定的 `google-bert/bert-base-chinese`，將 Chinese Multi-Emotion Dialogue Dataset 的文字情緒標籤轉為整數後，微調八分類模型。

所有數字與輸出均為 2026-10-25 在本機實際執行所得。

## 資料與模型

- 資料集：<https://huggingface.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset>
- 基礎模型：<https://huggingface.co/google-bert/bert-base-chinese>
- 80% 分層訓練集、20% 分層驗證集，random seed 42。
- 短句資料使用最大長度 128；RTX 2060 6 GB 使用 micro-batch 8、gradient accumulation 4。

## Windows 執行

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python download_data.py
.\.venv\Scripts\python prepare_data.py
.\.venv\Scripts\python train.py
.\.venv\Scripts\python inference.py
```

## 實際成果

資料共 4,159 筆，分層切分為 3,327 筆訓練與 832 筆驗證。訓練 3 epochs 後：

- validation loss：0.428993
- accuracy：0.888221
- macro F1：0.885430

老師指定五句推論結果（信心分數四位小數）：

| 文字 | 預測 | 信心分數 |
|---|---:|---:|
| 我每天都能跟她一起上學，我好開心！ | 開心語調 | 0.9922 |
| 最好的朋友要離開臺灣了，以後可能不容易再見面... | 悲傷語調 | 0.9896 |
| 我覺得我快不行了... | 悲傷語調 | 0.9864 |
| 剛剛收到研究所錄取的通知書！ | 開心語調 | 0.9811 |
| 今年的冬天好像比較晚來。 | 平淡語氣 | 0.9838 |

最佳模型、評估 JSON 與 tokenizer 存於 `output/best_model/`，原始推論結果存於 `results/required_predictions.json`。大型模型權重使用 Git LFS。

## 驗證環境

Windows 11、AMD Ryzen 7 3700X（8C/16T）、15.93 GB RAM、NVIDIA RTX 2060 6 GB、驅動 572.60（CUDA 12.8）、Python 3.12.10、Git 2.55。PyTorch 實際辨識 CUDA，訓練與推論均在本機 GPU 完成。

## 主要套件

版本固定於 `requirements.txt`，包含 PyTorch、Transformers、Datasets、Accelerate、scikit-learn 與 pandas。
