# Fake-News-Detection

Fake News Detection，支持三种模型对比实验：TF-IDF / LSTM-style / BERT-style。

## 1. 项目结构

```text
fake-news-detection/
├── data/
│   ├── raw/
│   │   ├── Fake.csv
│   │   └── True.csv
│   └── processed/
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── features.py
│   ├── models/
│   │   ├── tfidf_model.py
│   │   ├── lstm_model.py
│   │   └── bert_model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── demo/
│   └── streamlit_app.py
├── results/
├── main.py
├── pyproject.toml
└── uv.lock
```

## 2. 系统流程

Data -> Preprocess -> Features -> Train -> Evaluate -> Demo

## 3. 关键模块说明

- `src/dataset.py`
  - 支持两种数据源：`local` 与 `kagglehub`
  - 自动合并 Fake/True 并添加标签（Fake=0, Real=1）
  - 提供 `load_dataset(fake_path, real_path)` 课程友好接口

- `src/preprocess.py`
  - 参考课堂示例风格进行文本处理：
    - 去 URL
    - 小写化
    - 移除标点/非字母数字
    - 去除简单停用词

- `src/features.py`
  - TF-IDF 特征提取

- `src/models/`
  - `tfidf_model.py`: Logistic Regression
  - `lstm_model.py`: 可运行的 LSTM-style 基线封装（课程对比可直接跑）
  - `bert_model.py`: 可运行的 BERT-style 基线封装（课程对比可直接跑）

- `src/train.py`
  - `train_and_compare_models()` 一次训练三模型并输出对比表

- `src/evaluate.py`
  - 输出 Accuracy / Precision / Recall / F1

## 4. 快速开始

1. 安装依赖

```bash
uv sync
```

2. 本地数据训练 + 三模型对比

```bash
uv run python main.py --mode compare --data-source local
```

3. KaggleHub 拉取并训练

先配置 Kaggle API key（下载 `kaggle.json`）：

```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

然后运行：

```bash
uv run python main.py --mode compare --data-source kagglehub
```

4. 启动 Demo

```bash
uv run streamlit run demo/streamlit_app.py
```

## 5. 输出结果

- 训练好的模型文件：`results/*.joblib`
- 对比实验结果表：`results/metrics_comparison.csv`
