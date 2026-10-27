# Deep Learning

A centralized repository containing end-to-end deep learning projects covering **tabular neural networks**, **recurrent sequence modeling**, and **natural language processing (NLP)** using **TensorFlow/Keras**, **Scikit-Learn**, and **Streamlit**.

---

## 📂 Projects Overview

This repository is organized into three standalone, fully functional deep learning modules:

| Project | Architecture | Problem Type | Dataset | Web App / Demo | In-Depth Docs |
|---|---|---|---|---|---|
| **1. [ANN Classification](./ANN%20Classification)** | ANN (Dense, Dropout) | Binary Classification & Regression | Bank Customer Churn (10k records) | Streamlit (`app.py`) | [Read ANN Guide](./ANN%20Classification/README.md) |
| **2. [LSTM + RNN](./LSTM+RNN)** | Stacked LSTM & Stacked GRU | Language Modeling / Next Word Prediction | Shakespeare's *Hamlet* | Streamlit (`app.py`) | [Read LSTM/GRU Guide](./LSTM+RNN/README.md) |
| **3. [RNN IMDB](./RNN_IMDB)** | SimpleRNN + Embeddings | Binary Sentiment Analysis | IMDB Movie Reviews (50k reviews) | Jupyter Notebooks | [Read Simple RNN Guide](./RNN_IMDB/README.md) |

---

## ⚡ Quick Start

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # On Windows

# Install all shared dependencies
pip install -r requirements.txt
```

### 2. Launch Interactive Applications

- **Customer Churn Predictor (ANN)**:
  ```bash
  cd "ANN Classification"
  streamlit run app.py
  ```

- **Next Word Predictor (LSTM vs. GRU Comparison)**:
  ```bash
  cd "LSTM+RNN"
  streamlit run app.py
  ```

---

## 📁 Repository Structure

```text
Deep Learning/
├── requirements.txt            # Global Python dependencies
├── README.md                   # Repository overview (this file)
│
├── ANN Classification/         # Project 1: Tabular Churn Prediction & Salary Regression
│   └── README.md               # Detailed ANN documentation & architecture details
│
├── LSTM+RNN/                   # Project 2: Next Word Prediction with LSTM & GRU
│   └── README.md               # Detailed NLP & sequential modeling documentation
│
└── RNN_IMDB/                   # Project 3: Movie Review Sentiment Analysis
    └── README.md               # Detailed SimpleRNN & embedding layer documentation
```

> 💡 **For detailed architecture diagrams, training parameters, mathematical pipelines, and notebook walkthroughs, check the dedicated `README.md` inside each project folder.**
