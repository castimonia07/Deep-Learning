# IMDB Sentiment Analysis with Simple RNN

An end-to-end Natural Language Processing (NLP) deep learning project implementing a **Simple Recurrent Neural Network (SimpleRNN)** with **Word Embeddings** for binary sentiment classification on the **IMDB Movie Reviews** dataset using **TensorFlow/Keras**.

---

## 📌 Project Overview

Sentiment analysis classifies the emotional polarity of unstructured natural text. In this project:
- We investigate the mechanics of dense vector representations using Keras `Embedding` layers compared to sparse one-hot encoding.
- We build and train a **Simple Recurrent Neural Network (SimpleRNN)** on 50,000 polar movie reviews (25,000 train, 25,000 test).
- We implement regularized training with `EarlyStopping` to combat overfitting.
- We write a production-grade inference routine that accepts raw text strings, maps tokens with proper special-index offsets, and returns predicted sentiment (Positive / Negative) with confidence probability scores.

---

## 📁 Directory Structure

```text
RNN_IMDB/
├── embedding.ipynb          # Exploration of word embeddings & Keras Embedding layers
├── simplern.ipynb           # Model architecture, training with EarlyStopping, and evaluation
├── prediction.ipynb         # End-to-end inference pipeline for custom user reviews
└── simple_rnn_imdb.h5       # Saved trained SimpleRNN model
```

---

## 🔬 Core Concepts & Notebook Breakdown

### 1. Word Embeddings (`embedding.ipynb`)
- **Limitation of One-Hot Encoding**: Sparse, high-dimensional ($O(V)$), lacks semantic similarity (vectors for "king" and "queen" are orthogonal).
- **Dense Embeddings**: Maps discrete token integers into a continuous vector space $\mathbb{R}^D$ where geometric proximity corresponds to semantic similarity.
- **Keras Embedding Layer**:
  ```python
  Embedding(input_dim=voc_size, output_dim=dim, input_length=sent_length)
  ```
  Transforms integer sequences of shape `(batch_size, sequence_length)` into dense representation of shape `(batch_size, sequence_length, embedding_dim)`.

---

### 2. Simple RNN Training (`simplern.ipynb`)

#### Data Ingestion & Index Offsets:
- The IMDB dataset is loaded via `tensorflow.keras.datasets.imdb`:
  - Vocabulary size constrained to top 10,000 frequent words (`num_words=10000`).
  - Sequence length capped and padded to 500 words (`sequence.pad_sequences(..., maxlen=500)`).
- **Understanding Special Token Offsets**:
  - `0`: `<PAD>`
  - `1`: `<START>`
  - `2`: `<UNK>` (Out of vocabulary)
  - `3`: Unused
  - Actual words begin at index 3 (`index - 3`). To decode reviews:
    ```python
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in sample_review])
    ```

#### Model Architecture:
```
Input (Padded Sequence of length 500)
                 │
                 ▼
Embedding Layer (input_dim=10000, output_dim=128)
                 │
                 ▼
SimpleRNN Layer (128 units, activation='relu')
                 │
                 ▼
Dense Layer (1 unit, activation='sigmoid')
```

#### Training Specifications:
- **Loss Function**: `binary_crossentropy`
- **Optimizer**: `adam`
- **Metrics**: `['accuracy']`
- **Regularization**: `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)` prevents memorization of noisy review patterns.
- **Artifact Export**: Weights and topology saved to `simple_rnn_imdb.h5`.

---

### 3. Custom Review Prediction Pipeline (`prediction.ipynb`)

The model expects token IDs rather than plain text. A dedicated preprocessing and inference pipeline was engineered:

```
Raw Review Text ("This movie was very fantastic.")
                     │
                     ▼
Lowercasing & Word Splitting
                     │
                     ▼
Token Mapping using IMDB Word Index
(unseen words mapped to index 2 / <UNK>; word index shifted by 3)
                     │
                     ▼
Sequence Padding (maxlen=500, padding='pre')
                     │
                     ▼
Model Prediction (simple_rnn_imdb.h5)
                     │
                     ▼
Thresholding (> 0.5: Positive, <= 0.5: Negative)
```

#### Example Usage:
```python
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

sentiment, score = predict_sentiment("This movie was very fantastic.")
# Returns: Sentiment: Positive, Score: 0.89...
```

---

## 📊 Summary of Model Performance

- **Sequence Length**: 500 tokens
- **Vocabulary Size**: 10,000
- **Embedding Dimensions**: 128
- **SimpleRNN Units**: 128
- **Output**: Single scalar probability ($p \in [0, 1]$) indicating positive sentiment.

---

## 🚀 How to Run

### 1. Requirements
Ensure you have TensorFlow and standard scientific libraries installed:
```bash
pip install tensorflow numpy matplotlib
```

### 2. Running the Notebooks
Launch your Jupyter environment:
```bash
jupyter notebook
```
- Open `simplern.ipynb` to retrain the network or modify sequence lengths / embedding dimensions.
- Open `prediction.ipynb` to test sentiment classification on custom written movie reviews.
