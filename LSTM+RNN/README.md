# Next Word Prediction with Stacked LSTM & GRU

An end-to-end Natural Language Processing (NLP) text generation project comparing **Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)** architectures for next-word prediction on Shakespeare's *Hamlet*, complete with an interactive **Streamlit** side-by-side comparison app.

---

## 📌 Project Overview

Language modeling is a core task in sequence learning where the objective is to predict the next token given preceding context:

$$P(w_t \mid w_1, w_2, \dots, w_{t-1})$$

In this project:
- We tokenize and vectorize Shakespeare's *Hamlet* (`shakespeare-hamlet.txt`).
- We train two recurrent architectures: **Stacked LSTM** and **Stacked GRU**.
- We save the models and tokenizers for fast inference.
- We build a Streamlit web interface allowing users to input a sentence prompt and view real-time predictions from both models simultaneously.

---

## 📁 Directory Structure

```text
LSTM+RNN/
├── shakespeare-hamlet.txt     # Raw textual dataset (Shakespeare's Hamlet)
├── experiments.ipynb          # End-to-end data pipeline, model training & evaluation
├── app.py                     # Streamlit app comparing LSTM & GRU predictions
├── next_word_lstm.h5          # Trained Stacked LSTM model
├── next_word_gru.h5           # Trained Stacked GRU model
├── tokenizer.pickle           # Saved Keras Tokenizer for the LSTM model
└── tokenizer_gru.pickle       # Saved Keras Tokenizer for the GRU model
```

---

## 🛠 NLP Preprocessing Pipeline

1. **Text Corpus Ingestion**:
   - The entire text of *Hamlet* is loaded and converted to lowercase.
   - Text is split line-by-line into sequences.
2. **Tokenization**:
   - `tensorflow.keras.preprocessing.text.Tokenizer` assigns an integer ID to each unique word.
   - `total_words = len(tokenizer.word_index) + 1` forms the vocabulary size.
3. **N-gram Sequence Generation**:
   - For every sentence/line, rolling sub-sequences are generated:
     - Example: `"to be or not to be"`
     - $\to$ `["to", "be"]`
     - $\to$ `["to", "be", "or"]`
     - $\to$ `["to", "be", "or", "not"]`
     - $\to$ `["to", "be", "or", "not", "to"]`
     - $\to$ `["to", "be", "or", "not", "to", "be"]`
4. **Padding**:
   - Sequences are pre-padded with zeros up to `max_sequence_len` using `pad_sequences(..., padding='pre')`.
5. **Feature-Label Separation**:
   - Input $X$: The first $N-1$ tokens of the padded sequence.
   - Target $y$: The last token ($N$-th word), one-hot encoded using `to_categorical(y, num_classes=total_words)`.
6. **Data Partitioning**: 80% training split, 20% validation split.

---

## 🧠 Model Architectures

### 1. Stacked LSTM (`next_word_lstm.h5`)
```
Input Sequence (Padded Token IDs)
             │
             ▼
Embedding Layer (input_dim=total_words, output_dim=100)
             │
             ▼
LSTM Layer (150 units, return_sequences=True)
             │
             ▼
Dropout Layer (rate=0.2)
             │
             ▼
LSTM Layer (100 units, return_sequences=False)
             │
             ▼
Dense Layer (total_words units, Softmax)
```
- **Loss**: `categorical_crossentropy`
- **Optimizer**: `adam`
- **Metrics**: `['accuracy']`

### 2. Stacked GRU (`next_word_gru.h5`)
```
Input Sequence (Padded Token IDs)
             │
             ▼
Embedding Layer (input_dim=total_words, output_dim=100)
             │
             ▼
GRU Layer (150 units, return_sequences=True)
             │
             ▼
Dropout Layer (rate=0.2)
             │
             ▼
GRU Layer (100 units, return_sequences=False)
             │
             ▼
Dense Layer (total_words units, Softmax)
```
- **Loss**: `categorical_crossentropy`
- **Optimizer**: `adam`
- **Metrics**: `['accuracy']`

---

## ⚖️ LSTM vs. GRU Comparison

| Feature | LSTM (Long Short-Term Memory) | GRU (Gated Recurrent Unit) |
|---|---|---|
| **Gates** | 3 (Forget, Input, Output) | 2 (Reset, Update) |
| **Internal States** | Hidden state ($h_t$) + Cell state ($c_t$) | Hidden state ($h_t$) only |
| **Parameters** | More parameters ($4 \times$ state size) | Fewer parameters ($3 \times$ state size) |
| **Training Speed** | Slower per epoch | Faster per epoch |
| **Strengths** | Handles intricate long-term dependencies | Faster convergence; efficient on smaller datasets |

---

## 🖥 Streamlit Web Application (`app.py`)

The application loads both trained models (`next_word_lstm.h5`, `next_word_gru.h5`) and both tokenizers (`tokenizer.pickle`, `tokenizer_gru.pickle`).

### Prediction Flow:
1. User provides a text prompt (e.g., *"To be or not to"*).
2. The prompt is converted into integer tokens via the tokenizer.
3. If prompt length $\ge \text{max\_sequence\_len}$, it keeps the trailing tokens.
4. Input sequence is pre-padded to match model input dimensions.
5. Models output softmax probabilities across the entire vocabulary.
6. `np.argmax(predicted, axis=1)` extracts the most probable token index.
7. Reverse index lookup maps the token index back to the human-readable word.
8. Displays both LSTM and GRU predictions side-by-side.

---

## 🚀 How to Run

### 1. Ensure Dependencies are Installed
From the repository root or project folder:
```bash
pip install tensorflow streamlit numpy pandas
```

### 2. Launch the Streamlit Web Application
```bash
cd "LSTM+RNN"
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Interactive Notebook Exploration
Open `experiments.ipynb` to view:
- Distribution of sequence lengths.
- Training loss curves over 50 epochs.
- Text generation tests with sample phrases.
