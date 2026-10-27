# Customer Churn Prediction & Salary Estimation (ANN)

An end-to-end deep learning project utilizing **Artificial Neural Networks (ANN)** for tabular data classification and regression, built with **TensorFlow/Keras**, **Scikit-Learn**, and deployed with **Streamlit**.

---

## 📌 Project Overview

This sub-project tackles two real-world tabular data problems using the Bank Customer Churn dataset:
1. **Customer Churn Classification**: Predicting whether a customer will exit the bank based on their financial and demographic behavior.
2. **Salary Estimation (Regression)**: Predicting customer salary based on bank relationship metrics.
3. **Hyperparameter Tuning**: Finding optimal neural network architectures (neurons, hidden layers, batch size) via grid search cross-validation.

---

## 📁 Directory Structure

```text
ANN Classification/
├── Churn_Modelling.csv          # Source dataset (10,000 customer records)
├── experiments.ipynb            # Data cleaning, EDA, ANN training & evaluation
├── hyperparaNN.ipynb            # Grid search hyperparameter optimization (SciKeras)
├── salaryregression.ipynb       # ANN regression model for salary prediction
├── prediction.ipynb             # Testing inference pipeline on unseen inputs
├── app.py                       # Interactive Streamlit web application
├── model.h5                     # Saved trained ANN classification model
├── scaler.pkl                   # Pickled StandardScaler object
├── label_encoder_gender.pkl     # Pickled LabelEncoder for Gender
├── onehot_encoder_geo.pkl       # Pickled OneHotEncoder for Geography
├── requirements.txt             # Project-specific dependencies
├── logs/                        # TensorBoard logs for churn classification
└── regression_logs/             # TensorBoard logs for salary regression
```

---

## 📊 Dataset: Bank Customer Churn

The dataset contains **10,000 customer records** with 14 attributes:

| Feature | Type | Description |
|---|---|---|
| `RowNumber`, `CustomerId`, `Surname` | Identifiers | Dropped during preprocessing |
| `CreditScore` | Numeric | Customer credit rating score |
| `Geography` | Categorical | Country (`France`, `Germany`, `Spain`) |
| `Gender` | Categorical | `Male`, `Female` |
| `Age` | Numeric | Customer age (18 - 100) |
| `Tenure` | Numeric | Number of years as a bank client (0 - 10) |
| `Balance` | Numeric | Current account balance |
| `NumOfProducts` | Numeric | Number of bank products used (1 - 4) |
| `HasCrCard` | Binary | Customer possesses credit card (0 or 1) |
| `IsActiveMember` | Binary | Active status in the bank (0 or 1) |
| `EstimatedSalary` | Numeric | Annual estimated salary |
| `Exited` | Binary (Target) | 1 if customer churned, 0 otherwise |

---

## 🛠 Data Preprocessing Pipeline

1. **Feature Pruning**: `RowNumber`, `CustomerId`, and `Surname` are dropped.
2. **Categorical Encoding**:
   - `Gender`: Encoded using `LabelEncoder()` (saved to `label_encoder_gender.pkl`).
   - `Geography`: Encoded using `OneHotEncoder()` to prevent artificial ordinal hierarchy (saved to `onehot_encoder_geo.pkl`).
3. **Train-Test Split**: 80% train, 20% test split.
4. **Feature Scaling**: Numerical features scaled using `StandardScaler()` (saved to `scaler.pkl`) to stabilize gradient updates.

---

## 🧠 Model Architectures

### 1. Churn Classification ANN (`model.h5`)
```
Input (11 features)
       │
       ▼
Dense Layer (64 units, ReLU)
       │
       ▼
Dense Layer (32 units, ReLU)
       │
       ▼
Dense Layer (1 unit, Sigmoid)
```
- **Loss**: `binary_crossentropy`
- **Optimizer**: `adam`
- **Metrics**: `['accuracy']`
- **Callbacks**:
  - `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`
  - `TensorBoard(log_dir=logs/...)`

### 2. Salary Regression ANN
- **Task**: Predict continuous `EstimatedSalary`.
- **Target**: `EstimatedSalary` (Features exclude salary; `Exited` is included).
- **Architecture**: Sequential Dense layers with ReLU activations and a single linear output neuron.
- **Loss**: `mae` / `mean_squared_error`
- **Callbacks**: Early stopping and TensorBoard logging (`regression_logs/fit`).

### 3. Hyperparameter Tuning (`hyperparaNN.ipynb`)
- **Wrapper**: `SciKeras` (`KerasClassifier`)
- **Search Strategy**: `GridSearchCV` (3-fold cross-validation)
- **Tuned Hyperparameters**:
  - Neurons per hidden layer (e.g., 16, 32, 64)
  - Number of hidden layers (1 to 3)
  - Batch sizes and epoch counts

---

## 🖥 Streamlit Web Application (`app.py`)

An interactive UI that enables real-time prediction for individual customers.

### Inputs:
- Dropdown: Geography (France, Germany, Spain)
- Dropdown: Gender (Female, Male)
- Sliders: Age (18-100), Tenure (0-10), Number of Products (1-4)
- Numeric inputs: Credit Score, Balance, Estimated Salary
- Binary selectors: Has Credit Card (0/1), Is Active Member (0/1)

### Pipeline execution in `app.py`:
1. Formulates input row into a pandas DataFrame.
2. Transforms `Gender` using `label_encoder_gender.pkl`.
3. One-hot encodes `Geography` using `onehot_encoder_geo.pkl`.
4. Scales all 11 features with `scaler.pkl`.
5. Feeds input vector to `model.h5` and outputs prediction probability ($> 0.5 \to$ Churn alert).

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Application
```bash
streamlit run app.py
```
Access the application at `http://localhost:8501`.

### 3. Visualize Training with TensorBoard
To review the training history, loss convergence, and validation metrics:
```bash
# Classification logs
tensorboard --logdir logs

# Regression logs
tensorboard --logdir regression_logs/fit
```
Then visit `http://localhost:6006`.
