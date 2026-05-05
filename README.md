<div align="center">

# 🧠 ChurnSense AI
### Customer Churn Prediction · Deep Learning · ANN Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

> **Predict whether a bank customer will churn — in real time — using a trained Artificial Neural Network.**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Running the App](#-running-the-app)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Results](#-results)

---

##  Overview

**ChurnSense AI** is a full-stack machine learning application that predicts customer churn for a bank using a trained **Artificial Neural Network (ANN)**. Built with TensorFlow/Keras on the backend and a sleek dark-themed Streamlit interface on the frontend, it lets you plug in any customer's profile and instantly see their churn risk with a probability score and confidence meter.

The model was trained on 10,000 real-world banking records and achieves strong classification performance across three geographies, two genders, and a variety of financial indicators.

---

##  Demo

```
https://deep-learning-churn-classifier-bc8f9gin67od46fucqwwab.streamlit.app/
```

Fill in the customer's demographics, account details, and product usage → click **⚡ Run Churn Prediction** → get an instant verdict with probability, risk badge, and confidence score.

| High Risk Output | ✅ Low Risk Output |
|---|---|
| Red result card | Green result card |
| Churn probability > 50% | Churn probability ≤ 50% |
| "Immediate retention action recommended" | "Customer appears stable. Monitor quarterly." |

---

##  Features

- **Real-time ANN inference** — predictions in milliseconds using a cached TensorFlow model
- **Full preprocessing pipeline** — Label Encoding, One-Hot Encoding, and Standard Scaling applied automatically
- **Probability meter** — visual fill bar from 0% to 100% with a clear 50% decision threshold marker
- **Model confidence score** — shows how decisive the prediction is, not just the raw probability
- **Customer profile summary** — a 6-tile grid recapping all input values alongside the result
- **Stunning dark UI** — cosmic purple/cyan theme with `Syne` + `Space Mono` typography
- **Responsive 2-column layout** — inputs on the left, live results on the right
- **Cached model loading** — `@st.cache_resource` ensures the model loads only once per session

---

## 📁 Project Structure

```
annclassification/
│
├── app.py                        #  Main Streamlit application
├── model.h5                      #  Trained ANN model (TensorFlow/Keras)
│
├── label_encoder_gender.pkl      #  LabelEncoder for Gender feature
├── onehot_encoder_geo.pkl        #  OneHotEncoder for Geography feature
├── scaler.pkl                    #  StandardScaler for numerical features
│
├── Churn_Modelling.csv           #  Training dataset (10,000 records)
├── requirements.txt              #  Python dependencies
│
├── experiments.ipynb             #  Model training & experimentation
├── hyperparametertuningann.ipynb #  Hyperparameter tuning with Keras Tuner
├── prediction.ipynb              #  Inference and evaluation notebook
└── salaryregression.ipynb        #  Bonus: Salary regression ANN
```

---

##  Model Architecture

The ANN is built with TensorFlow/Keras and trained for binary classification (churn = 1 / no churn = 0).

```
Input Layer     →  11 features (after encoding)
                        ↓
Hidden Layer 1  →  Dense(64, activation='relu')  +  Dropout
                        ↓
Hidden Layer 2  →  Dense(32, activation='relu')  +  Dropout
                        ↓
Output Layer    →  Dense(1, activation='sigmoid')
                        ↓
              Binary Cross-Entropy Loss · Adam Optimizer
```

**Input Features (post-encoding):**

| Feature | Type | Encoding |
|---|---|---|
| CreditScore | Numerical | StandardScaler |
| Geography | Categorical | One-Hot (3 cols) |
| Gender | Categorical | Label Encoding |
| Age | Numerical | StandardScaler |
| Tenure | Numerical | StandardScaler |
| Balance | Numerical | StandardScaler |
| NumOfProducts | Numerical | StandardScaler |
| HasCrCard | Binary | As-is |
| IsActiveMember | Binary | As-is |
| EstimatedSalary | Numerical | StandardScaler |

---

## 📊 Dataset

**File:** `Churn_Modelling.csv`

| Property | Value |
|---|---|
| Records | 10,000 customers |
| Target | `Exited` (1 = churned, 0 = retained) |
| Geographies | France, Germany, Spain |
| Features | 14 columns (10 used for training) |
| Class balance | ~20% churn / ~80% retained |

The dataset is a classic bank customer churn benchmark, widely used for classification and ANN tutorials.

---

##  Installation

**1. Clone the repository**
```bash
git clone https://github.com/Tulsi14/Deep-Learning-Churn-Classifier.git
cd annclassification
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
```
tensorflow==2.15.0
pandas
numpy
scikit-learn
tensorboard==2.15.0
matplotlib
seaborn
streamlit
```

> TensorFlow 2.15 requires **Python 3.9–3.11**. Python 3.12+ is not yet supported.

---

##  Running the App

Make sure you are inside the `annclassification/` directory so the model and encoder files are found correctly:

```bash
cd annclassification
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

##  How It Works

```
User Input (Streamlit UI)
        ↓
Label Encode Gender          →  label_encoder_gender.pkl
One-Hot Encode Geography     →  onehot_encoder_geo.pkl
Standard Scale all features  →  scaler.pkl
        ↓
Concatenate into feature vector (shape: 1 × 12)
        ↓
ANN Forward Pass             →  model.h5
        ↓
Sigmoid Output  →  Churn Probability (0.0 – 1.0)
        ↓
Threshold @ 0.5  →  CHURN / NO CHURN verdict
```

The preprocessing artefacts are loaded **once** at startup using `@st.cache_resource`, so subsequent predictions are near-instant.

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow 2.15 / Keras |
| Preprocessing | Scikit-Learn (LabelEncoder, OneHotEncoder, StandardScaler) |
| Frontend | Streamlit |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib (notebooks) |
| Hyperparameter Tuning | Keras Tuner / SciKeras |

---

##  Results

The trained model achieves the following on the held-out test set:

| Metric | Score |
|---|---|
| Accuracy | ~86% |
| Precision | ~75% |
| Recall | ~48% |
| AUC-ROC | ~87% |

> Exact numbers depend on the random seed and train/test split. Check `experiments.ipynb` for full evaluation details and confusion matrix.

---

##  Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

##  License

This project is open-source and available under the [MIT License](LICENSE).

---

<div align="center">

Built with ❤️ using **TensorFlow** + **Streamlit**

</div>
