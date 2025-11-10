# 🧠 Sentiment Analysis RNN Comparison

## 📄 Overview
This project compares multiple **Recurrent Neural Network (RNN)** architectures — **Simple RNN, LSTM, and Bidirectional LSTM** — for **binary sentiment classification** on the **IMDb Movie Review Dataset**.

The goal is to systematically evaluate performance across:
- Architectures
- Activation functions
- Optimizers
- Sequence lengths
- Gradient clipping

All experiments are **fully reproducible** with a fixed random seed.

---

## 🚀 Features
- Text preprocessing: lowercase, punctuation removal, top 10,000 words, padding/truncating
- Models implemented in **PyTorch**:
  - Simple RNN
  - LSTM
  - Bidirectional LSTM
- Hyperparameter ablation study (10 experiments)
- Metrics: **Accuracy, F1-Score, Training Time**
- Visualizations: accuracy, training time, optimizer & sequence impact
- Detailed **PDF report** included: `Homework_3_Report.pdf`

---

## 🧰 Technologies Used
- **Python 3.10**
- **PyTorch 2.0**
- **NumPy, Pandas, Matplotlib**
- **IMDb Dataset** (50,000 balanced reviews)

---

## 📊 Results Summary

| Arch        | Activ | Optim   | Seq | Clip | Acc     | F1      | Time     |
|-------------|-------|---------|-----|------|---------|---------|----------|
| **LSTM**    | tanh  | adam    | 100 | 1.0  | **79.54%** | **0.7944** | 201.1s |
| BiLSTM      | tanh  | adam    | 50  | 1.0  | 76.79%  | 0.7671  | 340.4s   |
| LSTM        | tanh  | rmsprop | 50  | 1.0  | 76.04%  | 0.7603  | 149.8s   |
| LSTM        | tanh  | adam    | 50  | 1.0  | 75.81%  | 0.7580  | 1567.6s  |
| LSTM        | relu  | adam    | 50  | 1.0  | 75.73%  | 0.7572  | 181.8s   |
| LSTM        | sigmoid| adam   | 50  | 1.0  | 75.53%  | 0.7544  | 171.0s   |
| LSTM        | tanh  | adam    | 50  | None | 75.47%  | 0.7544  | 84.9s    |
| LSTM        | tanh  | adam    | 25  | 1.0  | 71.09%  | 0.7108  | 81.0s    |
| RNN         | tanh  | adam    | 50  | 1.0  | 63.08%  | 0.6048  | 78.6s    |
| LSTM        | tanh  | sgd     | 50  | 1.0  | 51.72%  | 0.5032  | 137.0s   |

> **Best Model**: `LSTM`, `tanh`, `adam`, `seq=100`, `clip=1.0` → **79.54% Accuracy**

---

## 📈 Key Insights
- **LSTM > BiLSTM > RNN** in performance
- **Longer sequences (100 words)** → better accuracy
- **Adam** outperforms SGD and RMSprop
- **Tanh** is best activation for RNNs
- **Gradient clipping (1.0)** improves stability
- **BiLSTM** gives marginal gain at ~2x training cost

---

## 🛠️ Installation

```bash
git clone https://github.com/sravani150602/sentiment-analysis-rnn-comparison.git
cd sentiment-analysis-rnn-comparison
pip install torch numpy pandas matplotlib scikit-learn


#**# Optimal Configuration (79.54% Accuracy)**
Architecture: LSTM
Sequence Length: 100 words
Optimizer: Adam (lr=0.001)
Activation: Tanh
Gradient Clipping: 1.0
Embedding Dim: 100
Hidden Layers: 2 × 64 units
Dropout: 0.3
Batch Size: 32
Epochs: 5
Loss: Binary Cross-Entropy
Random Seed: 42
