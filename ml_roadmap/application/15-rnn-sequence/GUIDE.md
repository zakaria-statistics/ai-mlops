# Lab 15: RNN and Sequence Models
> RNN cell from scratch, vanishing gradients demo, LSTM/GRU for time series and text classification

## Table of Contents
1. [Setup](#setup) - Datasets and libraries
2. [RNN Cell from Scratch](#step-1-rnn-cell-from-scratch) - Implement the recurrence
3. [Vanishing Gradient Demo](#step-2-vanishing-gradient-demo) - Why simple RNNs fail
4. [PyTorch RNN for Time Series](#step-3-pytorch-rnn-for-time-series) - Predict next value
5. [PyTorch LSTM for Time Series](#step-4-pytorch-lstm-for-time-series) - Compare with RNN
6. [Compare Predictions](#step-5-compare-predictions) - RNN vs LSTM vs ARIMA
7. [Text Classification Setup](#step-6-text-classification-setup) - Prepare text data
8. [Word Embeddings](#step-7-word-embeddings) - nn.Embedding explained
9. [LSTM for Text Classification](#step-8-lstm-for-text-classification) - Sentiment analysis
10. [Compare with Naive Bayes](#step-9-compare-with-naive-bayes) - LSTM vs classical NLP
11. [GRU Comparison](#step-10-gru-comparison) - Faster alternative to LSTM

## Prerequisites
- Read `theory/15-rnn-sequence/GUIDE.md` first
- Completed Lab 14 (time series fundamentals, ARIMA baseline)
- Completed Lab 12 (PyTorch basics, training loops)
- Completed Lab 08 (Naive Bayes text classification for comparison)

## Datasets
- **Time series:** Airline Passengers (same as Lab 14) or a similar monthly series
- **Text:** IMDB sentiment dataset or a small text classification dataset
  - IMDB via torchtext or Hugging Face datasets
  - Alternative: sklearn's 20 newsgroups (smaller, faster)

## Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
```

---

## Step 1: RNN Cell from Scratch

Implement the simple RNN recurrence with numpy:

```
h_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)
```

1. Define dimensions: input_size=1, hidden_size=8
2. Initialize weights:
   ```python
   np.random.seed(42)
   input_size, hidden_size = 1, 8
   W_h = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden-to-hidden
   W_x = np.random.randn(hidden_size, input_size) * 0.01   # input-to-hidden
   b = np.zeros(hidden_size)
   ```
3. Create a simple input sequence (5 time steps):
   ```python
   x_seq = np.array([[0.1], [0.5], [0.9], [0.3], [0.7]])  # shape: (5, 1)
   ```
4. Forward pass through the sequence:
   ```python
   h = np.zeros(hidden_size)  # initial hidden state
   hidden_states = [h.copy()]
   for t in range(len(x_seq)):
       h = np.tanh(W_h @ h + W_x @ x_seq[t] + b)
       hidden_states.append(h.copy())
   ```
5. Print hidden state at each time step — each step accumulates information from all previous inputs
6. The final hidden state `h` summarizes the entire sequence

**Expected output:** Hidden states change at each step. Values are between -1 and 1 (due to tanh). The final state depends on all 5 inputs.

> **Checkpoint 1:** You understand the RNN recurrence — each step reads the previous hidden state plus current input.

---

## Step 2: Vanishing Gradient Demo

Show why simple RNNs struggle with long sequences:

1. Take a weight matrix and multiply it by itself many times:
   ```python
   W = np.random.randn(8, 8) * 0.5
   result = np.eye(8)
   norms = []
   for i in range(100):
       result = result @ W
       norms.append(np.linalg.norm(result))
   plt.plot(norms)
   plt.xlabel('Number of multiplications')
   plt.ylabel('Matrix norm')
   plt.title('Gradient magnitude over time steps')
   ```

2. Try with different W scales:
   - `W * 0.5` — norms go to 0 (vanishing)
   - `W * 1.5` — norms go to infinity (exploding)

3. This is why gradients from step 100 can't reach step 1 — the signal dies or explodes

4. LSTM solves this with a **cell state** that uses addition (not multiplication), allowing gradients to flow unchanged

**Expected output:** Clear demonstration — small W causes vanishing (curve drops to 0), large W causes exploding (curve shoots up). There's a narrow band where it's stable.

---

## Step 3: PyTorch RNN for Time Series

Predict next month's passengers using previous values:

1. Prepare windowed data from Airline Passengers:
   ```python
   def create_sequences(data, seq_length):
       X, y = [], []
       for i in range(len(data) - seq_length):
           X.append(data[i:i+seq_length])
           y.append(data[i+seq_length])
       return np.array(X), np.array(y)

   # Normalize
   from sklearn.preprocessing import MinMaxScaler
   scaler = MinMaxScaler()
   scaled = scaler.fit_transform(passengers.values.reshape(-1, 1))

   seq_length = 12  # use 12 months to predict next month
   X, y = create_sequences(scaled, seq_length)
   ```

2. Build RNN model:
   ```python
   class TimeSeriesRNN(nn.Module):
       def __init__(self, input_size=1, hidden_size=32, num_layers=1):
           super().__init__()
           self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
           self.fc = nn.Linear(hidden_size, 1)

       def forward(self, x):
           out, _ = self.rnn(x)
           out = self.fc(out[:, -1, :])  # take last time step
           return out
   ```

3. Train/test split (chronological), train for 100 epochs
4. Plot predictions vs actual on test set

**Expected output:** RNN captures general trend but may miss sharp seasonal peaks. Predictions are smoother than actual values.

---

## Step 4: PyTorch LSTM for Time Series

Replace RNN with LSTM — same task, same data:

1. Build LSTM model:
   ```python
   class TimeSeriesLSTM(nn.Module):
       def __init__(self, input_size=1, hidden_size=32, num_layers=1):
           super().__init__()
           self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
           self.fc = nn.Linear(hidden_size, 1)

       def forward(self, x):
           out, (h, c) = self.lstm(x)
           out = self.fc(out[:, -1, :])
           return out
   ```

2. Train with same hyperparameters as the RNN
3. Plot predictions vs actual

**Expected output:** LSTM should capture seasonality better than simple RNN, especially for longer sequences.

> **Checkpoint 2:** You've trained both RNN and LSTM on time series. LSTM handles long-term dependencies better.

---

## Step 5: Compare Predictions

1. Plot all predictions on the same test period:
   - Actual values
   - RNN predictions
   - LSTM predictions
   - ARIMA predictions (from Lab 14)

2. Compute MAE/RMSE for each:
   ```
   | Model       | MAE  | RMSE |
   |-------------|------|------|
   | ARIMA       |      |      |
   | Simple RNN  |      |      |
   | LSTM        |      |      |
   ```

3. Key insight: for this small, clean dataset, ARIMA may actually beat RNN/LSTM. Deep learning needs more data to shine. But LSTM handles multivariate and non-linear patterns better at scale.

**Expected output:** ARIMA may win on this small dataset. LSTM should be competitive. Simple RNN likely trails.

---

## Step 6: Text Classification Setup

Switch to text data for sequence classification:

1. Load a text dataset:
   ```python
   # Option A: IMDB (via torchtext or manual download)
   # Option B: sklearn 20 newsgroups (simpler)
   from sklearn.datasets import fetch_20newsgroups
   cats = ['sci.space', 'rec.sport.baseball']  # binary classification
   train = fetch_20newsgroups(subset='train', categories=cats)
   test = fetch_20newsgroups(subset='test', categories=cats)
   ```

2. Build a vocabulary from training text:
   ```python
   from collections import Counter
   word_counts = Counter()
   for text in train.data:
       word_counts.update(text.lower().split())

   vocab = {word: idx+2 for idx, (word, _) in enumerate(word_counts.most_common(5000))}
   vocab['<PAD>'] = 0
   vocab['<UNK>'] = 1
   ```

3. Convert texts to integer sequences and pad to fixed length:
   ```python
   def text_to_seq(text, vocab, max_len=200):
       tokens = text.lower().split()
       seq = [vocab.get(t, 1) for t in tokens[:max_len]]
       seq += [0] * (max_len - len(seq))  # pad
       return seq
   ```

**Expected output:** Each text becomes a fixed-length sequence of integers. Vocabulary size ~5000.

---

## Step 7: Word Embeddings

1. Understand `nn.Embedding`:
   ```python
   embed = nn.Embedding(num_embeddings=5000, embedding_dim=64)
   # Input: integer sequence (batch, seq_len)
   # Output: (batch, seq_len, 64) — each word becomes a 64-dim vector
   ```

2. Demonstrate: same word always maps to the same vector
3. The embedding is learned during training — similar words end up with similar vectors
4. This replaces TF-IDF: instead of sparse hand-crafted features, the network learns dense representations

**Expected output:** Embedding layer takes integer indices and returns learned dense vectors. Shape: (batch, seq_len) -> (batch, seq_len, embed_dim).

---

## Step 8: LSTM for Text Classification

1. Build the model:
   ```python
   class TextLSTM(nn.Module):
       def __init__(self, vocab_size, embed_dim=64, hidden_dim=64):
           super().__init__()
           self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
           self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
           self.fc = nn.Linear(hidden_dim, 1)
           self.sigmoid = nn.Sigmoid()

       def forward(self, x):
           x = self.embedding(x)           # (batch, seq_len, embed_dim)
           _, (h, _) = self.lstm(x)         # h: (1, batch, hidden_dim)
           out = self.sigmoid(self.fc(h.squeeze(0)))
           return out
   ```

2. Train for 10-20 epochs with BCELoss and Adam
3. Track train/val accuracy each epoch
4. Evaluate on test set

**Expected output:** Test accuracy around 90-95% on the 2-class newsgroup task. The LSTM learns to distinguish topic-specific vocabulary.

> **Checkpoint 3:** You've built an LSTM text classifier with learned word embeddings.

---

## Step 9: Compare with Naive Bayes

1. From Lab 08, train a Naive Bayes classifier with TF-IDF on the same data:
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.naive_bayes import MultinomialNB
   from sklearn.metrics import accuracy_score

   tfidf = TfidfVectorizer(max_features=5000)
   X_train_tfidf = tfidf.fit_transform(train.data)
   X_test_tfidf = tfidf.transform(test.data)

   nb = MultinomialNB()
   nb.fit(X_train_tfidf, train.target)
   nb_acc = accuracy_score(test.target, nb.predict(X_test_tfidf))
   ```

2. Compare:
   ```
   | Model                | Test Accuracy | Training Time |
   |----------------------|--------------|---------------|
   | Naive Bayes + TF-IDF |              |               |
   | LSTM + Embedding     |              |               |
   ```

3. Key insight: for simple binary text classification with clear topic separation, Naive Bayes is surprisingly competitive and much faster. LSTM shines on subtler tasks (sentiment, sarcasm) where word order matters.

**Expected output:** Both should be above 90%. Naive Bayes trains in seconds; LSTM takes minutes. For simple topic classification, Naive Bayes may match LSTM.

---

## Step 10: GRU Comparison

1. Replace LSTM with GRU — same architecture:
   ```python
   class TextGRU(nn.Module):
       def __init__(self, vocab_size, embed_dim=64, hidden_dim=64):
           super().__init__()
           self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
           self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
           self.fc = nn.Linear(hidden_dim, 1)
           self.sigmoid = nn.Sigmoid()

       def forward(self, x):
           x = self.embedding(x)
           _, h = self.gru(x)               # GRU: no cell state, just hidden
           out = self.sigmoid(self.fc(h.squeeze(0)))
           return out
   ```

2. Train on same data, same hyperparameters
3. Compare with LSTM:
   ```
   | Model | Test Accuracy | Training Time | Parameters |
   |-------|--------------|---------------|------------|
   | RNN   |              |               |            |
   | LSTM  |              |               |            |
   | GRU   |              |               |            |
   ```

4. GRU has fewer parameters than LSTM (2 gates vs 3) — slightly faster
5. Accuracy is usually very similar

**Expected output:** GRU matches LSTM accuracy within 1-2%, trains slightly faster, has ~25% fewer parameters.

> **Checkpoint 4:** You understand the RNN family — simple RNN, LSTM, GRU — their tradeoffs, and when each is appropriate. You can apply them to both time series and text.

---

## Summary Deliverables
- [ ] RNN cell from scratch (numpy forward pass)
- [ ] Vanishing gradient demonstration
- [ ] RNN and LSTM trained on time series
- [ ] Comparison: RNN vs LSTM vs ARIMA on same forecast task
- [ ] LSTM text classifier with learned embeddings
- [ ] Naive Bayes vs LSTM comparison
- [ ] GRU vs LSTM speed/accuracy comparison
- [ ] Understanding of when to use RNNs vs classical methods
