# Lab 08 — Naive Bayes and Text Classification
> Implement Bag of Words, TF-IDF, and Naive Bayes from scratch — classify SMS spam

**Prerequisites:** Read `theory/08-naive-bayes-and-text/GUIDE.md`
**Dataset:** SMS Spam Collection (UCI — download or from kaggle)
**Libraries:** numpy, pandas, matplotlib, sklearn

---

## Steps

### Step 1: Load and Explore
```
- Load SMS data (label + message text)
- Check class distribution: how many spam vs ham?
- Show 5 example spam and 5 ham messages
```

### Step 2: Text Preprocessing
```python
- Lowercase all text
- Remove punctuation and numbers
- Tokenize (split into words)
- Optional: remove stopwords
```

### Step 3: FROM SCRATCH — Bag of Words
```
- Build vocabulary: sorted list of all unique words
- For each message: count occurrences of each vocabulary word
- Result: document-term matrix [n_messages × vocab_size]
```

### Step 4: FROM SCRATCH — TF-IDF
```
- TF(t,d) = count(t in d) / total_words(d)
- IDF(t) = log(N / df(t))  where df = docs containing term t
- TF-IDF(t,d) = TF × IDF
- Apply to your count matrix
```

### Step 5: sklearn Vectorizers
```
- CountVectorizer → compare with your scratch BoW
- TfidfVectorizer → compare with your scratch TF-IDF
```

### Checkpoint: Your scratch vectors ≈ sklearn vectors

### Step 6: FROM SCRATCH — Naive Bayes
```python
# Training:
P_spam = count(spam) / count(total)
P_ham = 1 - P_spam

# For each word w:
P(w|spam) = (count(w in spam) + 1) / (total_words_spam + vocab_size)  # Laplace smoothing
P(w|ham)  = (count(w in ham) + 1) / (total_words_ham + vocab_size)

# Prediction (use LOG probabilities to avoid underflow):
log_P(spam|message) = log(P_spam) + Σ log(P(wᵢ|spam))
log_P(ham|message)  = log(P_ham)  + Σ log(P(wᵢ|ham))

predict = argmax(log_P(spam|msg), log_P(ham|msg))
```

### Step 7: sklearn MultinomialNB
```
- Fit on CountVectorizer output
- Compare accuracy with scratch version
```

### Step 8: Evaluate
```
- Confusion matrix
- Precision and recall (precision matters here — don't block real messages!)
- For spam filter: high precision > high recall
```

### Step 9: Top Spammy/Hammy Words
```
- Sort words by P(w|spam)/P(w|ham) ratio
- Top 20 "spammiest" words (e.g., "free", "win", "prize")
- Top 20 "hammiest" words (e.g., names, casual words)
```

### Step 10: Gaussian NB on Non-Text Data
```
- Try GaussianNB on Iris dataset (continuous features)
- Compare with other classifiers from previous labs
- Shows: NB works on non-text data too, just different variant
```

---

## Checkpoint

- [ ] BoW and TF-IDF implemented from scratch and matching sklearn
- [ ] Naive Bayes from scratch classifies spam with >95% accuracy
- [ ] Understand Laplace smoothing and log-probability trick
- [ ] Know which words are most indicative of spam
