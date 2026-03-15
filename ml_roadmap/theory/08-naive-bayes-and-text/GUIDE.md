# Lab 08 Theory — Naive Bayes & Text Classification
> Apply Bayes' theorem with a bold independence assumption to classify text fast.

## Table of Contents
1. [Bayes' Theorem](#1-bayes-theorem) — The foundation
2. [Prior, Likelihood, Posterior](#2-prior-likelihood-posterior) — The three pieces
3. [The Naive Assumption](#3-the-naive-assumption) — Why "naive" and why it works
4. [Gaussian Naive Bayes](#4-gaussian-naive-bayes) — Continuous features
5. [Multinomial Naive Bayes](#5-multinomial-naive-bayes) — Word counts
6. [Bernoulli Naive Bayes](#6-bernoulli-naive-bayes) — Binary features
7. [Laplace Smoothing](#7-laplace-smoothing) — Handling zero probabilities
8. [Text as Features](#8-text-as-features) — Bag of Words and TF-IDF
9. [By-Hand Example](#9-by-hand-example) — Spam classification from scratch
10. [What to Look for in the Application Lab](#10-what-to-look-for-in-the-application-lab)

---

## 1. Bayes' Theorem

The foundation of everything in this lab:

```
                    P(B | A) * P(A)
P(A | B) = ─────────────────────────
                      P(B)
```

In the context of classification:

```
                    P(features | class) * P(class)
P(class | features) = ────────────────────────────────
                              P(features)
```

```
P(class | features):  POSTERIOR  — what we want to know
P(features | class):  LIKELIHOOD — how likely these features are given the class
P(class):             PRIOR     — how common the class is overall
P(features):          EVIDENCE  — probability of seeing these features (constant)
```

### Why It Works for Classification

We don't need P(features) because we're COMPARING classes:

```
Classify x as:  argmax_c  P(c | x)
              = argmax_c  P(x | c) * P(c) / P(x)
              = argmax_c  P(x | c) * P(c)          <-- P(x) is same for all c
```

We just need to find which class makes P(x|c) * P(c) largest.

---

## 2. Prior, Likelihood, Posterior

### Prior: P(class)

What you know BEFORE seeing any features:

```
Training data: 100 emails, 30 spam, 70 ham

P(spam) = 30/100 = 0.30
P(ham)  = 70/100 = 0.70

This is the "base rate" — without any other info,
a random email is more likely ham than spam.
```

### Likelihood: P(features | class)

How likely these particular features are, given the class:

```
P("free" appears | spam) = 25/30 = 0.833
P("free" appears | ham)  = 5/70  = 0.071

The word "free" is much more likely in spam emails.
```

### Posterior: P(class | features)

What you know AFTER seeing the features — this is the answer:

```
P(spam | "free") = P("free"|spam) * P(spam) / P("free")
                 = 0.833 * 0.30 / P("free")
                 = 0.250 / P("free")

P(ham | "free")  = P("free"|ham) * P(ham) / P("free")
                 = 0.071 * 0.70 / P("free")
                 = 0.050 / P("free")

Since 0.250 > 0.050: classify as SPAM

(If you want actual probabilities, normalize:
 P(spam|"free") = 0.250 / (0.250 + 0.050) = 0.833
 P(ham|"free")  = 0.050 / (0.250 + 0.050) = 0.167)
```

> **Key Intuition:** Bayes' theorem lets you FLIP the conditional.
> You can't directly observe P(spam | features). But you CAN count
> P(features | spam) from training data. Bayes flips it for you.

---

## 3. The Naive Assumption

The "naive" part: assume all features are **conditionally independent**
given the class.

```
P(x1, x2, ..., xd | c) = P(x1|c) * P(x2|c) * ... * P(xd|c)
                        = product_j P(xj | c)
```

### Why This Is "Wrong"

Words in text are NOT independent:

```
P("New" AND "York" | ham) != P("New"|ham) * P("York"|ham)

"New" and "York" are highly correlated.
Naive Bayes treats them as independent.
```

### Why It Works Anyway

```
1. We only need the RANKING to be correct (which class has highest score),
   not the actual probability values.

2. Even if individual probabilities are wrong, the RATIO between classes
   can still be correct enough:

   P(spam|x) / P(ham|x) > 1  -->  predict spam

   As long as this ratio is on the correct side of 1, the classification
   is correct, even if the magnitude is wrong.

3. In high dimensions (many features), overestimates and underestimates
   of individual P(xj|c) tend to cancel out.

4. It's a bias-variance tradeoff: the naive assumption adds BIAS but
   dramatically reduces VARIANCE (fewer parameters to estimate).
```

```
  Without naive assumption (full joint):
    Parameters to estimate: O(d^n)   -- exponential in features
    For 1000 words:  2^1000 entries   -- impossible

  With naive assumption:
    Parameters to estimate: O(d * c)  -- linear in features
    For 1000 words:  2 * 1000 = 2000 values  -- trivial
```

> **Key Intuition:** Naive Bayes is biased (independence assumption is wrong)
> but has very low variance (few parameters). For small/medium datasets,
> this tradeoff is often favorable. It's also extremely fast.

---

## 4. Gaussian Naive Bayes

For **continuous** features, assume each feature follows a Gaussian
(normal) distribution within each class:

```
P(xj | c) = (1 / sqrt(2*pi*sigma_jc^2)) * exp(-(xj - mu_jc)^2 / (2*sigma_jc^2))

where:
  mu_jc    = mean of feature j for class c     (from training data)
  sigma_jc = std dev of feature j for class c  (from training data)
```

```
  P(xj|c)
    |       Class 0           Class 1
    |        /\                 /\
    |       /  \               /  \
    |      /    \             /    \
    |     /      \           /      \
    |    /        \         /        \
    +--/----------\-------/----------\-- xj
      mu_0               mu_1

  For a new point xj:
    If P(xj|c=0) > P(xj|c=1), feature j "votes" for class 0.
```

### Training

Just compute mean and variance per feature per class:

```
For class c and feature j:
  mu_jc = mean of xj values where y = c
  sigma_jc^2 = variance of xj values where y = c

That's it. No optimization. No iteration.
This is why Naive Bayes training is O(N*d).
```

---

## 5. Multinomial Naive Bayes

For **count** features — how many times each word appears in a document.

```
P(xj | c) = (N_jc + alpha) / (N_c + alpha * d)

where:
  N_jc  = number of times feature j appears in class c documents
  N_c   = total count of ALL features in class c
  d     = vocabulary size (number of distinct features)
  alpha = smoothing parameter (Laplace smoothing, typically 1)
```

### Example

```
Training corpus:
  Spam docs: "free money free free win money"
  Ham docs:  "hi friend meeting schedule friend"

Word counts per class:
  Word      Spam count   Ham count
  free         3            0
  money        2            0
  win          1            0
  hi           0            1
  friend       0            2
  meeting      0            1
  schedule     0            1

  N_spam = 6 (total words in spam)
  N_ham  = 5 (total words in ham)
  d = 7 (vocabulary size)

P("free"|spam) = (3+1)/(6+7) = 4/13 = 0.308
P("free"|ham)  = (0+1)/(5+7) = 1/12 = 0.083
```

---

## 6. Bernoulli Naive Bayes

For **binary** features — word present (1) or absent (0).

```
P(xj | c) = P(j|c)^xj * (1 - P(j|c))^(1-xj)

If xj = 1 (word present):   P(xj|c) = P(j|c)
If xj = 0 (word absent):    P(xj|c) = 1 - P(j|c)
```

The key difference from Multinomial: **absence of a word is evidence too**.

```
Multinomial NB:  Only considers words that ARE present
Bernoulli NB:    Considers words that are present AND absent

Example: If "free" never appears in ham:
  Multinomial: ignores "free" when classifying a doc without "free"
  Bernoulli:   "free" is absent -> P(free=0|ham) is HIGH -> evidence for ham
```

> **Key Intuition:** Use Multinomial for word counts (long documents).
> Use Bernoulli for presence/absence (short documents, binary features).

---

## 7. Laplace Smoothing

### The Problem

If a word never appears in a class during training:

```
P("urgent" | ham) = 0/70 = 0

Then for ANY email containing "urgent":
P(ham | email) = P(w1|ham) * P(w2|ham) * ... * P("urgent"|ham) * ... * P(ham)
               = ... * 0 * ...
               = 0

One unseen word ZEROS OUT the entire probability!
The email can NEVER be classified as ham, regardless of other words.
```

### The Fix: Add-1 (Laplace) Smoothing

Add a fake count of alpha (usually 1) to every feature count:

```
P(xj | c) = (count(xj, c) + alpha) / (count(c) + alpha * |V|)

where:
  alpha = smoothing parameter (1 for Laplace, can be < 1 for Lidstone)
  |V|   = vocabulary size

Without smoothing: P("urgent"|ham) = 0/70 = 0.000
With smoothing:    P("urgent"|ham) = (0+1)/(70+|V|)  > 0
```

```
  alpha = 0:    No smoothing (MLE). Zero probabilities possible.
  alpha = 1:    Laplace smoothing. Classic choice.
  alpha = 0.5:  Jeffreys prior. Sometimes better.
  alpha -> inf: All probabilities become 1/|V| (uniform, ignores data).
```

> **Key Intuition:** Smoothing says "even if I haven't seen this word
> in this class, it's not IMPOSSIBLE — just rare." It's a form of
> regularization that prevents overconfident zeros.

---

## 8. Text as Features

### Bag of Words (BoW)

Represent a document as a vector of word counts, ignoring order:

```
Vocabulary: [apple, banana, cat, dog, eat]

Document: "the cat eat the apple and the cat eat banana"

BoW vector: [1, 1, 2, 0, 2]
             apple=1, banana=1, cat=2, dog=0, eat=2

Note: "the" and "and" are typically removed (stop words).
Word ORDER is lost: "dog eat cat" = "cat eat dog"
```

### Document-Term Matrix

```
Each row = a document, each column = a word from vocabulary

              apple  banana  cat  dog  eat  free  money
Doc 1 (ham):    1      1      2    0    2    0      0
Doc 2 (spam):   0      0      0    0    0    3      2
Doc 3 (ham):    0      1      1    1    1    0      0
Doc 4 (spam):   0      0      0    0    0    1      1
```

### TF-IDF: Term Frequency - Inverse Document Frequency

Raw word counts have a problem: common words ("the", "is") dominate.
TF-IDF downweights words that appear in many documents.

```
TF-IDF(t, d) = TF(t, d) * IDF(t)

TF(t, d) = count of term t in document d
           (or normalized: count / total words in d)

IDF(t) = log(N / df(t))

where:
  N    = total number of documents
  df(t) = number of documents containing term t
```

### IDF Intuition

```
Word appears in ALL docs:   IDF = log(N/N) = log(1) = 0     (useless word)
Word appears in 1 doc:      IDF = log(N/1) = log(N)          (rare, informative)

Example with N=1000 documents:
  "the":    df=1000  IDF = log(1000/1000) = 0.00   (appears everywhere)
  "python": df=100   IDF = log(1000/100)  = 2.30   (somewhat rare)
  "xgboost":df=5     IDF = log(1000/5)    = 5.30   (very rare, discriminative)
```

```
  IDF
    |
  7 |.
    | .
  5 |  .
    |   .
  3 |    ..
    |      ...
  1 |         .......
  0 +----+----+----+--- df/N
    0   0.25  0.5   1.0

  Words appearing in fewer documents get HIGHER IDF.
```

### Full TF-IDF Example

```
Corpus: 4 documents
  Doc1: "cat cat dog"
  Doc2: "cat bird"
  Doc3: "dog dog dog"
  Doc4: "bird bird cat"

TF (raw counts):
         cat  dog  bird
  Doc1:   2    1    0
  Doc2:   1    0    1
  Doc3:   0    3    0
  Doc4:   1    0    2

IDF:
  cat:  log(4/3) = 0.288   (appears in 3 of 4 docs)
  dog:  log(4/2) = 0.693   (appears in 2 of 4 docs)
  bird: log(4/2) = 0.693   (appears in 2 of 4 docs)

TF-IDF = TF * IDF:
          cat     dog     bird
  Doc1:  0.576   0.693   0.000
  Doc2:  0.288   0.000   0.693
  Doc3:  0.000   2.079   0.000
  Doc4:  0.288   0.000   1.386

  "cat" gets lower TF-IDF because it's common (low IDF).
  "dog" in Doc3 gets high TF-IDF: high count AND not universal.
```

---

## 9. By-Hand Example

**Classify the email "free money" as spam or ham.**

### Training Data

```
Email #   Text                    Class
1         "free free free"        spam
2         "free money win"        spam
3         "money money prize"     spam
4         "hi friend"             ham
5         "meeting schedule"      ham
6         "hi friend meeting"     ham
7         "free lunch friend"     ham
```

### Step 1: Compute Priors

```
P(spam) = 3/7 = 0.429
P(ham)  = 4/7 = 0.571
```

### Step 2: Build Vocabulary and Count

```
Vocabulary: {free, money, win, prize, hi, friend, meeting, schedule, lunch}
|V| = 9

Word counts per class:
              spam   ham
  free          4      1     (3+1+0 in spam; 0+0+0+1 in ham)
  money         3      0
  win           1      0
  prize         1      0
  hi            0      2
  friend        0      3
  meeting       0      2
  schedule      0      1
  lunch         0      1
  ─────────────────────
  TOTAL         9     10
```

### Step 3: Compute Likelihoods (with Laplace smoothing, alpha=1)

```
P(word | class) = (count(word, class) + 1) / (total_words_in_class + |V|)

P(free  | spam) = (4+1) / (9+9) = 5/18 = 0.278
P(money | spam) = (3+1) / (9+9) = 4/18 = 0.222

P(free  | ham)  = (1+1) / (10+9) = 2/19 = 0.105
P(money | ham)  = (0+1) / (10+9) = 1/19 = 0.053
```

### Step 4: Classify "free money"

```
P(spam | "free money") proportional to:
  P(spam) * P(free|spam) * P(money|spam)
  = 0.429 * 0.278 * 0.222
  = 0.429 * 0.0617
  = 0.0265

P(ham | "free money") proportional to:
  P(ham) * P(free|ham) * P(money|ham)
  = 0.571 * 0.105 * 0.053
  = 0.571 * 0.00557
  = 0.00318
```

### Step 5: Normalize to Get Probabilities

```
P(spam | "free money") = 0.0265 / (0.0265 + 0.00318) = 0.893
P(ham  | "free money") = 0.00318 / (0.0265 + 0.00318) = 0.107

Classification: SPAM (89.3% confidence)
```

### Step-by-Step Summary

```
  "free money"
       |
       v
  +-----------+     +-----------+
  | P(spam)   |     | P(ham)    |
  | = 0.429   |     | = 0.571   |
  +-----------+     +-----------+
       |                  |
       v                  v
  P(free|spam)=0.278  P(free|ham)=0.105
  x                   x
  P(money|spam)=0.222 P(money|ham)=0.053
       |                  |
       v                  v
    0.0265              0.00318
       |                  |
       v                  v
    89.3%               10.7%
       |
       v
     SPAM
```

> **Key Intuition:** The word "money" was the decisive factor. It appeared
> 3 times in spam, 0 times in ham. Even with Laplace smoothing,
> P(money|spam) is 4x higher than P(money|ham). Combined with "free"
> also favoring spam, the evidence overwhelmingly points to spam.

---

## 10. What to Look for in the Application Lab

When you move to the coding lab:

1. **Pipeline:** Use `Pipeline([('tfidf', TfidfVectorizer()), ('nb', MultinomialNB())])` — keeps preprocessing and model together
2. **CountVectorizer vs TfidfVectorizer:** Try both, compare. TF-IDF usually wins for text
3. **Alpha tuning:** Try `alpha` values from 0.01 to 10 — small alpha can overfit, large alpha smooths too much
4. **Stop words:** Compare `stop_words='english'` vs None — removing stop words often helps
5. **N-grams:** Try `ngram_range=(1,2)` to capture bigrams like "New York" — helps overcome the independence assumption
6. **Confusion matrix:** Look at false positives (ham marked as spam) vs false negatives (spam in inbox) — different costs
7. **Speed:** Time NB vs SVM vs Random Forest on text. NB will be orders of magnitude faster
8. **Feature inspection:** Look at the most informative features per class — `model.feature_log_prob_` shows which words are most "spammy" or "hammy"
9. **Gaussian NB comparison:** Try `GaussianNB()` on the same data with TF-IDF features — it usually performs worse than MultinomialNB for text
