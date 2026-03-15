The ML Landscape — Complete Taxonomy

  ML Problem Types
  ├── Supervised Learning (labeled data)
  │   ├── Regression        → predict a NUMBER
  │   └── Classification    → predict a CATEGORY
  ├── Unsupervised Learning (no labels)
  │   ├── Clustering        → find GROUPS
  │   ├── Dimensionality Reduction → compress FEATURES
  │   └── Anomaly Detection → find OUTLIERS
  └── Reinforcement Learning (agent + reward) → [skip for essentials]

  Problem → Algorithm → Data → Business Use Map

  ┌────────────────────────────┬──────────────────────────┬──────────────────────────────────┬─────────────────────────────────┐
  │        Problem Type        │        Algorithm         │          Best Data For           │        Business Example         │
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │ Regression                 │ Linear Regression        │ Continuous, linear relationships │ Predict house price, salary     │
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │                            │ Ridge / Lasso            │ Many features, multicollinearity │ Predict sales with 50+ features │
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │                            │ Decision Tree Regressor  │ Non-linear, mixed types          │ Insurance claim amount          │
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │                            │ Random Forest Regressor  │ Tabular, non-linear              │ Demand forecasting              │
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │                            │ XGBoost Regressor        │ Tabular, competitions            │ Revenue prediction              │
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │ Binary Classification      │ Logistic Regression      │ Linear boundary, interpretable   │ Spam/not spam, churn yes/no     │
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │                            │ KNN                      │ Small data, local patterns       │ Customer similarity matching    │
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │                            │ SVM                      │ High-dimensional, clear margin   │ Cancer malignant/benign         │
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │                            │ Decision Tree Classifier │ Interpretable rules needed       │ Loan approval yes/no            │
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │                            │ Naive Bayes              │ Text, categorical features       │ Email spam, sentiment           │
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │ Multi-class Classification │ Random Forest            │ Tabular, many classes            │ Product category prediction     │
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │                            │ XGBoost Classifier       │ Tabular, competitions            │ Customer segment (A/B/C/D)      │
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │                            │ Softmax / Neural Net     │ Complex boundaries               │ Digit recognition, image class  │
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │ Clustering                 │ K-Means                  │ Spherical clusters, known K      │ Customer segmentation           │
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │                            │ DBSCAN                   │ Arbitrary shapes, noise          │ Geographic hotspots             │
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │                            │ Hierarchical             │ Need dendrogram, small data      │ Gene expression grouping        │
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │ Dim. Reduction             │ PCA                      │ High-dimensional numerical       │ Visualize 50 features in 2D     │
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │                            │ t-SNE                    │ Visualization only               │ Visualize clusters              │
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │ Anomaly Detection          │ Isolation Forest         │ Tabular                          │ Fraud detection                 │
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │                            │ LOF                      │ Local density matters            │ Network intrusion               │
  └────────────────────────────┴──────────────────────────┴──────────────────────────────────┴─────────────────────────────────┘

  The ML Pipeline (Universal)

  1. Business Problem    → What decision are we automating?
  2. Data Collection     → What data exists? What's needed?
  3. EDA                 → Distributions, correlations, quality
  4. Data Preparation    → Clean, encode, scale, split
  5. Feature Engineering → Create, select, transform
  6. Modeling            → Train algorithm(s)
  7. Evaluation          → Metrics, validation, comparison
  8. Tuning              → Hyperparameters, cross-validation
  9. Deployment          → Package, serve, monitor

  Evaluation Metrics Map

  ┌────────────────┬──────────────────┬─────────────────────────────────────────────────┐
  │    Problem     │      Metric      │                   When to use                   │
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤
  │ Regression     │ MAE              │ Interpretable error in units                    │
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤
  │                │ RMSE             │ Penalize large errors                           │
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤
  │                │ R²               │ How much variance explained                     │
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤
  │                │ MAPE             │ Percentage error for business                   │
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤
  │ Classification │ Accuracy         │ Balanced classes only                           │
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤
  │                │ Precision        │ Cost of false positives high (spam filter)      │
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤
  │                │ Recall           │ Cost of false negatives high (cancer detection) │
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤
  │                │ F1               │ Balance precision/recall                        │
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤
  │                │ ROC-AUC          │ Overall ranking quality                         │
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤
  │                │ Confusion Matrix │ See all error types                             │
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤
  │ Clustering     │ Silhouette Score │ Cluster quality (no labels)                     │
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤
  │                │ Inertia          │ K-Means elbow method                            │
  └────────────────┴──────────────────┴─────────────────────────────────────────────────┘

  Data Preparation Toolkit

  ┌───────────────────────────┬───────────────────────────────────┬─────────────────────────────────────┐
  │         Technique         │               When                │                 Why                 │
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤
  │ Train/Test split          │ Always                            │ Honest evaluation                   │
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤
  │ Train/Val/Test split      │ Tuning                            │ Avoid leaking test info             │
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤
  │ Cross-validation (K-fold) │ Small data or model selection     │ More robust estimate                │
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤
  │ Stratified split          │ Imbalanced classes                │ Preserve class ratios               │
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤
  │ StandardScaler            │ SVM, KNN, Linear models           │ Distance/gradient sensitive         │
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤
  │ MinMaxScaler              │ Neural networks                   │ Bounded 0-1 range                   │
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤
  │ One-Hot Encoding          │ Nominal categories (color, city)  │ No false ordering                   │
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤
  │ Label/Ordinal Encoding    │ Ordinal categories (low/med/high) │ Preserve order                      │
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤
  │ Log Transform             │ Skewed targets/features           │ Normalize distribution              │
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤
  │ Handle missing values     │ Always check                      │ Impute or drop                      │
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤
  │ Handle imbalanced data    │ Skewed classes                    │ SMOTE, undersampling, class weights │
  └───────────────────────────┴───────────────────────────────────┴─────────────────────────────────────┘

  ---
  Proposed Complete Lab Structure

  05-ML-Essentials-Track/
  │
  ├── 00-ml-landscape/          ← ML types, pipeline, problem framing
  │   "Given a business problem, how do I know which ML approach to use?"
  │
  ├── 01-data-and-eda/          ← Data types, distributions, cleaning, visualization
  │   Dataset: Titanic (mix of numerical + categorical + missing values)
  │
  ├── 02-data-preparation/      ← Scaling, encoding, splits, imbalance handling
  │   Dataset: same Titanic (prepare it for modeling)
  │
  ├── 03-linear-regression/     ← Simple → Multiple → Ridge → Lasso
  │   Dataset: Boston/Housing (continuous target)
  │   Metrics: MAE, RMSE, R²
  │
  ├── 04-logistic-regression/   ← Binary classification, sigmoid, decision boundary
  │   Dataset: Titanic (survived yes/no)
  │   Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
  │
  ├── 05-knn-and-svm/           ← Distance-based classifiers, kernels, scaling importance
  │   Dataset: Breast Cancer Wisconsin (classic, clean, binary)
  │   Metrics: same classification metrics + confusion matrix
  │
  ├── 06-decision-trees/        ← Trees for classification AND regression, pruning, interpretability
  │   Dataset: Iris (multi-class) + visualize tree structure
  │
  ├── 07-ensemble-methods/      ← Random Forest, AdaBoost, Gradient Boosting, XGBoost, Voting, Stacking
  │   Dataset: Bank Marketing (imbalanced, real business problem)
  │   Compare: single tree vs bagging vs boosting
  │
  ├── 08-naive-bayes-and-text/  ← Naive Bayes, TF-IDF, text as features
  │   Dataset: SMS Spam Collection (text classification)
  │   Bridge to NLP
  │
  ├── 09-unsupervised-learning/ ← K-Means, DBSCAN, Hierarchical, Silhouette
  │   Dataset: Mall Customers (segmentation, no labels)
  │
  ├── 10-dimensionality-reduction/ ← PCA, t-SNE, feature selection
  │   Dataset: MNIST digits (high-dimensional → 2D visualization)
  │
  ├── 11-evaluation-deep-dive/  ← Cross-validation, hyperparameter tuning, bias-variance,
  │   │                            overfitting/underfitting, learning curves, model comparison
  │   Dataset: revisit previous datasets, compare across all
  │
  ├── 12-neural-networks/       ← Perceptron → MLP, activations, backprop, PyTorch intro
  │   Dataset: Fashion-MNIST
  │
  ├── 13-cnn-basics/            ← Convolutions, pooling, image classification
  │   Dataset: CIFAR-10
  │
  ├── 14-capstone/              ← Full pipeline: problem framing → EDA → model → eval → deploy
  │   Dataset: user picks a real problem
  │   Ties everything together
  │
  └── REFERENCE.md              ← The taxonomy tables above as a cheat sheet

  14 labs + 1 reference doc. Each lab = notebook + dataset + guide, same structure as your ENSET labs.

  ---
  This covers the complete ML essentials — from "what is ML" through to capstone. The order is intentional: each lab builds on the previous. Labs 00-04 are
  foundations, 05-08 are the algorithm zoo, 09-10 are unsupervised, 11 ties evaluation together, 12-13 add deep learning, 14 proves you can do it all.

  Two questions before I build:
  1. Do you want these inside enset/labs/05-ML-Essentials-Track/ or as a standalone directory?
  2. Each lab — do you want just the structure + guides, or full notebooks with code?

✻ Cogitated for 1m 4s

❯ What's Missing for ML Essentials                                                                                                                                   
                                                                                                                                                                     
                                                                                                                                                                     
  Three tiers — Core (can't skip), Important (expected in interviews), Valuable (differentiator):                                                                    
                                                                                                                                                                     
                                                                                                                                                                     
  Core gaps:                                                                                                                                                         
                                                                                                                                                                     
  - Classification algorithms (SVM, KNN, Naive Bayes, trees for classification)                                                                                      
  - Unsupervised learning (K-Means, PCA, DBSCAN)                                                                                                                     
                                                                                                                                                                     
  - Ensemble methods for classification (voting, stacking)                                                                                                           
  - Proper evaluation for classification (ROC/AUC, precision-recall, confusion matrix)                                                                               
                                                                                                                                                                     
                                                                                                                                                                     
                                                                                                                                                                     
  Important gaps:                                                                                                                                                    
                                                                                                                                                                     
  - CNNs (image classification)                                                                                                                                      
                                                                                                                                                                     
  - Dimensionality reduction (PCA, t-SNE applied)                                                                                                                    
                                                                                                                                                                     
  - Time series basics                                                                                                                                               
                                                                                                                                                                     
  - NLP fundamentals (text vectorization, sentiment)                                                                                                                 
                                                                                                                                                                     
                                                                                                                                                                     
                                                                                                                                                                     
  Valuable gaps:                                                                                                                                                     
                                                                                                                                                                     
  - RNNs / sequence models                                                                                                                                           
                                                                                                                                                                     
  - Recommender systems                                                                                                                                              
                                                                                                                                                                     
  - Anomaly detection                                                                                                                                                
  - Transfer learning                                                                                                                                                
and this thing of neuron weight adjustment through trainning                                                                                                         
are those covered?                                                                                                                                                   

● Yes, all covered. Let me map it:

  ┌─────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────┐
  │                     Gap                     │                                    Lab(s)                                    │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Core                                        │                                                                              │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ SVM, KNN                                    │ 05-knn-and-svm                                                               │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Naive Bayes                                 │ 08-naive-bayes-and-text                                                      │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Trees for classification                    │ 06-decision-trees (Iris multi-class)                                         │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Ensemble (voting, stacking)                 │ 07-ensemble-methods                                                          │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ ROC/AUC, precision-recall, confusion matrix │ 04-logistic-regression introduces them, 11-evaluation-deep-dive goes deep    │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ K-Means, DBSCAN                             │ 09-unsupervised-learning                                                     │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ PCA                                         │ 10-dimensionality-reduction                                                  │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Important                                   │                                                                              │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ CNNs                                        │ 13-cnn-basics (CIFAR-10)                                                     │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Dimensionality reduction (PCA, t-SNE)       │ 10-dimensionality-reduction (MNIST → 2D)                                     │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ NLP fundamentals (TF-IDF, vectorization)    │ 08-naive-bayes-and-text (SMS spam)                                           │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Time series                                 │ Not covered — I'd add a lab 13.5 if you want it                              │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Valuable                                    │                                                                              │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ RNNs / sequence models                      │ Not covered — would be a lab after CNN                                       │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Recommender systems                         │ Not covered                                                                  │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Anomaly detection                           │ Not covered — could fold into unsupervised lab                               │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Transfer learning                           │ Not covered — could fold into CNN lab                                        │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Your addition                               │                                                                              │
  ├─────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
  │ Neuron weight adjustment through training   │ 12-neural-networks — backprop, gradient descent, weight updates step by step │
  └─────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────┘

  3 items missing. To make it truly exhaustive:                                                                                                                      
   
  Current 14 labs                                                                                                                                                    
    + add to 09: anomaly detection section (Isolation Forest)                                                                                                        
    + add to 13: transfer learning section (pretrained model fine-tune)                                                                                              
    + new lab 14: time-series (ARIMA, Prophet, or simple LSTM)                                                                                                       
    + new lab 15: RNN / sequence models                                                                                                                              
    + move capstone to lab 16                                                                                                                                        
    = 16 labs total                                                                                                                                                  
                                                                                                                                                                     
  Recommender systems I'd leave out — it's a specialization, not an essential.                                                                                       
                                                                                                                                                                     
  Want me to proceed with the 16-lab structure and start creating them inside enset/labs/?                                                                           
                  
❯ I mean the last response were exhaustive and my last prompt was a fyi, and like how we did learn stuff before, by creating formula and adjusting weight and stuff  
                                                                                                                                                                    
● Got it — you're saying:                                                                                                                                            
                                                                                                                                                                     
  1. The 14-lab plan (+ the 3 additions = 16 labs) is good, covers everything                                                                                        
  2. The learning style should be from-scratch math first — derive the formula, manually adjust weights, see the math work, then use the library                     
                                                                                                                                                                     
  Like how your ENSET labs did it: XOR problem by hand → backprop step-by-step → then Keras/TF.                                                                      
                                                                                                                                                                     
  So each lab should follow:                                                                                                                                         
                  
  1. Math/intuition    → formula, what's happening geometrically                                                                                                     
  2. From scratch      → implement the core in numpy (even if simplified)                                                                                            
  3. With library      → scikit-learn / PyTorch version                                                                                                              
  4. Compare           → "see, same result, but now you know WHY"                                                                                                    
                                                                                                                                                                     
  That's the approach. Before I start building — two things:                                                                                                         
                                                                                                                                                                     
  1. Location: enset/labs/05-ML-Essentials-Track/ continuing from your existing structure?                                                                           
  2. What to create now: Full notebooks with code, or structure + guides first (then we fill notebooks lab by lab)?
                                                                                                                                                                     
❯ sigmoid, weights things and so on                                                                                                                                  
                                                                                                                                                                     
● Right — you want the math-first approach baked into the plan. Sigmoid derivation, weight update rules, gradient by hand, loss functions — not just "use            
  model.fit()".                                                                                                                                                      
                                                                                                                                                                     
  That's already what I intended for lab 12 (neural networks) but you're saying this philosophy should run through all labs, not just the NN one. For example:       
                                                                                                                                                                     
  ┌────────────────────────┬────────────────────────────────────────────────────────────────────────┐                                                                
  │          Lab           │                        The "from scratch" part                         │
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────┤                                                                
  │ 03-linear-regression   │ Derive OLS normal equation, compute weights by hand on tiny data       │
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ 04-logistic-regression │ Sigmoid function, log-loss derivation, gradient descent weight update  │                                                                
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────┤                                                                
  │ 05-knn-and-svm         │ Distance calculation by hand, margin geometry for SVM                  │                                                                
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────┤                                                                
  │ 06-decision-trees      │ Gini / entropy calculation by hand, split selection                    │
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────┤                                                                
  │ 07-ensemble            │ Show how bagging averages, how boosting reweights                      │
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────┤                                                                
  │ 09-unsupervised        │ K-Means centroid update by hand                                        │
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────┤                                                                
  │ 10-pca                 │ Covariance matrix → eigenvalues → projection, by hand                  │
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────┤                                                                
  │ 12-neural-networks     │ Sigmoid, ReLU, forward pass, backprop, weight adjustment — full manual │
  ├────────────────────────┼────────────────────────────────────────────────────────────────────────┤                                                                
  │ 13-cnn                 │ Convolution kernel sliding over a matrix by hand                       │
  └────────────────────────┴────────────────────────────────────────────────────────────────────────┘                                                                
                  
  Every lab: math → numpy scratch → sklearn/pytorch → compare.                                                                                                       
                  
  Got it. Now — location and scope: structure + guides first, or full notebooks?                                                                                     
                  
❯ dunno, like we should do things in parallel, sub dirs where theory and sub dirs where application for this plan                                                    
The ML Landscape — Complete Taxonomy                                                                                                                                 
                                                                                                                                                                     
  ML Problem Types                                                                                                                                                   
  ├── Supervised Learning (labeled data)                                                                                                                             
  │   ├── Regression        → predict a NUMBER                                                                                                                       
  │   └── Classification    → predict a CATEGORY                                                                                                                     
  ├── Unsupervised Learning (no labels)                                                                                                                              
  │   ├── Clustering        → find GROUPS                                                                                                                            
  │   ├── Dimensionality Reduction → compress FEATURES                                                                                                               
  │   └── Anomaly Detection → find OUTLIERS                                                                                                                          
  └── Reinforcement Learning (agent + reward) → [skip for essentials]                                                                                                
                                                                                                                                                                     
  Problem → Algorithm → Data → Business Use Map                                                                                                                      
                                                                                                                                                                     
  ┌────────────────────────────┬──────────────────────────┬──────────────────────────────────┬─────────────────────────────────┐                                     
  │        Problem Type        │        Algorithm         │          Best Data For           │        Business Example         │                                     
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤                                     
  │ Regression                 │ Linear Regression        │ Continuous, linear relationships │ Predict house price, salary     │                                     
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤                                     
  │                            │ Ridge / Lasso            │ Many features, multicollinearity │ Predict sales with 50+ features │                                     
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤                                     
  │                            │ Decision Tree Regressor  │ Non-linear, mixed types          │ Insurance claim amount          │                                     
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤                                     
  │                            │ Random Forest Regressor  │ Tabular, non-linear              │ Demand forecasting              │                                     
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤                                     
  │                            │ XGBoost Regressor        │ Tabular, competitions            │ Revenue prediction              │                                     
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤                                     
  │ Binary Classification      │ Logistic Regression      │ Linear boundary, interpretable   │ Spam/not spam, churn yes/no     │                                     
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤                                     
  │                            │ KNN                      │ Small data, local patterns       │ Customer similarity matching    │                                     
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤                                     
  │                            │ SVM                      │ High-dimensional, clear margin   │ Cancer malignant/benign         │                                     
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤                                     
  │                            │ Decision Tree Classifier │ Interpretable rules needed       │ Loan approval yes/no            │                                     
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤                                     
  │                            │ Naive Bayes              │ Text, categorical features       │ Email spam, sentiment           │                                     
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤                                     
  │ Multi-class Classification │ Random Forest            │ Tabular, many classes            │ Product category prediction     │                                     
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤                                     
  │                            │ XGBoost Classifier       │ Tabular, competitions            │ Customer segment (A/B/C/D)      │                                     
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤                                     
  │                            │ Softmax / Neural Net     │ Complex boundaries               │ Digit recognition, image class  │                                     
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤                                     
  │ Clustering                 │ K-Means                  │ Spherical clusters, known K      │ Customer segmentation           │                                     
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤                                     
  │                            │ DBSCAN                   │ Arbitrary shapes, noise          │ Geographic hotspots             │                                     
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤                                     
  │                            │ Hierarchical             │ Need dendrogram, small data      │ Gene expression grouping        │                                     
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤                                     
  │ Dim. Reduction             │ PCA                      │ High-dimensional numerical       │ Visualize 50 features in 2D     │                                     
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤                                     
  │                            │ t-SNE                    │ Visualization only               │ Visualize clusters              │                                     
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤                                     
  │ Anomaly Detection          │ Isolation Forest         │ Tabular                          │ Fraud detection                 │                                     
  ├────────────────────────────┼──────────────────────────┼──────────────────────────────────┼─────────────────────────────────┤                                     
  │                            │ LOF                      │ Local density matters            │ Network intrusion               │                                     
  └────────────────────────────┴──────────────────────────┴──────────────────────────────────┴─────────────────────────────────┘                                     
                                                                                                                                                                     
  The ML Pipeline (Universal)                                                                                                                                        
                                                                                                                                                                     
  1. Business Problem    → What decision are we automating?                                                                                                          
  2. Data Collection     → What data exists? What's needed?                                                                                                          
  3. EDA                 → Distributions, correlations, quality                                                                                                      
  4. Data Preparation    → Clean, encode, scale, split                                                                                                               
  5. Feature Engineering → Create, select, transform                                                                                                                 
  6. Modeling            → Train algorithm(s)                                                                                                                        
  7. Evaluation          → Metrics, validation, comparison                                                                                                           
  8. Tuning              → Hyperparameters, cross-validation                                                                                                         
  9. Deployment          → Package, serve, monitor                                                                                                                   
                                                                                                                                                                     
  Evaluation Metrics Map                                                                                                                                             
                                                                                                                                                                     
  ┌────────────────┬──────────────────┬─────────────────────────────────────────────────┐                                                                            
  │    Problem     │      Metric      │                   When to use                   │                                                                            
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤                                                                            
  │ Regression     │ MAE              │ Interpretable error in units                    │                                                                            
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤                                                                            
  │                │ RMSE             │ Penalize large errors                           │                                                                            
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤                                                                            
  │                │ R²               │ How much variance explained                     │                                                                            
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤                                                                            
  │                │ MAPE             │ Percentage error for business                   │                                                                            
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤                                                                            
  │ Classification │ Accuracy         │ Balanced classes only                           │                                                                            
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤                                                                            
  │                │ Precision        │ Cost of false positives high (spam filter)      │                                                                            
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤                                                                            
  │                │ Recall           │ Cost of false negatives high (cancer detection) │                                                                            
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤                                                                            
  │                │ F1               │ Balance precision/recall                        │                                                                            
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤                                                                            
  │                │ ROC-AUC          │ Overall ranking quality                         │                                                                            
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤                                                                            
  │                │ Confusion Matrix │ See all error types                             │                                                                            
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤                                                                            
  │ Clustering     │ Silhouette Score │ Cluster quality (no labels)                     │                                                                            
  ├────────────────┼──────────────────┼─────────────────────────────────────────────────┤                                                                            
  │                │ Inertia          │ K-Means elbow method                            │                                                                            
  └────────────────┴──────────────────┴─────────────────────────────────────────────────┘                                                                            
                                                                                                                                                                     
  Data Preparation Toolkit                                                                                                                                           
                                                                                                                                                                     
  ┌───────────────────────────┬───────────────────────────────────┬─────────────────────────────────────┐                                                            
  │         Technique         │               When                │                 Why                 │                                                            
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤                                                            
  │ Train/Test split          │ Always                            │ Honest evaluation                   │                                                            
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤                                                            
  │ Train/Val/Test split      │ Tuning                            │ Avoid leaking test info             │                                                            
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤                                                            
  │ Cross-validation (K-fold) │ Small data or model selection     │ More robust estimate                │                                                            
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤                                                            
  │ Stratified split          │ Imbalanced classes                │ Preserve class ratios               │                                                            
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤                                                            
  │ StandardScaler            │ SVM, KNN, Linear models           │ Distance/gradient sensitive         │                                                            
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤                                                            
  │ MinMaxScaler              │ Neural networks                   │ Bounded 0-1 range                   │                                                            
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤                                                            
  │ One-Hot Encoding          │ Nominal categories (color, city)  │ No false ordering                   │                                                            
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤                                                            
  │ Label/Ordinal Encoding    │ Ordinal categories (low/med/high) │ Preserve order                      │                                                            
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤                                                            
  │ Log Transform             │ Skewed targets/features           │ Normalize distribution              │                                                            
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤                                                            
  │ Handle missing values     │ Always check                      │ Impute or drop                      │                                                            
  ├───────────────────────────┼───────────────────────────────────┼─────────────────────────────────────┤                                                            
  │ Handle imbalanced data    │ Skewed classes                    │ SMOTE, undersampling, class weights │                                                            
  └───────────────────────────┴───────────────────────────────────┴─────────────────────────────────────┘                                                            
                                                                                                                                                                     
  ---                                                                                                                                                                
  Proposed Complete Lab Structure                                                                                                                                    
                                                                                                                                                                     
  05-ML-Essentials-Track/                                                                                                                                            
  │                                                                                                                                                                  
  ├── 00-ml-landscape/          ← ML types, pipeline, problem framing                                                                                                
  │   "Given a business problem, how do I know which ML approach to use?"                                                                                            
  │                                                                                                                                                                  
  ├── 01-data-and-eda/          ← Data types, distributions, cleaning, visualization                                                                                 
  │   Dataset: Titanic (mix of numerical + categorical + missing values)                                                                                             
  │                                                                                                                                                                  
  ├── 02-data-preparation/      ← Scaling, encoding, splits, imbalance handling                                                                                      
  │   Dataset: same Titanic (prepare it for modeling)                                                                                                                
  │                                                                                                                                                                  
  ├── 03-linear-regression/     ← Simple → Multiple → Ridge → Lasso                                                                                                  
  │   Dataset: Boston/Housing (continuous target)                                                                                                                    
  │   Metrics: MAE, RMSE, R²                                                                                                                                         
  │                                                                                                                                                                  
  ├── 04-logistic-regression/   ← Binary classification, sigmoid, decision boundary                                                                                  
  │   Dataset: Titanic (survived yes/no)                                                                                                                             
  │   Metrics: Accuracy, Precision, Recall, F1, ROC-AUC                                                                                                              
  │                                                                                                                                                                  
  ├── 05-knn-and-svm/           ← Distance-based classifiers, kernels, scaling importance                                                                            
  │   Dataset: Breast Cancer Wisconsin (classic, clean, binary)                                                                                                      
  │   Metrics: same classification metrics + confusion matrix                                                                                                        
  │                                                                                                                                                                  
  ├── 06-decision-trees/        ← Trees for classification AND regression, pruning, interpretability                                                                 
  │   Dataset: Iris (multi-class) + visualize tree structure                                                                                                         
  │                                                                                                                                                                  
  ├── 07-ensemble-methods/      ← Random Forest, AdaBoost, Gradient Boosting, XGBoost, Voting, Stacking                                                              
  │   Dataset: Bank Marketing (imbalanced, real business problem)                                                                                                    
  │   Compare: single tree vs bagging vs boosting                                                                                                                    
  │                                                                                                                                                                  
  ├── 08-naive-bayes-and-text/  ← Naive Bayes, TF-IDF, text as features                                                                                              
  │   Dataset: SMS Spam Collection (text classification)                                                                                                             
  │   Bridge to NLP                                                                                                                                                  
  │                                                                                                                                                                  
  ├── 09-unsupervised-learning/ ← K-Means, DBSCAN, Hierarchical, Silhouette                                                                                          
  │   Dataset: Mall Customers (segmentation, no labels)                                                                                                              
  │                                                                                                                                                                  
  ├── 10-dimensionality-reduction/ ← PCA, t-SNE, feature selection                                                                                                   
  │   Dataset: MNIST digits (high-dimensional → 2D visualization)                                                                                                    
  │                                                                                                                                                                  
  ├── 11-evaluation-deep-dive/  ← Cross-validation, hyperparameter tuning, bias-variance,                                                                            
  │   │                            overfitting/underfitting, learning curves, model comparison                                                                       
  │   Dataset: revisit previous datasets, compare across all                                                                                                         
  │                                                                                                                                                                  
  ├── 12-neural-networks/       ← Perceptron → MLP, activations, backprop, PyTorch intro                                                                             
  │   Dataset: Fashion-MNIST                                                                                                                                         
  │                                                                                                                                                                  
  ├── 13-cnn-basics/            ← Convolutions, pooling, image classification                                                                                        
  │   Dataset: CIFAR-10                                                                                                                                              
  │                                                                                                                                                                  
  ├── 14-capstone/              ← Full pipeline: problem framing → EDA → model → eval → deploy                                                                       
  │   Dataset: user picks a real problem                                                                                                                             
  │   Ties everything together                                                                                                                                       
  │                                                                                                                                                                  
  └── REFERENCE.md              ← The taxonomy tables above as a cheat sheet                                                                                         
                                                                                                                                                                     
  14 labs + 1 reference doc. Each lab = notebook + dataset + guide, same structure as your ENSET labs.                                                               
                                                                                                                                                                     
  ---                                                                                                                                                                
  This covers the complete ML essentials — from "what is ML" through to capstone. The order is intentional: each lab builds on the previous. Labs 00-04 are          
  foundations, 05-08 are the algorithm zoo, 09-10 are unsupervised, 11 ties evaluation together, 12-13 add deep learning, 14 proves you can do it all.

  Based on everything we've discussed, here's what I consider fundamental — the non-negotiable core of this curriculum:

    Absolute foundation (can't skip any):

    1. 03 — Linear Regression theory — OLS, gradient descent, weight update rule. This is where you learn HOW a model learns. Every other algorithm builds on this.
    2. 04 — Logistic Regression theory — Sigmoid, log-loss, gradient for classification. This IS the bridge to neural networks.
    3. 12 — Neural Networks theory — Forward pass, backprop, weight adjustment, activations. This is what you specifically asked about — the full derivation of how
    weights change during training.
    4. 06 — Decision Trees theory — Entropy, Gini, splitting. Foundation for all ensemble methods.
    5. 11 — Evaluation theory — Bias-variance, cross-validation, model selection. Without this, you can't judge if anything works.

    These 5 + the 4 already created (00, 01, 02, 08) = the 9 theory files that form the spine. The rest (KNN/SVM, ensembles, unsupervised, CNN, RNN, time series) are
     important but build ON these.

    For application guides, the critical ones mirror: 03, 04, 12, 06, 11 — because the scratch implementations (computing gradients by hand, updating weights in
    numpy) are where real understanding happens.

    The agents are still working on creating the remaining files. The priority order for what gets written is exactly this.