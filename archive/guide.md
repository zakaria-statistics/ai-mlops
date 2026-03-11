┌────────────────────────┬────────────────────────────────────────────────────────┐                                                                                                                    
  │         Phase          │                        Content                         │                                                                                                                    
  ├────────────────────────┼────────────────────────────────────────────────────────┤                                                                                                                    
  │ 1. Problem Definition  │ Business framing, cost matrix, success metrics         │                                                                                                                    
  ├────────────────────────┼────────────────────────────────────────────────────────┤                                                                                                                    
  │ 2. Environment Setup   │ Dependencies, Docker, docker-compose for MLflow        │                                                                                                                    
  ├────────────────────────┼────────────────────────────────────────────────────────┤                                                                                                                    
  │ 3. Data Pipeline       │ DVC versioning, validation with Pandera, preprocessing │                                                                                                                    
  ├────────────────────────┼────────────────────────────────────────────────────────┤                                                                                                                    
  │ 4. Feature Engineering │ Feature building, feature registry pattern             │                                                                                                                    
  ├────────────────────────┼────────────────────────────────────────────────────────┤                                                                                                                    
  │ 5. Experimentation     │ MLflow tracking, experiment comparison                 │                                                                                                                    
  ├────────────────────────┼────────────────────────────────────────────────────────┤                                                                                                                    
  │ 6. Model Training      │ DVC stages, hyperparameter tuning                      │                                                                                                                    
  ├────────────────────────┼────────────────────────────────────────────────────────┤                                                                                                                    
  │ 7. Evaluation          │ Metrics, plots, quality gates                          │                                                                                                                    
  ├────────────────────────┼────────────────────────────────────────────────────────┤                                                                                                                    
  │ 8. Model Registry      │ MLflow registry, model cards                           │                                                                                                                    
  ├────────────────────────┼────────────────────────────────────────────────────────┤                                                                                                                    
  │ 9. Deployment          │ FastAPI serving, K8s manifests                         │                                                                                                                    
  ├────────────────────────┼────────────────────────────────────────────────────────┤                                                                                                                    
  │ 10. Monitoring         │ Drift detection (Evidently), Prometheus metrics        │                                                                                                                    
  ├────────────────────────┼────────────────────────────────────────────────────────┤                                                                                                                    
  │ 11. CI/CD Pipeline     │ GitHub Actions for lint/test, training, deployment     │                                                                                                                    
  └────────────────────────┴────────────────────────────────────────────────────────┘                                                                                                                    
  Key MLOps tools integrated:                                                                                                                                                                            
  - DVC - Data versioning and pipeline orchestration                                                                                                                                                     
  - MLflow - Experiment tracking and model registry                                                                                                                                                      
  - Pandera - Data validation                                                                                                                                                                            
  - Evidently - Drift detection                                                                                                                                                                          
  - FastAPI - Model serving                                                                                                                                                                              
  - Prometheus - Production metrics                                                                                                                                                                      
  - GitHub Actions - CI/CD automation                                                                                                                                                                    
                                                                                                                                                                                                         
  To start the project, create the directory structure and place raw data in data/raw/. The document includes all the Python code modules, YAML configs, and deployment manifests you need.


Created docs/algorithm-selection.md with comprehensive algorithm categorization:                                                                              
  ┌──────────────────────────┬──────────────────────────────────────────────────────────────────┐                                                               
  │         Section          │                             Contents                             │                                                               
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────┤                                                               
  │ Quick Decision Tree      │ Visual flow to pick the right problem type                       │                                                               
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────┤                                                               
  │ Classification           │ Binary, multi-class, multi-label with data size guidelines       │                                                               
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────┤                                                               
  │ Regression               │ Linear, regularized, tree-based + special cases (count, bounded) │                                                               
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────┤                                                               
  │ Clustering               │ K-Means, DBSCAN, HDBSCAN, GMM with shape/scalability comparison  │                                                               
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────┤                                                               
  │ Dimensionality Reduction │ PCA, t-SNE, UMAP, LDA use cases                                  │                                                               
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────┤                                                               
  │ Time Series              │ ARIMA, Prophet, LSTM, XGBoost with feature engineering           │                                                               
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────┤                                                               
  │ Anomaly Detection        │ Isolation Forest, One-Class SVM, Autoencoders by domain          │                                                               
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────┤                                                               
  │ NLP                      │ Traditional (TF-IDF) vs Transformers (BERT) by task              │                                                               
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────┤                                                               
  │ Recommendation           │ Collaborative filtering, Matrix Factorization, Content-based     │                                                               
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────┤                                                               
  │ Deep Learning            │ When to use, architecture by data type                           │                                                               
  └──────────────────────────┴──────────────────────────────────────────────────────────────────┘                                                               
  Each section includes:                                                                                                                                        
  - Comparison tables (best when / avoid when)                                                                                                                  
  - Code templates ready to use                                                                                                                                 
  - Selection guides based on data size and characteristics                                                                                                     
                                                                                                                                                                
  Quick reference card at the end for fast lookup. 