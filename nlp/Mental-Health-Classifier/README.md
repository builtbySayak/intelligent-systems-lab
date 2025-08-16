# Mental Health Classifier

This repository contains my MSc project on **Detecting and Interpreting Mental Health Distress in UK University Reddit Posts**.  
The system classifies Reddit-style posts into categories: Academic Stress, Relationship Issues, Existential Crisis, Social Isolation, and Neutral.  
It integrates both **classical ML models** and **transformer-based models** into an interactive **Streamlit dashboard**.

---

## Logistic Regression (TF-IDF + LR)
**Techniques Used:**
- TF-IDF based feature extraction with unigram + bigram features  
- Regularized Logistic Regression for binary and multi-class classification  
- Class balancing using under-sampling to handle label imbalance  
- Baseline benchmark for classical ML performance  

**Results:**
- Accuracy: **86%**  
- F1-Score: **0.58**  
- ROC-AUC: **0.90**

---

## RoBERTa Transformer
**Techniques Used:**
- Pretrained RoBERTa-base fine-tuned on 7,689 Reddit posts  
- Tokenization with Hugging Face Transformers  
- Optimized with AdamW + learning rate scheduling  
- Transfer learning for contextual understanding of distress expressions  

**Results:**
- Accuracy: **94%**  
- F1-Score: **0.75**  
- ROC-AUC: **0.93**

---

## Stacked Ensemble (LR + SVM + RoBERTa)
**Techniques Used:**
- Combined predictions from Logistic Regression, SVM, and RoBERTa  
- Majority voting with weighted confidence scores  
- Ensemble designed to reduce single-model biases  

**Results:**
- Accuracy: **95%**  
- Improved robustness on ambiguous examples  
- Reduced variance in misclassifications across categories  

---

## Key Features
-  Multiple models: Logistic Regression, SVM, XGBoost, DistilBERT, DeBERTa, and Stacked Ensemble  
-  Streamlit Web App: Interactive predictions with confidence scores, word clouds, and ROC curves  
-  Robust classification of real Reddit posts  
-  Evaluation metrics: Accuracy, F1-score, ROC-AUC, confusion matrices  

---

## Future Work
- Multi-class emotion categorisation (e.g., Academic Stress vs. Social Isolation)  
- Emotion intensity regression for distress severity  
- Real-time Reddit post streaming & deployment  

---

 **Citation**:  
Sayak Chakraborty,  
*"Detecting and Interpreting Mental Health Distress in UK University Reddit Posts"*,  
MSc Project, Queen Mary University of London, 2025.  

---
