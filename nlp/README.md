## Natural Language Processing Projects ##

This directory contains my work on NLP systems including abstractive summarization and sentiment analysis demonstrating practical applications of Machine Learning and Deep Learning in textual data understanding,
generation and classification.

# Abstractive Text Summarization #

Techniques Used:
  - Pointer-Generator Network with coverage loss
  - Auxiliary loss function to control novelty via 'p_gen'
  - Novelty score evaluation using normalized n-gram novelty
  - Training on CNN/DailyMail dataset with Gigaword for survey references
  - Architectural tuning with attention mechanisms and context vectors
Results:
  - Achieved higher n-gram novelty scores compared to vanilla PG.
  - Balanced novelty vs ROUGE trade-off using custom auxiliary loss.
Tools: PyTorch, CNN/DailyMail Dataset, ROUGE Evaluation
Link to the paper: https://arxiv.org/abs/2002.10959


# Sentiment Analysis on Amazon Product Reviews #

Techniques Used:
  - TF-IDF based feature extraction
  - Class balancing using under-sampling
  - Models: Logistic Regression vs Support Vector Machines (SVM)
Results:
  - SVM achieved 76% accuracy outperforming Logistic Regression (74%)
  - Visual analysis using confusion matrices and precision/recall curves
Tools: Python,Scikit-learn,PyTorch,Transformers,NLTK,Matplotlib,Pandas,Seaborn
Paper and Code Available.
