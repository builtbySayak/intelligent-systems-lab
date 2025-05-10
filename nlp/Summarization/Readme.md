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
