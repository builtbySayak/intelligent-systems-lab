## Computer Vision Projects ##

This folder contains computer vision projects covering image classification, image translation and semantic segmentation using a combination of deep learning and traditional techniques. 
Projects include published and academic work, with real-world applications in robotics, art-tech and human perception analysis.

# Cartoon vs Sketch Classification #

A comparative study using real celebrity images from the IIIT-CFW dataset. We generated cartoon and sketch versions via image translation techniques, then classified them using CNNs and classical ML.

- Models Used:
  - VGG-16 (transfer learning)
  - Random Forest, XGBoost (for low-resource testing)
- Techniques:
  - Image preprocessing: bilateral filtering, edge detection
  - Sketch transformation using grayscale inversion + division
  - Data augmentation (rotation, color jitter, crop)
- Accuracy:
  - Cartoons: ~25%
  - Sketches: ~38%
Code: 'Cartoon_Sketch.ipynb'

# Two-Stage Semantic Segmentation #

Published work combining superpixel segmentation (SEEDS) and a custom deep net architecture (**Fork Net**) for semantic segmentation on Pascal VOC.

- Key Contributions:
  - Efficient semantic segmentation using superpixels
  - Dual-input ForkNet architecture for context-aware labeling
  - Achieved competitive mIoU vs FCN, DeepLabv2, and PSPNet
- Tools: OpenCV (C++), Keras (Python), TensorFlow, Pascal VOC dataset
- Link to the Paper: https://www.researchgate.net/publication/338685909_Two_Stage_Semantic_Segmentation_by_SEEDS_and_Fork_Net
