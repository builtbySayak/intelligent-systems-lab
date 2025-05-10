Cartoon vs Sketch Classification

A comparative study using real celebrity images from the IIIT-CFW dataset. We generated cartoon and sketch versions via image translation techniques, then classified them using CNNs and classical ML.

Models Used:
VGG-16 (transfer learning)
Random Forest, XGBoost (for low-resource testing)
Techniques:
Image preprocessing: bilateral filtering, edge detection
Sketch transformation using grayscale inversion + division
Data augmentation (rotation, color jitter, crop)
Accuracy:
Cartoons: ~25%
Sketches: ~38% Code: 'Cartoon_Sketch.ipynb'
