# Fingerprint-Spoof-Detection
A fingerprint spoof detection system that uses Local Binary Pattern Histogram (LBPH) features (proposed in [A comparative study of texture measures with classification based on featured distributions](https://www.sciencedirect.com/science/article/abs/pii/0031320395000674)). It uses Support Vector Machines ([SVM](https://scikit-learn.org/stable/modules/svm.html)) as a classifier.
## Requirements
```
Python 3
scikit-image
scikit-learn
NumPy
OpenCV
```
## Utility
These days, anyone can easily fabricate the fingerprint of anyone with the help of latex, gelatin, etc. claim his/her indentity. To avoid this, this fingerprint spoof detection system has been made. Given an input image of a fingerprint, it classifies it as spoof or real. The accuracy using a simple linear SVM is up to more than 80% (and can be improved much using other complex models). This system can be used as an effective counter-measure against spoof
attacks.
## Dataset
The model was trained & tested on the [LiveDet2011 Dataset](https://ieeexplore.ieee.org/document/6199810) . The dataset has not been uploaded due to copyright issues (The tree structure of the dataset is here in this code however for convenience)
