# Keras implementation of AutoAugment
This repository contains code for **AutoAugment** (only using paper's best policies) based on [AutoAugment:
Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501) implemented in Keras.

## Requirements
- Python 3.6
- Keras 2.2.4

## Training
### CIFAR-10
WideResNet28-10 baseline on CIFAR-10:
```
python train.py
```
WideResNet28-10 +Cutout, AutoAugment on CIFAR-10:
```
python train.py --cutout True --auto-augment True
```

## Results
### CIFAR-10
| Model                              |   Accuracy (%)    |   Loss   |
|:-----------------------------------|:-----------------:|:--------:|
|WideResNet28-10 baseline            |              95.32|    0.3717|
|WideResNet28-10 +Cutout, AutoAugment|              96.06|    0.3565|
