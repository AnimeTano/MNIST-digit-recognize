# MNIST Digit Recognizer

Image classification using Scikit-Learn, PyTorch and LightGBM

## Goal

Build a machine learning model capable of recognizing handwritten digits from 28×28 grayscale images.

## Dataset

- 42 000 training images
- 28 000 test images
- 10 classes (0–9)

The dataset is available on Kaggle:

https://www.kaggle.com/competitions/digit-recognizer

Download the files and place them into:
data/raw/

## Project stages

- Exploratory Data Analysis
- Image visualization
- Mean digit images
- Variance maps
- Baseline
- Error analysis
- Kaggle submission

## Project Structure

```
MNIST-digit-recognize/
│
├── data/
│   ├── raw/                     # train.csv, test.csv, sample_submission.csv
│   └── processed/              
│
├── notebooks/
│   └── mnist-digit-recognizer.ipynb   # полный пайплайн: EDA, обучение, оценка
│   └── mnist_cnn.ipynb # Архитектура CNN для прогнозирования
│
├── results/
│   ├── figures/                 # графики и визуализации
│   └── submissions/             # submission.csv для Kaggle
│
│
├── draw_digit.py                # PyQt6 приложение для рисования и распознавания
├── README.md
├── requirements.txt
└── .gitignore
```

## Results

|Model | Accuracy|
|------------|-------|
|LogRes      | 91.9% |
|LightGBM    | 97.1% |
|PyTorch CNN | 98.7% |

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- LightGBM
- PyTorch
- PyQt6
- Scikit-Learn
- Jupyter

## Result

Public Leaderboard Accuracy:

**0.96260**