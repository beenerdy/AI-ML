# AI & Machine Learning Projects Portfolio

Welcome to my AI & Machine Learning Projects Portfolio! This repository showcases a collection of AI and Machine Learning projects I’ve completed through various courses and personal exploration. Each project demonstrates concepts, techniques, and skills applied in data science, machine learning, and deep learning. Below is a summary of the projects included, highlighting key skills and methodologies.

## Table of Contents

1. [Music Categorization and Playlist Creation](#1-music-categorization-and-playlist-creation)
2. [Facial Recognition Project](#2-facial-recognition-project)
3. [Stock Market Analysis](#3-stock-market-analysis)
4. [California Housing Price Prediction](#4-california-housing-price-prediction)
5. [Mercedes-Benz Greener Manufacturing](#5-mercedes-benz-greener-manufacturing)
6. [Loan Default Prediction](#6-loan-default-prediction)
7. [Image Classifier - Flower Species Recognition](#7-image-classifier---flower-species-recognition)

## Projects Overview

### 1. Music Categorization and Playlist Creation

- **Objective**: Classify music tracks into "liked" or "disliked" categories using audio feature extraction and machine learning. The project generates playlists compatible with Apple Music.
- **Techniques**:
  - Audio Feature Extraction (MFCCs, Chroma, Spectral Contrast)
  - Deep Learning (Conv1D, LSTM, Attention Mechanism)
  - Data Preprocessing, Balancing, Outlier Removal
- **Key Skills**: Deep Learning, Audio Signal Processing, Playlist Generation, TensorFlow

[Read more about this project](Music-Sorting-with-ML/Readme.md)

---

### 2. Facial Recognition Project

- **Objective**: Develop a deep learning model for facial recognition tasks.
- **Techniques**:
  - Convolutional Neural Networks (CNN)
  - Data Preprocessing and Normalization
  - Model Evaluation with Accuracy Metrics
- **Key Skills**: TensorFlow, Deep Learning, Image Classification

[Read more about this project](Facial-Recognition/Readme.md)

---

### 3. Stock Market Analysis

- **Objective**: Analyze stock market data, calculate returns, and implement technical indicators to gain insights into market trends.
- **Techniques**:
  - Exploratory Data Analysis (EDA)
  - Moving Averages, Bollinger Bands, RSI
  - Time Series Forecasting (if applicable)
- **Key Skills**: Python, Pandas, Data Visualization, Financial Analysis

[Read more about this project](Stock-Market-Analysis/Readme.md)

---

### 4. California Housing Price Prediction

- **Objective**: Predict median house values in California using linear regression.
- **Techniques**:
  - Data Preprocessing (Imputation, Encoding, Scaling)
  - Linear Regression
  - Model Evaluation using RMSE and R² Score
- **Key Skills**: Scikit-learn, Data Processing, Regression Analysis

[Read more about this project](California-Housing-Prediction/Readme.md)

---

### 5. Mercedes-Benz Greener Manufacturing

- **Objective**: Optimize the time cars spend on the test bench during manufacturing to reduce carbon emissions without compromising safety.
- **Techniques**:
  - Dimensionality Reduction
  - XGBoost for Predictive Modeling
  - Feature Selection
- **Key Skills**: Machine Learning, XGBoost, Feature Engineering

[Read more about this project](Mercedes-Benz-Greener-Manufacturing/Readme.md)

---

### 6. Loan Default Prediction

- **Objective**: Predict loan defaults using historical data and machine learning models.
- **Techniques**:
  - Deep Learning Model (Neural Networks)
  - Data Imbalance Handling (Oversampling)
  - Regularization (Early Stopping, Dropout)
- **Key Skills**: TensorFlow/Keras, Deep Learning, Classification

[Read more about this project](Loan-Default-Prediction/Readme.md)

---

### 7. Image Classifier - Flower Species Recognition

- **Objective**: Develop a deep learning model to classify flower species based on images.
- **Techniques**:
  - Transfer Learning with Pre-trained CNN (VGG16)
  - Data Augmentation
  - Model Evaluation and Testing
- **Key Skills**: PyTorch, Transfer Learning, Image Classification

[Read more about this project](Image-Classifier-Flower/Readme.md)

---

## Skills and Techniques

Across these projects, I’ve developed expertise in:

- **Deep Learning**: CNNs, Transfer Learning, Model Training, Evaluation
- **Data Preprocessing**: Handling missing data, normalization, feature engineering, dataset balancing
- **Exploratory Data Analysis (EDA)**: Data visualization, trend analysis, correlation studies
- **Machine Learning**: Regression, Classification, XGBoost, Neural Networks
- **End-to-End Development**: From data acquisition to model deployment
- **Audio Signal Processing**: Feature extraction (MFCCs, Chroma, Spectral Contrast) and playlist generation for music recommendation systems

## How to Use This Repository

1. Install required packages using `requirements.txt`, ideally in a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run Jupyter Notebook at the root level:
   ```bash
   jupyter notebook
   ```
3. Navigate through the provided notebooks to explore the projects.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
