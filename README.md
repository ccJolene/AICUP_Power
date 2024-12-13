# Power Generation Prediction Project

This repository contains the code and resources for the **Power Generation Prediction Project**. The goal of this project is to predict power generation based on environmental data (e.g., temperature, humidity, wind speed, sunlight, and other meteorological variables) using machine learning models, including **Random Forest Regressor**, **XGBoost**, **LSTM (Long Short-Term Memory)**, and more.

## Competition Name: AICUP 2024 - Electricity Generation Forecasting

Leaderboard Score: 702044.7  
**Rank: 29 / 934 (Top 3.1%)**  
The competition aims to utilize microclimate data to predict solar panel power output at various locations.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Models Used](#models-used)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Results](#results)
7. [Data Sources](#data-sources)
8. [Future Improvements](#future-improvements)

## Project Overview

In this project, various machine learning models are applied to predict power generation based on environmental features such as **wind speed**, **temperature**, **sunlight**, and **humidity**. The models utilized include:

- **Random Forest Regressor** (for general-purpose regression tasks)
- **XGBoost** (a gradient boosting algorithm known for its efficiency)
- **LSTM (Long Short-Term Memory)** (for time-series prediction tasks)
- **SVM (Support Vector Machine)**, **KNN (K-Nearest Neighbors)**, and **Linear Regression** for comparison

The project involves:
1. Preprocessing data: cleaning and handling missing values, generating features, and scaling.
2. Model training and evaluation: training various models and comparing their performance.
3. Hyperparameter tuning: optimizing model parameters for better accuracy.
4. Evaluation metrics: using **Mean Squared Error (MSE)** and **R²** as the evaluation metrics.

## Installation

To run this project on your local machine, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/ccJolene/AICUP_Power.git
cd AICUP_Power
```

### 2. Install dependencies
You can install the necessary Python libraries using **pip**. It's recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset
Make sure you have access to the dataset (`AvgData.csv`) used in this project. It contains environmental features and power generation data. 

## Usage

### LSTM
After setting up the environment and installing the dependencies, run the following command to start the training and testing of the models:

```bash
python lstm/main.py
```

The `main.py` script will:
1. Load the dataset.
2. Perform data preprocessing (handling missing values, scaling, etc.).
3. Train models (LSTM, GRU, etc.).
4. Evaluate the models and print the results.
5. Make predictions using a trained model, use the following:

### Random Forest
`random_forest.ipynb` is the main script, which contains model training, testing, and results visualization.

If you want to perform other analyses or feature transformations, you can use the following Notebooks:

`climate.ipynb`: This notebook processes the data fetched from the meteorological bureau's website and performs feature transformation. It is a failed case, as the discrepancy between the meteorological data and the competition data led to poor performance.

`data_preprocess/`: Contains all files related to data preprocessing.
1. `EDA.ipynb`: This notebook performs Exploratory Data Analysis (EDA), analyzing the distribution, outliers, and other statistical features of the data.
2. `selection.ipynb`: This notebook uses Lasso for feature selection to identify the most important features for prediction.
Project Files

## Models Used

### Random Forest Regressor
- Random Forest is an ensemble method that combines multiple decision trees to reduce overfitting and variance, making it a powerful algorithm for regression tasks.

### XGBoost
- XGBoost is a gradient boosting algorithm that optimizes decision trees for high predictive accuracy and performance.

### LSTM (Long Short-Term Memory)
- LSTM is a type of recurrent neural network (RNN) that is particularly useful for time-series prediction, capturing long-term dependencies in sequential data.

### Support Vector Machine (SVM)
- SVM is used for classification and regression tasks by finding a hyperplane that best separates the data.

### KNN (K-Nearest Neighbors)
- KNN is a non-parametric method used for classification and regression by comparing data points to their nearest neighbors.

### Linear Regression
- Linear Regression is a basic regression technique that assumes a linear relationship between the independent variables and the dependent variable.

## Evaluation Metrics

We evaluate the models using the following metrics:

- **Mean Squared Error (MSE)**: Measures the average of the squared differences between predicted and actual values. The lower the MSE, the better the model.
  
- **R² (Coefficient of Determination)**: Measures how well the predicted values match the actual values. An R² value closer to 1 indicates a better fit.

## Results

After training and evaluation, the results of different models are printed, including the **MSE** for each feature and the **average MSE** across all features. Below is a summary of the performance of each model:

| **Model**           | **Average MSE** |
|---------------------|-----------------|
| **Random Forest**    | 0.0864          |
| **XGBoost**          | 0.0980          |
| **SVM**              | 0.7854          |
| **KNN**              | 0.5168          |
| **Linear Regression**| 0.7704          |

As shown, **Random Forest** performed the best in predicting the power generation, followed by **XGBoost**. **SVM** and **Linear Regression** performed poorly due to their inability to capture complex relationships in the data.

## Data Sources

This project uses a combination of:
1. **Environmental Data**: Includes data such as temperature, wind speed, sunlight, humidity, and pressure, which are used to predict power generation.
2. **Weather Station Data**: For missing feature completion, we used official weather station data and applied linear transformations to supplement missing features in the main dataset.

You can find the processed data and model parameters in meteorological bureau's website.

## Future Improvements

- **Model Tuning**: Further hyperparameter tuning using Grid Search or Random Search could improve the models’ performance.
- **Ensemble Methods**: Combining the predictions of multiple models (e.g., Random Forest + XGBoost) could lead to improved results.
- **Additional Features**: We can include more weather features such as wind direction, cloud cover, etc., or use more advanced techniques like PCA for dimensionality reduction.
- **Deep Learning**: Experimenting with deeper neural networks like **CNN** or **Transformers** for time-series prediction could improve performance, especially when more data is available.
- **Data Augmentation**: Applying techniques like synthetic data generation or data augmentation could help in improving model performance, particularly for cases with missing data or imbalanced datasets.

---
