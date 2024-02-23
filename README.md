<img src="https://i.pinimg.com/originals/be/91/28/be9128d7793328687beefe59b5062c0e.gif" alt="MasterHead" width="1000" height="500">

# Heart Disease Analysis

This repository contains an analysis of heart disease using various machine learning and statistical techniques. The analysis is conducted in R programming language and utilizes several libraries for data manipulation, visualization, and modeling.

## Overview

Heart disease is a significant health concern worldwide, and understanding its risk factors and prediction methods is crucial for preventive healthcare. This analysis aims to explore the relationship between various factors such as smoking, physical health, age, and gender with the incidence of heart disease. It employs several machine learning algorithms and statistical techniques to predict heart disease risk and identify significant predictors.

## Key Indicators of Heart Disease

The dataset used in this analysis is derived from the 2022 annual CDC survey data, which includes information from over 400,000 adults regarding their health status. Key indicators of heart disease include high blood pressure, high cholesterol, smoking, diabetes status, obesity (high BMI), lack of physical activity, and excessive alcohol consumption. Detecting and preventing these risk factors is essential in healthcare, and machine learning methods can help identify patterns in the data to predict a patient's condition.

## Dataset Source and Processing

The dataset originates from the CDC's Behavioral Risk Factor Surveillance System (BRFSS), which conducts annual telephone surveys to collect data on the health status of U.S. residents. With over 400,000 adult interviews conducted each year, BRFSS is the largest continuously conducted health survey system globally. The dataset used in this analysis includes data from 2020 and has been processed to select the most relevant variables related to heart disease.

# Heart Disease Analysis

This repository contains an analysis of heart disease using various machine learning and statistical techniques. The analysis is conducted in R programming language and utilizes several libraries for data manipulation, visualization, and modeling.

## Contents

- **Scripts**: This folder contains the R script (`heart disease.R`) used for data preprocessing, exploratory data analysis, modeling, and evaluation.
- **Data**: This folder stores the dataset (`heart_2020.xlsx`) used for the analysis.

## Installation and Usage

To run the analysis script, follow these steps:

1. Ensure you have R and RStudio installed on your machine.
2. Clone or download this repository to your local machine.
3. Open RStudio and set the working directory to the root folder of the cloned repository.
4. Install the required R packages listed in the analysis script using the `install.packages()` function if you haven't already.
5. Run the `heart disease.R` script in RStudio.

## Steps of Analysis

1. **Data Cleaning and Manipulation**: The dataset (`heart_2020.xlsx`) is loaded into R, and necessary preprocessing steps are performed, including converting data types and checking for missing values.

2. **Exploratory Data Analysis (EDA)**: Various visualizations such as histograms, scatter plots, and correlation matrices are created to understand the distribution of variables and explore relationships between predictors and the target variable (heart disease).

3. **Modeling**:
   - **Lasso Regression**: A Lasso regression model is fitted to select significant predictors of heart disease using cross-validation to find the optimal regularization parameter.
   - **Random Forest**: A random forest classifier is trained to predict heart disease risk based on selected predictors.
   - **K-Nearest Neighbors (KNN)**: The optimal number of neighbors (k) for KNN classification is determined using cross-validation, and the KNN model is fitted accordingly.
   - **Linear Discriminant Analysis (LDA)**: LDA and Quadratic Discriminant Analysis (QDA) models are trained to classify heart disease based on selected predictors.
   - **Logistic Regression**: A logistic regression model is fitted to predict heart disease risk based on selected predictors.

4. **Evaluation**: The performance of each model is evaluated using various metrics such as accuracy, confusion matrix, and ROC curves.

## Contributors

- This data is from Kaggle by KAMIL PYTLAK. Thank you so much for the dataset.
