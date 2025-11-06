# Heart Disease Prediction using Machine Learning

This repository contains a machine learning project to predict heart disease based on the Cleveland Heart Disease dataset.

## Project Overview

The project explores different machine learning models to predict the presence of heart disease. The data is loaded, preprocessed, and various classification models are trained and evaluated. The best performing model (Gaussian Naive Bayes) is saved for future use in the `app.py` script.

## Files in this Repository

*   `README.md`: This file.
*   `heart_disease_prediction.ipynb`: The Jupyter notebook containing the data loading, preprocessing, model training, and evaluation steps.
*   `NB_model.pkl`: The serialized Gaussian Naive Bayes model trained on the dataset.
*   `app.py`: A Python script to load the trained model and make predictions on new data.
*   `requirements.txt`: A file listing the Python dependencies required to run the project.
*   `processed.cleveland.data`: The dataset used for training and evaluation.

## Setup Instructions

1.  Clone the repository:
2. cd <repository_name>
3. pip install -r requirements.txt
4. python app.py
