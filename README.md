# ISO-ID
## Introduction
ISO-ID is a project that utilizes machine learning techniques to identify isotopes from gamma-ray spectra. The project consists of two main models - a classification model that predicts the isotope class and a regression model that predicts the isotope concentration.

## Brief

Data_training.py: Main script that takes in preped Data, finds the best parameters for training, trains and dumps the trained model as a joblib file.
Custom_loss.py : Contains custom loss function
comb_model_evaluation.py : Generates a test report for the models.
