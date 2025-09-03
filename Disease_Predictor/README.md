#Heart Disease Predictor

This project demonstrates a machine learning pipeline for predicting 
the presence of heart disease using a dataset from the UCI Machine Learning Repository. 

It showcases key steps in a data science workflow, including data preprocessing, 
exploratory data analysis, model training, and evaluation.

The goal of this project is to build a predictive model that 
can classify patients as having or not having heart disease based on their clinical and demographic data.

##Project Structure

* Heart_Disease_Pred.ipynb : The main Jupyter Notebook containing the entire workflow, from data ingestion to model prediction.
* heart_disease_uci.csv: The dataset used for training and testing the model (downloaded via Kaggle).
* heart_disease_rf_model.pkl: A serialized machine learning model (Random Forest Classifier).
* heart_scaler.pkl: A serialized scaler object for standardizing feature data.
* heart_dataset.csv :  A sample dataset used to demonstrate predictions on new data.

##Methodology

* Data Preprocessing and EDA
   - Missing Value Handling: Missing numeric values are imputed with the mean of their respective columns. Categorical columns with missing values are filled with 'Unknown'.
   - Feature Engineering: One-hot encoding is applied to all categorical features to convert them into a numerical format suitable for machine learning models.
   - Exploratory Data Analysis (EDA): Histograms and a correlation heatmap are generated to visualize data distributions and relationships between features.
   - Data Scaling: Features are standardized using StandardScaler to ensure that all features contribute equally to the model training process.
  
  


