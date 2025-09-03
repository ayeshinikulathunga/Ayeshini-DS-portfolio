# Heart Disease Predictor

This project demonstrates a machine learning pipeline for predicting 
the presence of heart disease using a dataset from the UCI Machine Learning Repository. 

It showcases key steps in a data science workflow, including data preprocessing, 
exploratory data analysis, model training, and evaluation.

The goal of this project is to build a predictive model that 
can classify patients as having or not having heart disease based on their clinical and demographic data.

## Project Structure

* Heart_Disease_Pred.ipynb : The main Jupyter Notebook containing the entire workflow, from data ingestion to model prediction.
* heart_disease_uci.csv: The dataset used for training and testing the model (downloaded via Kaggle).
* heart_disease_rf_model.pkl: A serialized machine learning model (Random Forest Classifier).
* heart_scaler.pkl: A serialized scaler object for standardizing feature data.
* heart_dataset.csv :  A sample dataset used to demonstrate predictions on new data.

## Methodology

* Data Preprocessing and EDA
   - Missing Value Handling: Missing numeric values are imputed with the mean of their respective columns. Categorical columns with missing values are filled with 'Unknown'.
   - Feature Engineering: One-hot encoding is applied to all categorical features to convert them into a numerical format suitable for machine learning models.
   - Exploratory Data Analysis (EDA): Histograms and a correlation heatmap are generated to visualize data distributions and relationships between features.
   - Data Scaling: Features are standardized using StandardScaler to ensure that all features contribute equally to the model training process.
 
 * Model Training and Evaluation
   - Model Selection: Two classification models, Logistic Regression and Random Forest Classifier, are used to predict the target variable.
   - Model Training: Both models are trained on the preprocessed and scaled training data.
   - Evaluation: The models are evaluated based on their accuracy scores and classification reports. A confusion matrix is plotted for the Logistic Regression model to visualize its performance.
   - Feature Importance: Feature importance is calculated using the Random Forest model to identify the most influential features in the prediction.

* Model Persistence & Prediction
   - The trained Random Forest model and the StandardScaler are saved using joblib for future use.
   - The script demonstrates how to load the saved model and scaler to make predictions on new, unseen data provided in a separate CSV file.

## Libraries Used
  * Pandas
  * Matplotlib 
  * Seaborn
  * Scikit-learn
  * joblib

## Future Improvements
   * Model Deployment: The trained model can be deployed as a web application or an API, allowing users to input their data and receive real-time predictions.
   * Advanced Modeling: More complex machine learning algorithms, such as Gradient Boosting Machines, Support Vector Machines, or even a simple neural network, could be explored to potentially improve predictive accuracy.
   * Interpretability: Techniques like SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) could be used to provide detailed insights into why the model made a specific prediction for a given patient.
   * Automated Data Pipelines: The entire process could be automated using a data pipeline tool to regularly update the model with new data, ensuring it remains accurate and relevant.
  
  


