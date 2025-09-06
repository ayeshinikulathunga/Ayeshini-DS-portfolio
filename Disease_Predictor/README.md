# Heart Disease Predictor

This project is an end-to-end machine learning pipeline for predicting heart disease, culminating in a deployed, user-friendly web application. The goal is to not only build an accurate predictive model but also to make health insights accessible and understandable to different audiences, from medical professionals to patients and their families.

## Key Features

* End-to-End Pipeline: A comprehensive workflow covering data acquisition, cleaning, exploratory data analysis (EDA), model training, and deployment.
* Multi-Model Approach: Explores two distinct classification models, Logistic Regression and Random Forest Classifier, for performance comparison.
* User-Centric Application: Deployed a Streamlit web application with a unique multi-mode interface that tailors the language and information based on the user's role (Professional, Patient, or Family).
* Predictive Insights: Identifies the most impactful clinical features in heart disease prediction using feature importance analysis from the Random Forest model.

## Project Structure

* Heart_Disease_Pred.ipynb : The main Jupyter Notebook containing the entire workflow, from data ingestion to model prediction.
* heart_disease_uci.csv: The dataset used for training and testing the model (downloaded via Kaggle).
* heart_disease_rf_model.pkl: A serialized machine learning model (Random Forest Classifier).
* heart_scaler.pkl: A serialized scaler object for standardizing feature data.
* heart_dataset.csv :  A sample dataset used to demonstrate predictions on new data.
* app_streamlit.py: The main Python script for the deployed web application.
* requirements.txt: A list of all Python libraries required to run the project.
* README.md: The project overview and documentation file.
* feature_columns.pkl: A serialized file containing the list of feature columns used during training, ensuring consistency when making new predictions.

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

* Deployment & Application
    - An app_streamlit.py script was developed to serve the model as a web application, allowing users to input data and receive instant, interpretable predictions.
    - The application's different modes present information in a way that is relevant and understandable to a diverse audience.

## Libraries Used
  * Pandas
  * Matplotlib 
  * Seaborn
  * Scikit-learn
  * joblib
  * streamlit

## Future Improvements
   * Continuous Integration/Continuous Deployment (CI/CD): Implement a CI/CD pipeline to automate the deployment process.
   * Advanced Algorithms: Explore more complex models like Gradient Boosting or Neural Networks to potentially enhance predictive accuracy.
   * Enhanced Interpretability: Integrate tools like SHAP or LIME to provide deeper, more granular explanations for individual predictions.
  


