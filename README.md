Steel Plate Fault Classification Using Machine Learning
1. Problem Statement
The objective of this project is to develop a machine learning-based classification system to identify different types of faults in steel plates.

Five classification algorithms are implemented and evaluated using the Steel Plates Faults dataset. A Streamlit web application is developed to allow users to upload test data, select a classification model, and view its evaluation metrics.

2. Dataset Description
Dataset Name: Steel Plates Faults Dataset
Source: UCI Machine Learning Repository
Problem Type: Multiclass Classification
Target Variable: Fault_Type
Features: 27 numerical input features
Fault Classes: Seven fault categories
The dataset is used to train and evaluate five machine learning classification models.

3. GitHub Repository Link
Repository: [Insert your GitHub repository URL here]

4. Models Used and Evaluation Metrics
The following five classification algorithms are implemented:

Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Gaussian Naive Bayes
Support Vector Machine (SVM)
The following evaluation metrics are calculated:

Accuracy
AUC Score
Precision
Recall
F1 Score
Matthews Correlation Coefficient (MCC)
Model Comparison Table

ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	Insert actual result	Insert actual result	Insert actual result	Insert actual result	Insert actual result	Insert actual result
Decision Tree	Insert actual result	Insert actual result	Insert actual result	Insert actual result	Insert actual result	Insert actual result
KNN	Insert actual result	Insert actual result	Insert actual result	Insert actual result	Insert actual result	Insert actual result
Naive Bayes	Insert actual result	Insert actual result	Insert actual result	Insert actual result	Insert actual result	Insert actual result
Support Vector Machine	Insert actual result	Insert actual result	Insert actual result	Insert actual result	Insert actual result	Insert actual result
5. Observations on Model Performance
The performance of each classifier is evaluated using the same test dataset.


ML Model Name	Observation
Logistic Regression	Evaluate its performance based on the calculated metrics.
Decision Tree	Evaluate its performance based on the calculated metrics.
KNN	Evaluate its performance based on the calculated metrics.
Naive Bayes	Evaluate its performance based on the calculated metrics.
Support Vector Machine	Evaluate its performance based on the calculated metrics.
Overall Winner
The best-performing model will be selected based on the overall evaluation metrics, particularly F1 Score, AUC and MCC.

Best Model: To be updated based on actual experimental results.

6. Streamlit Application
The project includes an interactive Streamlit application with the following features:

CSV test dataset upload.
Selection of five trained classification models.
Display of six evaluation metrics.
Confusion matrix visualization.
Classification report.
Prediction results.
7. Project Structure

text
Steel-Plate-Fault-Classification/
│
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── test_data.csv
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
├── results/
│   └── model_comparison.csv
└── screenshots/
8. Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Streamlit
Joblib
9. Deployment
The application is intended to be deployed using Streamlit Community Cloud.

Live Application Link: To be updated after deployment.
