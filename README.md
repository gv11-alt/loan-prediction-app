Loan Approval Prediction System 🏦
A Full-Stack Machine Learning web application that predicts the likelihood of loan approval based on applicant profiles. This project utilizes Supervised Learning and an Ensemble Learning approach to provide high-accuracy predictions.

🧠 Core Machine Learning Concepts
1. Supervised Learning (Classification)
This project is a Binary Classification problem. We use labeled historical data to train the model to categorize new applications into two distinct classes: Approved (1) or Rejected (0).

2. Random Forest Classifier (Ensemble Learning)
Instead of relying on a single Decision Tree, we use a Random Forest. It builds multiple decision trees and merges them together to get a more accurate and stable prediction through a "majority voting" mechanism. This significantly reduces the risk of Overfitting.

3. Data Preprocessing & Feature Engineering
Imputation: Handled missing values using Median for numerical data (Income/Loan Amount) and Mode for categorical data (Gender/Married).

Label Encoding: Converted categorical text data into numerical format for mathematical processing.

Feature Scaling: Analyzed the impact of features like Credit History, which was identified as the most significant predictor.

🛠️ Tech Stack
Language: Python 3.x

Machine Learning: Scikit-Learn, Pandas, NumPy

Web Framework: Streamlit

Model Persistence: Joblib

🏗️ Project Architecture
The application follows a 3-tier architecture:

Data Layer: Raw CSV dataset containing applicant demographics and financial history.

Logic Layer (model.py): The training pipeline that cleans data, trains the Random Forest model, and serializes it into a .pkl file.

Presentation Layer (app.py): A Streamlit-based web interface that accepts user input, performs real-time inference, and displays results.

📊 Evaluation Results
The model was evaluated using a Confusion Matrix to ensure reliability:

Accuracy: ~82% (Standard for this dataset).

Precision/Recall: Optimized to minimize "False Positives" (approving a loan for a risky candidate).