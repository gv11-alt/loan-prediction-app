import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# STEP 1: LOAD DATA
# ==========================================
try:
    df = pd.read_csv('data/loan_data.csv')
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'loan_data.csv' not found in the 'data/' folder.")
    exit()

# ==========================================
# STEP 3: DATA PREPROCESSING (Handling Missing Values)
# ==========================================
# Filling Categorical nulls with Mode
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# Filling Numerical nulls with Median/Mode
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)

# Drop Loan_ID as it doesn't help in prediction
df.drop('Loan_ID', axis=1, inplace=True)

print("✅ Missing values handled.")

# ==========================================
# STEP 4: ENCODING (Text to Numbers)
# ==========================================
# We use LabelEncoder to convert columns like Gender, Education, etc.
le = LabelEncoder()
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("✅ Categorical variables encoded.")

# ==========================================
# STEP 5: MODEL TRAINING
# ==========================================
# Split features (X) and target (y)
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check Accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"✅ Model trained! Accuracy: {accuracy * 100:.2f}%")

# ==========================================
# STEP 6: SAVE THE MODEL (Serialization)
# ==========================================
# We save the model file so app.py can load it
joblib.dump(model, 'loan_model.pkl')
print("✅ Model saved as 'loan_model.pkl'")

# STEP 7: CONFUSION MATRIX (Optional, for evaluation)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Generate the matrix
cm = confusion_matrix(y_test, predictions)

# Display it beautifully
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Rejected', 'Approved'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Loan Prediction")
plt.show()