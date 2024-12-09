# Customer Churn Prediction for Streaming Services

## Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Set style for visualizations
sns.set_style('whitegrid')

## Data Loading
data_path = 'streaming_services_churn.csv'  # Replace with the actual dataset path
data = pd.read_csv(data_path)
print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")

# Preview the data
data.head()

## Data Preprocessing
# Check for missing values
missing_values = data.isnull().sum()
print("Missing values per column:")
print(missing_values)

# Drop unnecessary columns
data = data.drop(['CustomerID'], axis=1)  # Removing ID column

# Convert categorical columns to numerical
categorical_cols = ['SubscriptionType', 'PaymentMethod']  # Adjust as per dataset
for col in categorical_cols:
    data[col] = pd.factorize(data[col])[0]

# Handle missing values (if any)
data.fillna(data.median(), inplace=True)

## Exploratory Data Analysis
# Distribution of target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='Churn', data=data)
plt.title('Distribution of Churn')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

## Feature Selection and Splitting Data
# Features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

## Model Training
# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

## Model Evaluation
# Predictions
y_pred = model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

## Save the Model
import joblib
model_path = 'streaming_services_churn_model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}.")
