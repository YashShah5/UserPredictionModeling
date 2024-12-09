import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from xgboost import XGBClassifier

import psycopg2

# Load dataset
data = pd.read_csv("Dataset_2.csv")

# Example preprocessing
data.dropna(inplace=True)  # Drop missing values if any

# Basic EDA (Exploratory Data Analysis) Columns Check
print(data.info())

# Example Data Cleaning (if needed)

# PostgreSQL Database Setup
def create_database_connection():
    print("Connecting to PostgreSQL database...")
    conn = psycopg2.connect(
        dbname="churn_analysis",
        user="final",
        password="project",
        host="localhost",
        port="5432"
    )
    print("Connection successful!")
    return conn

def create_tables(conn):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS customer_profiles (
        customer_id SERIAL PRIMARY KEY,
        age INT,
        location TEXT,
        
        payment_method TEXT,
        subscription_type TEXT
    );

    CREATE TABLE IF NOT EXISTS subscription_history (
        subscription_id SERIAL PRIMARY KEY,
        customer_id INT REFERENCES customer_profiles(customer_id),
        payment_plan TEXT,
        num_subscription_pauses INT,
        signup_date DATE
    );

    CREATE TABLE IF NOT EXISTS engagement_data (
        engagement_id SERIAL PRIMARY KEY,
        customer_id INT REFERENCES customer_profiles(customer_id),
        weekly_hours FLOAT,
        average_session_length FLOAT,
        song_skip_rate FLOAT,
        weekly_songs_played INT,
        churned BOOLEAN
    );
    """)
    conn.commit()
    cur.close()

def populate_tables(conn, data):
    cur = conn.cursor()
    import datetime

    for _, row in data.iterrows():
        # Convert signup_date to DATE (assuming it's an integer offset from a reference year, e.g., 1970)
        signup_date = datetime.date(1970, 1, 1) + datetime.timedelta(days=row['signup_date'])
        churned = bool(row['churned'])  # Convert churned to boolean explicitly

        cur.execute(
            """
            INSERT INTO customer_profiles (age, location, payment_method, subscription_type)
            VALUES (%s, %s, %s, %s) RETURNING customer_id;
            """,
            (row['age'], row['location'], row['payment_method'], row['subscription_type'])
        )
        customer_id = cur.fetchone()[0]

        cur.execute(
            """
            INSERT INTO subscription_history (customer_id, payment_plan, num_subscription_pauses, signup_date)
            VALUES (%s, %s, %s, %s);
            """,
            (customer_id, row['payment_plan'], row['num_subscription_pauses'], signup_date)
        )

        cur.execute(
            """
            INSERT INTO engagement_data (customer_id, weekly_hours, average_session_length, song_skip_rate,
                                         weekly_songs_played, churned)
            VALUES (%s, %s, %s, %s, %s, %s);
            """,
            (customer_id, row['weekly_hours'], row['average_session_length'], row['song_skip_rate'],
             row['weekly_songs_played'], churned)
        )

    conn.commit()
    cur.close()
    cur.close()

# Initialize and populate the database
conn = create_database_connection()
create_tables(conn)
populate_tables(conn, data)
conn.close()
# MonthlyCharges column not present in dataset, skipping this step
# TotalCharges column not present in dataset, skipping this step
data.dropna(inplace=True)

# 1. Churn Distribution
plt.figure(figsize=(6, 6))
data['churned'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'salmon'])
plt.title('Churn Distribution')
plt.annotate('Insight: Identifies churn percentage across all customers.', xy=(0.5, -0.2), xycoords='axes fraction', ha='center')
plt.ylabel('')
plt.savefig("churn_distribution.png")
plt.close()

# 2. Monthly Plan Preferences
# Graph skipped due to missing or incompatible data.


# 3. Age Distribution
plt.figure(figsize=(8, 6))
sns.histplot(data, x='age', hue='churned', kde=True, palette='coolwarm')
plt.title('Age Distribution by Churn Status')
plt.annotate('Insight: Shows churn behavior across different age groups.', xy=(0.5, -0.2), xycoords='axes fraction', ha='center')
plt.savefig("age_distribution.png")
plt.close()

# 4. Region-Wise Churn Rates
plt.figure(figsize=(10, 6))
region_churn = data.groupby('location')['churned'].mean()
region_churn.plot(kind='bar', color='salmon')
plt.title('Region-Wise Churn Rates')
plt.annotate('Insight: Reveals geographical churn patterns.', xy=(0.5, -0.2), xycoords='axes fraction', ha='center')
plt.xlabel('Region')
plt.ylabel('Churn Rate')
plt.savefig("region_churn_rates.png")
plt.close()

# 5. Customer Tenure Analysis
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='churned', y='weekly_hours', hue='churned', palette='coolwarm')
plt.title('Customer Tenure by Churn Status')
plt.annotate('Insight: Longer tenure may correlate with lower churn.', xy=(0.5, -0.2), xycoords='axes fraction', ha='center')
plt.savefig("tenure_analysis.png")
plt.close()

# 6. Average Monthly Usage
# Graph skipped due to missing or incompatible data.


# 7. Engagement by Subscription Plan
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='subscription_type', y='weekly_hours', hue='churned', palette='coolwarm')
plt.title('Engagement by Subscription Plan')
plt.annotate('Insight: Compares usage across subscription tiers.', xy=(0.5, -0.2), xycoords='axes fraction', ha='center')
plt.xlabel('Subscription Plan')
plt.ylabel('Average Engagement')
plt.savefig("engagement_by_plan.png")
plt.close()

# 8. Device Usage Trends
# Graph skipped due to missing or incompatible data.


# Encoding categorical variables using LabelEncoder
from sklearn.preprocessing import LabelEncoder
categorical_cols = ['location', 'subscription_type', 'payment_plan', 'payment_method', 'customer_service_inquiries']
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Splitting Data for Model Training
y = data['churned']
X = data.drop(['churned', 'customer_id'], axis=1)  # Drop non-numeric or target columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training a Simple Model
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 9. Feature Importance
plt.figure(figsize=(10, 6))
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(X.shape[1]), importances[indices], color='skyblue')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.title('Feature Importance')
plt.annotate('Insight: Key factors driving churn prediction.', xy=(0.5, -0.2), xycoords='axes fraction', ha='center')
plt.savefig("feature_importance.png")
plt.close()

# 10. Model Evaluation Metrics
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('ROC Curve')
plt.annotate('Insight: Evaluates model performance in classification.', xy=(0.5, -0.2), xycoords='axes fraction', ha='center')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.close()
