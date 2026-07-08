"""
Employee Pay Scale Estimator Classification
Model Training Script
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)



# =========================
# Load Dataset
# =========================

DATA_PATH = "../dataset/adult.csv"

df = pd.read_csv(DATA_PATH)


print("Dataset Loaded")
print(df.head())



# =========================
# Data Cleaning
# =========================

# Remove spaces from column names

df.columns = df.columns.str.strip()


# Remove missing values

df.replace("?", pd.NA, inplace=True)

df.dropna(inplace=True)



# =========================
# Target Column
# =========================

# Change this if your dataset uses another name

target_column = "income"


# Convert target labels

df[target_column] = df[target_column].replace(
    {
        "<=50K": 0,
        ">50K": 1
    }
)



# =========================
# Separate Features and Target
# =========================

X = df.drop(
    target_column,
    axis=1
)

y = df[target_column]



# =========================
# Encode Categorical Features
# =========================

X = pd.get_dummies(
    X,
    drop_first=False
)



# Save feature columns

joblib.dump(
    X.columns,
    "../columns.pkl"
)


print(
    "Feature columns saved"
)



# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# =========================
# Model Training
# =========================

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)


print("Training Model...")


model.fit(
    X_train,
    y_train
)



# =========================
# Evaluation
# =========================

y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    f"\nAccuracy: {accuracy:.4f}"
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred
    )
)


print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)



# =========================
# Save Model
# =========================

joblib.dump(
    model,
    "../best_model.pkl"
)


print(
    "\nModel saved successfully as best_model.pkl"
)