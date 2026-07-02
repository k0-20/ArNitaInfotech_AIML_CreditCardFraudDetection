import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

print("=" * 60)
print("LOADING TEST DATASET")
print("=" * 60)

# Load test dataset
df = pd.read_csv("data/fraudTest.csv")

print("Original Shape:", df.shape)

# -----------------------------
# Feature Engineering
# -----------------------------

df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

df["hour"] = df["trans_date_trans_time"].dt.hour
df["day"] = df["trans_date_trans_time"].dt.day
df["month"] = df["trans_date_trans_time"].dt.month

df["dob"] = pd.to_datetime(df["dob"])
df["age"] = 2026 - df["dob"].dt.year

# -----------------------------
# Drop columns
# -----------------------------

drop_columns = [
    "Unnamed: 0",
    "trans_date_trans_time",
    "cc_num",
    "first",
    "last",
    "street",
    "trans_num",
    "dob"
]

df.drop(columns=drop_columns, inplace=True)

# -----------------------------
# Features & Target
# -----------------------------

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# -----------------------------
# Load Best Model
# -----------------------------

model = joblib.load("models/best_model.pkl")

print("\nBest Model Loaded Successfully!")

# -----------------------------
# Prediction
# -----------------------------

y_pred = model.predict(X)

# -----------------------------
# Metrics
# -----------------------------

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(f"Accuracy : {accuracy_score(y, y_pred):.4f}")
print(f"Precision: {precision_score(y, y_pred):.4f}")
print(f"Recall   : {recall_score(y, y_pred):.4f}")
print(f"F1 Score : {f1_score(y, y_pred):.4f}")

print("\nClassification Report:\n")
print(classification_report(y, y_pred))

# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Fraud"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()

print("\nConfusion Matrix saved as confusion_matrix.png")