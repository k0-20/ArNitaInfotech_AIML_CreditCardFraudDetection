import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ===================================================
# CREATE MODELS DIRECTORY
# ===================================================

os.makedirs("models", exist_ok=True)

# ===================================================
# LOAD DATASET
# ===================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("data/fraudTrain.csv")

print("Original Shape :", df.shape)

# ---------------------------------------------------
# Faster training (change to full dataset later)
# ---------------------------------------------------

df = df.sample(n=100000, random_state=42)

print("Working Shape :", df.shape)

# ===================================================
# FEATURE ENGINEERING
# ===================================================

print("\nPerforming Feature Engineering...")

df["trans_date_trans_time"] = pd.to_datetime(
    df["trans_date_trans_time"]
)

df["hour"] = df["trans_date_trans_time"].dt.hour
df["day"] = df["trans_date_trans_time"].dt.day
df["month"] = df["trans_date_trans_time"].dt.month

# Age of transaction holder

df["dob"] = pd.to_datetime(df["dob"])

df["age"] = (
    2026 -
    df["dob"].dt.year
)

# ===================================================
# DROP USELESS COLUMNS
# ===================================================

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

print("\nRemaining Columns")

print(df.columns.tolist())

# ===================================================
# FEATURES & TARGET
# ===================================================

X = df.drop("is_fraud", axis=1)

y = df["is_fraud"]

print("\nFeature Shape :", X.shape)
print("Target Shape  :", y.shape)

# ===================================================
# TRAIN TEST SPLIT
# ===================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ===================================================
# PREPROCESSING
# ===================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical Features")

print(categorical_features)

print("\nNumerical Features")

print(numerical_features)

preprocessor = ColumnTransformer(

    transformers=[

        (

            "cat",

            OneHotEncoder(

                handle_unknown="ignore"

            ),

            categorical_features

        ),

        (

            "num",

            StandardScaler(),

            numerical_features

        )

    ]

)

print("\nPreprocessor Ready!")

# ===================================================
# CREATE MODELS
# ===================================================

models = {

    "Logistic Regression": Pipeline(

        steps=[

            ("preprocessor", preprocessor),

            ("classifier",

             LogisticRegression(

                 max_iter=1000,

                 class_weight="balanced",

                 random_state=42

             ))

        ]

    ),

    "Decision Tree": Pipeline(

        steps=[

            ("preprocessor", preprocessor),

            ("classifier",

             DecisionTreeClassifier(

                 random_state=42

             ))

        ]

    ),

    "Random Forest": Pipeline(

        steps=[

            ("preprocessor", preprocessor),

            ("classifier",

             RandomForestClassifier(

                 n_estimators=100,

                 random_state=42,

                 n_jobs=-1,

                 class_weight="balanced"

             ))

        ]

    )

}

# ===================================================
# TRAIN & EVALUATE
# ===================================================

results = []

best_model = None
best_model_name = ""
best_f1 = 0

print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(
        model,
        os.path.join("models", filename)
    )

    if f1 > best_f1:

        best_f1 = f1
        best_model = model
        best_model_name = name

# ===================================================
# SAVE BEST MODEL
# ===================================================

joblib.dump(

    best_model,

    "models/best_model.pkl"

)

# ===================================================
# RESULTS TABLE
# ===================================================

results_df = pd.DataFrame(

    results,

    columns=[

        "Model",

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score"

    ]

)

results_df = results_df.sort_values(

    by="F1 Score",

    ascending=False

)

results_df.to_csv(

    "models/model_comparison.csv",

    index=False

)

print("\n")
print("="*70)
print("MODEL COMPARISON")
print("="*70)

print(results_df)

print("\nBest Model :", best_model_name)
print("Best F1 Score :", round(best_f1,4))

print("\nFiles Saved Successfully!")

print("""
Saved Files:

models/
│
├── logistic_regression.pkl
├── decision_tree.pkl
├── random_forest.pkl
├── best_model.pkl
└── model_comparison.csv
""")

print("="*70)
print("TRAINING COMPLETED")
print("="*70)