# ============================================================
# Credit Risk Prediction using Machine Learning
# German Credit Dataset
# Author: Gargi
# ============================================================

# -----------------------------
# Import Libraries
# -----------------------------

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# -----------------------------
# Create Required Folders
# -----------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print(" CREDIT RISK PREDICTION MODEL ")
print("=" * 60)

# ============================================================
# Load Dataset
# ============================================================

DATA_PATH = "data/german_credit_data.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")

# ============================================================
# Explore Dataset
# ============================================================

print("\nDataset Shape:", df.shape)

print("\nColumns")
print(df.columns.tolist())

print("\nFirst Five Rows")
print(df.head())

print("\nMissing Values")
print(df.isnull().sum())

# ============================================================
# Data Cleaning
# ============================================================

if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

print("\nRemoved unnecessary index column.")

# Encode Target

df["Risk"] = df["Risk"].map({
    "good": 1,
    "bad": 0
})

print("\nTarget Distribution")

print(df["Risk"].value_counts())

# ============================================================
# Features and Target
# ============================================================

X = df.drop("Risk", axis=1)

y = df["Risk"]

print("\nFeature Shape :", X.shape)
print("Target Shape :", y.shape)

# ============================================================
# Numerical and Categorical Columns
# ============================================================

numerical_features = [
    "Age",
    "Credit amount",
    "Duration"
]

categorical_features = [
    "Sex",
    "Job",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Purpose"
]

print("\nNumerical Features")

print(numerical_features)

print("\nCategorical Features")

print(categorical_features)

# ============================================================
# Preprocessing
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )

    ]
)

preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",
            numeric_transformer,
            numerical_features
        ),

        (
            "cat",
            categorical_transformer,
            categorical_features
        )

    ]

)

print("\nPreprocessing pipeline created.")

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Samples :", X_train.shape)

print("Testing Samples :", X_test.shape)

# ============================================================
# Models
# ============================================================

models = {

    "Logistic Regression":

        LogisticRegression(

            max_iter=1000,

            class_weight="balanced",

            random_state=42

        ),

    "Decision Tree":

        DecisionTreeClassifier(

            random_state=42,

            class_weight="balanced"

        ),

    "Random Forest":

        RandomForestClassifier(

            random_state=42,

            class_weight="balanced"

        )

}

# ============================================================
# Train Models
# ============================================================

results = []

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

for name, model in models.items():

    pipeline = Pipeline([

        ("preprocessor", preprocessor),

        ("classifier", model)

    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    auc = roc_auc_score(y_test, probabilities)

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": auc

    })

    print("\n" + "-" * 60)

    print(name)

    print("-" * 60)

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    print(f"ROC-AUC : {auc:.4f}")

    # ============================================================
# Model Comparison
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df)

# ============================================================
# Hyperparameter Tuning - Random Forest
# ============================================================

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5, 10, None],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

print("\nTraining Random Forest with GridSearchCV...\n")

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("Best Parameters")
print(grid_search.best_params_)

print("\nBest Cross Validation ROC-AUC")
print(round(grid_search.best_score_, 4))

# ============================================================
# Final Evaluation
# ============================================================

predictions = best_model.predict(X_test)

probabilities = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(y_test, predictions)

recall = recall_score(y_test, predictions)

f1 = f1_score(y_test, predictions)

auc = roc_auc_score(y_test, probabilities)

print("\n" + "=" * 60)
print("FINAL MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")

print("\nClassification Report\n")

print(classification_report(y_test, predictions))

# ============================================================
# Save Model
# ============================================================

joblib.dump(best_model, "models/best_model.pkl")

print("\nModel saved successfully!")

print("Location : models/best_model.pkl")

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Bad", "Good"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion Matrix saved.")

# ============================================================
# ROC Curve
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    probabilities
)

plt.title("ROC Curve")

plt.savefig(
    "outputs/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("ROC Curve saved.")

# ============================================================
# Feature Importance
# ============================================================

classifier = best_model.named_steps["classifier"]

encoder = best_model.named_steps["preprocessor"]\
    .named_transformers_["cat"]\
    .named_steps["encoder"]

encoded_columns = encoder.get_feature_names_out(categorical_features)

feature_names = numerical_features + list(encoded_columns)

importance = classifier.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,6))

plt.barh(
    importance_df["Feature"][:10],
    importance_df["Importance"][:10]
)

plt.gca().invert_yaxis()

plt.title("Top 10 Important Features")

plt.xlabel("Importance")

plt.savefig(
    "outputs/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Feature Importance saved.")

print("\n")
print("="*60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("="*60)



