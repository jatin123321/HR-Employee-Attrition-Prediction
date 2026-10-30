# ============================================================
# PART 1: IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# ============================================================
# PART 2: LOAD DATASET
# ============================================================

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:", df.isnull().sum().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nAttrition Distribution:")
print(df["Attrition"].value_counts())


# ============================================================
# PART 3: DATA CLEANING
# ============================================================

df = df.drop_duplicates()
df = df.dropna(subset=["Attrition"])

# Remove columns that contain constant values or employee IDs
df = df.drop(
    columns=[
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours"
    ],
    errors="ignore"
)

print("\nCleaned Dataset Shape:", df.shape)
print("\nMissing Values After Cleaning:", df.isnull().sum().sum())

df.to_csv("cleaned_hr_attrition.csv", index=False)

print("\nCleaned dataset saved!")


# ============================================================
# PART 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# 1. Attrition distribution
df["Attrition"].value_counts().plot(kind="bar")

plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.show()


# 2. Age vs Attrition
df.boxplot(column="Age", by="Attrition")

plt.title("Age vs Attrition")
plt.suptitle("")
plt.xlabel("Attrition")
plt.ylabel("Age")
plt.show()


# 3. Monthly income vs Attrition
df.boxplot(column="MonthlyIncome", by="Attrition")

plt.title("Monthly Income vs Attrition")
plt.suptitle("")
plt.xlabel("Attrition")
plt.ylabel("Monthly Income")
plt.show()


# 4. Overtime vs Attrition
pd.crosstab(
    df["OverTime"],
    df["Attrition"]
).plot(kind="bar")

plt.title("Overtime vs Attrition")
plt.xlabel("Overtime")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.show()


# 5. Job satisfaction vs Attrition
pd.crosstab(
    df["JobSatisfaction"],
    df["Attrition"]
).plot(kind="bar")

plt.title("Job Satisfaction vs Attrition")
plt.xlabel("Job Satisfaction Level")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.show()


# ============================================================
# PART 5: FEATURE ENCODING AND DATA SPLITTING
# ============================================================

# Separate input features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"].map({"No": 0, "Yes": 1})

# Convert categorical columns to numerical columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

# First, separate the final test set
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Separate training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=42,
    stratify=y_train_full
)

print("\nTraining Records:", len(X_train))
print("Validation Records:", len(X_val))
print("Testing Records:", len(X_test))

print("\nFeatures:", X.shape[1])


# ============================================================
# PART 6: FEATURE SCALING
# ============================================================

scaler = StandardScaler()

# Fit scaler only on training data
X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed!")


# ============================================================
# PART 7: DEFINE FOUR MODELS AND PARAMETERS
# ============================================================

models = {
    "Logistic Regression": (
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ),
        {
            "C": [0.01, 0.1, 1, 10]
        }
    ),

    "Decision Tree": (
        DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),
        {
            "max_depth": [3, 5, 10, 15],
            "min_samples_leaf": [2, 5, 10]
        }
    ),

    "KNN": (
        KNeighborsClassifier(),
        {
            "n_neighbors": [5, 7, 9, 11, 15],
            "weights": ["uniform", "distance"]
        }
    ),

    "SVM": (
        SVC(
            class_weight="balanced",
            probability=True,
            random_state=42
        ),
        {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"]
        }
    )
}


# ============================================================
# PART 8: TRAIN, TUNE AND SELECT THRESHOLDS
# ============================================================

results = []
trained_models = {}
best_thresholds = {}

for name, (model, parameters) in models.items():

    print("\nTraining:", name)

    # Decision Tree does not require scaled features
    if name == "Decision Tree":
        train_data = X_train
        val_data = X_val
        test_data = X_test
    else:
        train_data = X_train_scaled
        val_data = X_val_scaled
        test_data = X_test_scaled

    # Tune parameters using training data only
    search = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    search.fit(train_data, y_train)

    best_model = search.best_estimator_

    trained_models[name] = best_model

    print("Best Parameters:", search.best_params_)
    print("Training CV F1 Score:", round(search.best_score_, 3))

    # Select threshold using validation data only
    val_probabilities = best_model.predict_proba(val_data)[:, 1]

    precision_values, recall_values, thresholds = (
        precision_recall_curve(y_val, val_probabilities)
    )

    f1_values = (
        2 * precision_values[:-1] * recall_values[:-1]
        / (
            precision_values[:-1]
            + recall_values[:-1]
            + 1e-10
        )
    )

    best_index = np.argmax(f1_values)
    best_threshold = thresholds[best_index]

    best_thresholds[name] = best_threshold

    # Evaluate on the untouched test set
    test_probabilities = best_model.predict_proba(test_data)[:, 1]

    predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        best_threshold
    ])


# ============================================================
# PART 9: COMPARE MODELS
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Threshold"
    ]
)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\nMODEL COMPARISON")
print(results_df.round(3).to_string(index=False))

# Save model comparison
results_df.to_csv("model_comparison.csv", index=False)


# ============================================================
# PART 10: VISUALIZE MODEL PERFORMANCE
# ============================================================

# Accuracy comparison
plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"] * 100
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# F1 score comparison
plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Model"],
    results_df["F1 Score"] * 100
)

plt.title("Model F1 Score Comparison")
plt.xlabel("Model")
plt.ylabel("F1 Score (%)")
plt.ylim(0, 100)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# ============================================================
# PART 11: EVALUATE THE BEST MODEL
# ============================================================

# Choose the model with the highest test F1 score
best_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_name]

best_threshold = best_thresholds[best_name]

if best_name == "Decision Tree":
    test_data = X_test
else:
    test_data = X_test_scaled

best_probabilities = best_model.predict_proba(test_data)[:, 1]

best_predictions = (
    best_probabilities >= best_threshold
).astype(int)

print("\nBEST MODEL:", best_name)
print("Selected Threshold:", round(best_threshold, 3))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        best_predictions,
        labels=[0, 1],
        target_names=["No Attrition", "Attrition"],
        zero_division=0
    )
)


# Confusion matrix
cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Attrition", "Attrition"]
)

display.plot()

plt.title("Confusion Matrix - " + best_name)
plt.tight_layout()
plt.show()


# ============================================================
# PART 12: FEATURE IMPORTANCE
# ============================================================

# Use the Decision Tree to inspect feature importance
tree_model = trained_models["Decision Tree"]

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": tree_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 10 IMPORTANT FEATURES")

print(
    importance_df.head(10).to_string(index=False)
)

# Plot top 10 features
top_features = importance_df.head(10).sort_values(
    by="Importance"
)

plt.figure(figsize=(9, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 10 Features for Employee Attrition")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Save feature importance
importance_df.to_csv(
    "feature_importance.csv",
    index=False
)

print("\nPROJECT COMPLETED SUCCESSFULLY!")