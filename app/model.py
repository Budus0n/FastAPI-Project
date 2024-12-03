# Importing necessary libraries for data manipulation and machine learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Importing transformers and models for preprocessing and machine learning tasks
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.impute import SimpleImputer
from scipy.stats import ttest_ind
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Title and business understanding

"""
Project Title: Early Sepsis Prediction Using Machine Learning  
Framework: CRISP-DM (Cross-Industry Standard Process for Data Mining)

# Business Understanding
**Problem Statement**:  
Sepsis is a life-threatening condition caused by the body's response to infection. Early prediction of sepsis in ICU patients is crucial for timely interventions and improved patient outcomes.

**Goal**:  
Develop a machine learning model to predict whether a patient in the ICU will develop sepsis based on clinical data.

**Success Criteria**:  
The model should achieve high precision, recall, and F1-score, minimizing false negatives to ensure early detection of sepsis.

**Key Business Questions**:
1. What is the class distribution of sepsis in the dataset?
2. Are there any missing or duplicate records in the dataset?
3. Which features are most correlated with the target variable (Sepsis)?
4. Are there significant differences in mean age between patients with and without sepsis?
5. What is the distribution of key clinical features (e.g., plasma glucose, blood pressure)?
6. Which features exhibit multicollinearity (high VIF)?
7. What is the performance of baseline models (e.g., Logistic Regression)?
8. Can hyperparameter tuning improve model performance?
9. What are the most important features contributing to model predictions?
10. Can the final model be saved and deployed for future use?

**Hypotheses**:
- **Null Hypothesis (H₀)**: There is no significant difference in the mean age between patients with and without sepsis.
- **Alternative Hypothesis (H₁)**: Patients with sepsis have a significantly different mean age compared to those without sepsis.
"""

# Step 1: Data Understanding

df = pd.read_csv("data/Paitients_Files_Train.csv")

# Data Overview
print(df.info())
print(df.head())

# Column Descriptions
"""
- ID: Patient ID
- PRG: Plasma Glucose
- PL: Blood Work Result-1 (mu U/ml)
- PR: Blood Pressure (mm Hg)
- SK: Blood Work Result-2 (mm)
- TS: Blood Work Result-3 (mu U/ml)
- M11: Body Mass Index (BMI)
- BD2: Blood Work Result-4 (mu U/ml)
- Age: Patient Age (Years)
- Insurance: Insurance status
- Sepsis: Target variable (1 = Sepsis, 0 = No Sepsis)
"""

# Summary Statistics
print("\nSummary Statistics:")
print(df.describe(include="all").T)

# Checking for Missing and Duplicate Values
missing_values = df.isnull().sum()
duplicated_values = df.duplicated().sum()
print("\nMissing Values:\n", missing_values)
print("\nDuplicated Values:\n", duplicated_values)

# Class Distribution of Sepsis
print("\nClass Distribution")
print(df['Sepssis'].value_counts())

# Step 2: Data Exploration and Visualization

# Univariate analysis: Distribution of Sepsis
sns.countplot(x="Sepssis", data=df)
plt.title("Distribution of Sepsis Labels")
plt.show()

# Visualizing Numerical Variables
numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    plt.hist(df[col], bins=30, edgecolor='black')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Visualizing Categorical Variables
categorical_cols = df.select_dtypes(include=["object", "category"]).columns
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=df)
    plt.title(f"Count of {col}")
    plt.show()

# Bivariate Analysis: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[numerical_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Relationship with Target Variable
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x="Sepssis", y=col, data=df)
    plt.title(f"{col} vs Sepsis")
    plt.show()

# Hypothesis Testing: Age Difference between Sepsis and No Sepsis Groups
sepsis_group = df[df['Sepssis'] == 1]['Age']
no_sepsis_group = df[df['Sepssis'] == 0]['Age']

t_stat, p_value = ttest_ind(sepsis_group, no_sepsis_group, equal_var=False)
print("T-Statistic:", t_stat)
print("P-value:", p_value)

alpha = 0.05  # Significance level
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in age between sepsis and non-sepsis patients.")
else:
    print("Fail to reject the null hypothesis: No significant difference in age between sepsis and non-sepsis patients.")

# Step 3: Data Preparation
df = df.drop(columns=["ID"])  # Dropping patient ID column
test=df

print(test.info())

#save the test data
test.to_csv('data/test.csv', index=False)

#initialize predictions dictionary
test_predictions = {}


# Separating Features and Target
X = df.drop(columns=["Sepssis"])
y = df["Sepssis"]

# Preprocessing Pipeline: Handling missing values, outliers, and scaling
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # Handle missing values
    ("outlier_handler", RobustScaler()),  # Handle outliers
    ("scaler", QuantileTransformer(output_distribution='normal'))  # Normalize the features
])

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Feature Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("imputer", SimpleImputer(strategy='median'), X.columns),  
        ("outlier_handler", RobustScaler(), X.columns),
        ("scaler", QuantileTransformer(output_distribution='normal', n_quantiles=330), X.columns)
    ]
)

# Step 4: Model Building and Evaluation

# Step 4: Model Building and Evaluation
from sklearn.svm import SVC  # Add this import at the top of your file


# Initializing models for evaluation
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LightGBM": lightgbm.LGBMClassifier(),
    "SVM": SVC(kernel="linear", random_state=42)  # Adding SVM as an additional model
}

# Prepare results table to store performance metrics
results_table = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

# Train and evaluate models
for name, model in models.items():
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_val)

    # Compute the classification metrics
    metrics = classification_report(y_val, y_pred, output_dict=True)

    # Now append the metrics to the results table
    new_row = pd.DataFrame([{
        'Model': name,
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['weighted avg']['precision'],
        'Recall': metrics['weighted avg']['recall'],
        'F1-Score': metrics['weighted avg']['f1-score']
    }])

    # Use pd.concat to append the new row
    results_table = pd.concat([results_table, new_row], ignore_index=True)

    # Save each model after training
    joblib.dump(model_pipeline, f"{name}_model.pkl")
    print(f"{name} model saved as '{name}_model.pkl'")

# Sort results by F1 score
results_table = results_table.sort_values('F1 Score', ascending=False)
print("Model Performance Metrics:\n", results_table)

# Step 5: Model Saving
import joblib  # Import joblib at the top

# Save each model after training
for name, model_pipeline in models.items():  # Example of a loop
    joblib.dump(model_pipeline, f"{name}_model.pkl")
    print(f"{name} model saved as '{name}_model.pkl'")

# Sort results by F1 score
results_table = results_table.sort_values('F1 Score', ascending=False)
print("Model Performance Metrics:\n", results_table)

# Step 5: Model Saving
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Hyperparameter grid for SVC
params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Instantiate and fit the GridSearchCV
grid_search = GridSearchCV(estimator=SVC(), param_grid=params, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Save the best model
joblib.dump(grid_search.best_estimator_, "sepsis_model.pkl")
print("Model saved as 'sepsis_model.pkl'")

# Step 6: Model Deployment and Usage

# To use the saved model:
# loaded_model = joblib.load("sepsis


# Step 7: Answering Business Questions (with Visuals and Insights)
# Question 1: What is the class distribution of sepsis in the dataset?
sns.countplot(x="Sepssis", data=df, palette="pastel")
plt.title("Class Distribution of Sepsis")
plt.xlabel("Sepsis (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.show()

# **Insight**: Check the distribution of patients with and without sepsis.  
# If the dataset is imbalanced, it may require resampling techniques (e.g., SMOTE or undersampling).

# Question 2: Are there any missing or duplicate records in the dataset?
print("\nMissing Values:\n", missing_values)
print("\nDuplicate Records:\n", duplicated_values)

# **Action**: Missing values are addressed in the preprocessing pipeline via median imputation. Duplicate records can be dropped if found.

# Question 3: Which features are most correlated with the target variable (Sepsis)?
print(df.dtypes)
df_encoded = pd.get_dummies(df, drop_first=True)  # This will convert categorical variables into dummy/indicator variables
correlation = df_encoded.corr()  # or df_numeric.corr() if you dropped non-numeric columns

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# **Insight**: Observe which features have strong positive/negative correlations with Sepsis.

# Question 4: Are there significant differences in mean age between patients with and without sepsis?
print("\nT-Test Results:")
if p_value < alpha:
    print(f"Reject Null Hypothesis: Significant difference in mean age (p={p_value:.3f})")
else:
    print(f"Fail to Reject Null Hypothesis: No significant difference in mean age (p={p_value:.3f})")
    
sns.boxplot(x="Sepssis", y="Age", data=df, palette="coolwarm")
plt.title("Age Distribution by Sepsis Status")
plt.xlabel("Sepssis (1 = Yes, 0 = No)")
plt.ylabel("Age")
plt.show()

# **Insight**: Visualization shows the age distribution for patients with and without sepsis.

# Question 5: What is the distribution of key clinical features (e.g., plasma glucose, blood pressure)?
key_features = ["PRG", "PL", "PR", "SK", "TS"]
for feature in key_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature], kde=True, bins=30, color="blue")
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.show()

# **Insight**: These distributions help understand the clinical patterns in the dataset.

# Question 6: Which features exhibit multicollinearity (high VIF)?
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Add a constant to the features (required for VIF calculation)
X_train_with_const = add_constant(X_train)

# Calculate the VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X_train_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_with_const.values, i) for i in range(X_train_with_const.shape[1])]

# Display the VIF
print("\nVariance Inflation Factor (VIF):\n", vif_data)

print("\nVariance Inflation Factor (VIF):\n", vif_data)

# **Action**: High VIF values indicate multicollinearity. Consider removing highly collinear features to improve model interpretability.

# Question 7: What is the performance of baseline models (e.g., Logistic Regression)?
print("\nBaseline Logistic Regression Performance:")
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))
print("\nAccuracy Score:", accuracy_score(y_val, y_pred))

# **Insight**: Evaluate whether the baseline model meets performance expectations (e.g., F1-score, precision, recall).

# Question 8: Can hyperparameter tuning improve model performance?

# Example of GridSearchCV tuning for Logistic Regression
param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'saga']}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best model after tuning
best_model = grid_search.best_estimator_

# Now, proceed with the performance evaluation
print("\nBest Hyperparameters from GridSearchCV:\n", grid_search.best_params_)
print("\nBest Model Performance (After Tuning):")
y_pred_tuned = best_model.predict(X_val)
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred_tuned))
print("\nClassification Report:\n", classification_report(y_val, y_pred_tuned))
print("\nAccuracy Score:", accuracy_score(y_val, y_pred_tuned))

# **Insight**: Hyperparameter tuning can significantly improve model performance by adjusting parameters such as the regularization strength in Logistic Regression.

# Question 9: What are the most important features contributing to model predictions?
# For interpretability, feature importance can be extracted from tree-based models like RandomForest or LightGBM
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Encoding the target variable if it's categorical
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# train the RandomForest model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train_encoded)

# Create the feature importance DataFrame
input_features = X_train.columns  

feature_importances = pd.DataFrame(random_forest_model.feature_importances_,
                                   index=input_features, columns=["Importance"]).sort_values("Importance", ascending=False)

# Display feature importances
print(feature_importances)

feature_importances = pd.DataFrame(random_forest_model.feature_importances_,
                                   index=input_features, columns=["Importance"]).sort_values("Importance", ascending=False)

print("\nFeature Importances:")
print(feature_importances)

# Visualizing the top important features
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.index, y=feature_importances["Importance"], palette="viridis")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(rotation=90)
plt.show()

# **Insight**: The most important features for predicting sepsis can be derived from feature importance, and these will provide valuable insights into clinical risk factors.

# Question 10: Can the final model be saved and deployed for future use?
# Saving the trained model to a file for future deployment
import joblib
joblib.dump(best_model, "sepsis_model.pkl")
print("Model saved as 'sepsis_model.pkl'.")

# **Action**: The model is saved for deployment and future predictions in production environments.

# Conclusion: 
# The project focused on developing a machine learning model to predict sepsis in ICU patients based on clinical data. 
# Through the exploration and preparation of the data, the application of several machine learning models, and the evaluation of their performance, 
# the goal of achieving an early prediction of sepsis was successfully pursued. The model with the best performance can now be saved and deployed for use in clinical settings.

