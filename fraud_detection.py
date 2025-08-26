
# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

# -------------------------------
# Step 1: Load dataset
# -------------------------------
df = pd.read_csv("creditcard.csv")  # Adjust path if needed

print("Dataset shape:", df.shape)
print(df.info())
print(df.nunique())

# -------------------------------
# Step 2: Exploratory Data Analysis (EDA)
# Visualize distribution of selected features for fraud vs normal transactions
# -------------------------------
def plot_feature_distribution(feature):
    plt.figure(figsize=(8, 4))
    sns.kdeplot(df[df['Class'] == 0][feature], label='Normal', shade=True)
    sns.kdeplot(df[df['Class'] == 1][feature], label='Fraud', shade=True)
    plt.title(f'Distribution of {feature} by Class')
    plt.legend()
    plt.show()

# Example plots
plot_feature_distribution('V1')
plot_feature_distribution('V3')
plot_feature_distribution('V6')

# -------------------------------
# Step 3: Feature Scaling
# Scale 'Amount' and 'Time' features using StandardScaler
# -------------------------------
df['Amount_Scaled'] = StandardScaler().fit_transform(df[['Amount']])
df['Time_Scaled'] = StandardScaler().fit_transform(df[['Time']])

# -------------------------------
# Step 4: Prepare features and target
# Drop original 'Amount' and 'Time', keep scaled versions
# -------------------------------
X = df.drop(['Class', 'Time', 'Amount'], axis=1)
y = df['Class']

# -------------------------------
# Step 5: Split dataset into train and test sets
# Stratify split to keep class distribution
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------------
# Step 6: Balance training data using SMOTE
# Synthetic Minority Oversampling Technique to balance classes
# -------------------------------
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Before SMOTE class distribution:")
print(y_train.value_counts())
print("\nAfter SMOTE class distribution:")
print(y_train_smote.value_counts())

# -------------------------------
# Step 7: Train XGBoost Classifier
# -------------------------------
model_xgb = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model_xgb.fit(X_train_smote, y_train_smote)

# -------------------------------
# Step 8: Predict on test set and evaluate
# -------------------------------
y_pred = model_xgb.predict(X_test)

# Confusion matrix visualization
ConfusionMatrixDisplay.from_estimator(model_xgb, X_test, y_test)
plt.show()

# Print evaluation metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

# -------------------------------
# Step 9: Feature importance plot
# -------------------------------
plt.figure(figsize=(10, 6))
xgb.plot_importance(model_xgb, max_num_features=15, importance_type='gain', height=0.6)
plt.title("Top 15 Important Features")
plt.show()

# -------------------------------
# Step 10: Save trained model to disk
# -------------------------------
os.makedirs("model", exist_ok=True)
joblib.dump(model_xgb, "model/xgb_model.pkl")
print("Model saved to 'model/xgb_model.pkl'")