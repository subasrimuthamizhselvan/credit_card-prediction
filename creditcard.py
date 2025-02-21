import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# Step 1: Generate Synthetic Dataset
X, y = make_classification(
    n_samples=1000, 
    n_features=10, 
    n_informative=8, 
    n_redundant=2, 
    random_state=42, 
    class_sep=1.5
)

# Create a DataFrame
columns = [f'Feature_{i+1}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=columns)
df['Approval_Status'] = y

# Display basic dataset information
print("Dataset Information:")
print(df.info())
print("\nFirst few rows of the dataset:")
print(df.head())

# Step 2: Handle Imbalanced Dataset
print("\nClass distribution before SMOTE:")
print(df['Approval_Status'].value_counts())

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(df.drop('Approval_Status', axis=1), df['Approval_Status'])

print("\nClass distribution after SMOTE:")
print(pd.Series(y).value_counts())

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Multiple Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    results[name] = {"Accuracy": accuracy, "AUC": auc}
    print(f"{name}: Accuracy = {accuracy:.2f}, AUC = {auc:.2f}")

# Step 5: Visualize Model Performance
accuracy_scores = [result["Accuracy"] for result in results.values()]
auc_scores = [result["AUC"] for result in results.values()]

plt.figure(figsize=(12, 6))
plt.bar(results.keys(), accuracy_scores, color='skyblue', label='Accuracy')
plt.bar(results.keys(), auc_scores, color='orange', alpha=0.7, label='AUC', bottom=accuracy_scores)
plt.title('Model Performance (Accuracy and AUC)')
plt.ylabel('Score')
plt.legend()
plt.show()

# Step 6: Plot Confusion Matrix for the Best Model (Random Forest)
best_model = models["Random Forest"]
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 7: Feature Importances
feature_importances = pd.Series(best_model.feature_importances_, index=columns).sort_values(ascending=False)
plt.figure(figsize=(8, 6))
feature_importances.plot(kind='bar', color='green')
plt.title('Feature Importances - Random Forest')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.show()

# Step 8: ROC Curve for Random Forest
y_pred_prob = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='Random Forest (AUC = {:.2f})'.format(results["Random Forest"]["AUC"]))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('ROC Curve - Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()