# Diabetis-prediction-with-XGBoost
This project builds a robust machine learning pipeline to predict the likelihood of diabetes using the Pima Indians Diabetes dataset. It leverages XGBoost, one of the most powerful classification algorithms, and integrates preprocessing, hyperparameter tuning, and model evaluation techniques.
# Step 1: Install XGBoost (if not already installed)
!pip install xgboost

# Step 2: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

from xgboost import XGBClassifier, plot_importance

# Step 3: Load a Clean Diabetes Dataset (Kaggle or use UCI Pima)
# Here, we use Pima Indians Diabetes Dataset from UCI
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", 
           "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(url, header=None, names=columns)

# Step 4: Split Features and Target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build Pipeline (Scaler + XGBoost)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# Step 7: Hyperparameter Tuning
param_grid = {
    'xgb__n_estimators': [100, 150],
    'xgb__max_depth': [3, 5],
    'xgb__learning_rate': [0.05, 0.1],
}

grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# Step 8: Model Evaluation
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 10: ROC Curve & AUC Score
y_proba = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Step 11: Feature Importance (XGBoost)
xgb_final = best_model.named_steps['xgb']
plot_importance(xgb_final)
plt.title("Feature Importance")
plt.show()

Result images:
![image](https://github.com/user-attachments/assets/b73000dc-1a57-4848-8785-782d69607620)
![image](https://github.com/user-attachments/assets/78f2235f-b2f3-498c-ba98-8bb19a38d96c)
![image](https://github.com/user-attachments/assets/1065ddc4-6922-418b-afdf-4ff7a6633fb6)
![image](https://github.com/user-attachments/assets/7d9fdb02-e41c-4d10-8f01-cb6773b381f9)



