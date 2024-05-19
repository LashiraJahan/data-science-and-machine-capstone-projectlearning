# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Loading the dataset
data = pd.read_csv('spacex_launch_data.csv')

# Data Preprocessing
# Handling missing values
data = data.dropna()

# Encoding categorical variables
data = pd.get_dummies(data, columns=['launch_site', 'rocket_type', 'mission_outcome'], drop_first=True)

# Splitting data into features and target
X = data.drop(columns=['reusable'])
y = data['reusable']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
sns.countplot(x='reusable', data=data)
plt.title('Distribution of Reusable Rockets')
plt.show()

# Feature Engineering
# Adding any necessary feature engineering steps here

# Model Training
rf = RandomForestClassifier(random_state=42)
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(rf, param_grid=params, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Model Evaluation
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'ROC-AUC Score: {roc_auc:.2f}')
print('Confusion Matrix:')
print(conf_matrix)

# Model Deployment
# Function to predict reusability of new data
def predict_reusability(new_data):
    new_data_processed = scaler.transform(new_data)
    prediction = best_model.predict(new_data_processed)
    return prediction

# Example usage of deployment function
# new_data = pd.DataFrame({...})  # Replace with new data in the same format as X
# prediction = predict_reusability(new_data)
# print(f'Prediction: {"Reusable" if prediction[0] == 1 else "Not Reusable"}')

