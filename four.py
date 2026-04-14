import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df.head())

print(df.info())
print(df.describe())
print(df['target'].value_counts())

#Split Features and Target

X = df.drop('target', axis=1)
y = df['target']

#Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Feature Scaling (IMPORTANT)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Train Logistic Regression Model

model = LogisticRegression()
model.fit(X_train, y_train)

#Make Predictions

y_pred = model.predict(X_test)

#Evaluate Model

# Accuracy

print("Accuracy:", accuracy_score(y_test, y_pred))

#Confusion Matrix

print(confusion_matrix(y_test, y_pred))

#Classification Report

print(classification_report(y_test, y_pred))

#Visualization

#Confusion Matrix Plot

import seaborn as sns

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
