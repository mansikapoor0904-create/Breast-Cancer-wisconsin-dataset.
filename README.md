# Breast-Cancer-wisconsin-dataset.
Task 4: Classification with Logistic Regression.
📌 Project Overview

This project focuses on building a binary classification model using Logistic Regression to predict whether a tumor is malignant (cancerous) or benign (non-cancerous) using the Breast Cancer Wisconsin dataset.

🎯 Objective
Build a binary classifier using Logistic Regression
Understand model evaluation techniques
Analyze performance using metrics like accuracy, precision, recall, and ROC-AUC
🛠️ Tools & Technologies
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
📂 Dataset
Breast Cancer Wisconsin Dataset
Contains features computed from digitized images of breast mass
Target Variable:
0 → Malignant
1 → Benign
⚙️ Project Workflow
1️⃣ Data Loading
Loaded dataset using Scikit-learn
Converted into Pandas DataFrame
2️⃣ Data Exploration
Checked dataset structure using .info() and .describe()
Verified no missing values
Analyzed class distribution
3️⃣ Data Preprocessing
Separated features (X) and target (y)
Applied train-test split (80:20)
Standardized features using StandardScaler
4️⃣ Model Building
Used Logistic Regression model
Trained model on training data
5️⃣ Model Evaluation
Accuracy Score
Confusion Matrix
Classification Report (Precision, Recall, F1-score)
ROC-AUC Score
6️⃣ Visualization
Confusion Matrix heatmap using Seaborn
📊 Results
Achieved accuracy of around 95%–98%
Model performs well in distinguishing between malignant and benign tumors
Optional ROC Curve visualization
