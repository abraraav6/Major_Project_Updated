# Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import streamlit as st
# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# Import libraries
import pandas as pd
import plotly.express as px

# Create empty lists to store results
clf_names = []
acc_scores = []
prec_scores = []
rec_scores = []
f1_scores = []

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define classifiers
clf_lr = LogisticRegression()
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_dt = DecisionTreeClassifier(max_depth=5)
clf_rf = RandomForestClassifier(n_estimators=10)
clf_svc = SVC(kernel='linear')

# Train and test classifiers
classifiers = [('Logistic Regression', clf_lr), ('K-Nearest Neighbors', clf_knn), ('Decision Tree', clf_dt), ('Random Forest', clf_rf), ('Support Vector Machines', clf_svc)]
for clf_name, clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    clf_names.append(clf_name)
    acc_scores.append(acc)
    prec_scores.append(prec)
    rec_scores.append(rec)
    f1_scores.append(f1)

# Create dataframe from results
data = pd.DataFrame({
    'Classifier': clf_names,
    'Accuracy': acc_scores,
    'Precision': prec_scores,
    'Recall': rec_scores,
    'F1 Score': f1_scores
})

# Melt dataframe for easier plotting
melted = pd.melt(data, id_vars=['Classifier'], var_name='Metric', value_name='Score')

# Create bar chart using Plotly Express
fig = px.bar(melted, x='Classifier', y='Score', color='Metric', barmode='group')
fig.show()
