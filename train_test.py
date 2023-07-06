# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:52:21 2023

@author: sanjy
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
final_df = pd.read_excel('dataset.xlsx')

X = final_df[['Recency', 'Total_Expenses', 'Income', 'Total_Acc_Cmp', 'TotalPurchases']].values
y = final_df['clusters'].values

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
k=5
num_folds=10
knn = KNeighborsClassifier(n_neighbors=k)

kfold = KFold(n_splits=num_folds)

scores = cross_val_score(knn, X, y, cv=kfold)

# Perform k-fold cross-validation
for train_index, test_index in kfold.split(X):
    # Split the data into train and test sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model on the training data
    knn.fit(X_train, y_train)

    # Evaluate the model on the test data
    accuracy = knn.score(X_test, y_test)
    print("Fold Accuracy: %.2f" % accuracy)
    
    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Initialize the KFold object
kfold = KFold(n_splits=10)

# Initialize lists to store predicted labels and actual labels
# Initialize lists to store predicted labels and actual labels
predicted_labels = []
actual_labels = []

# Perform k-fold cross-validation
for train_index, test_index in kfold.split(X):
    # Split the data into train and test sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the KNN model to the training data
    knn.fit(X_train, y_train)

    # Perform predictions on the test data
    y_pred = knn.predict(X_test)

    # Store the predicted and actual labels
    predicted_labels.extend(y_pred)
    actual_labels.extend(y_test)

# Print the predicted labels and actual labels
print("Predicted Labels:", predicted_labels)
print("Actual Labels:", actual_labels)


#DECISON TREE
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
dec0=DecisionTreeClassifier(criterion='entropy',random_state=33,max_depth=None)
dec0.fit(X_train,y_train)
y_pred=dec0.predict(X_test)
y_pred

from sklearn.model_selection import GridSearchCV
params = {
    'max_depth': [2, 3, 5, 6, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}
grid_search = GridSearchCV(estimator=dec0,
                           param_grid=params,
                           cv=5,
                           n_jobs=-1)
grid = grid_search.fit(X_train,y_train)
grid.best_score_
grid.best_params_

dec1=DecisionTreeClassifier(criterion='entropy',max_depth=10,min_samples_leaf=5)
dec1.fit(X_train,y_train)
y_pred1=dec1.predict(X_test)
y_pred1



import joblib

# Assuming your KNN model is named 'knn_model'
joblib.dump(dec1,'dec1.pkl')
