import numpy as np
from urllib.request import urlopen
import urllib
import matplotlib.pyplot as plt # Visuals
import seaborn as sns 
import sklearn as skl
import pandas as pd

from sklearn.cross_validation import train_test_split # Create training and test sets
from sklearn.neighbors import KNeighborsClassifier # Kth Nearest Neighbor
from sklearn.tree import DecisionTreeClassifier # Decision Trees
from sklearn.tree import export_graphviz # Extract Decision Tree visual
from sklearn.tree import tree 
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn import svm #SVM
from sklearn.metrics import roc_curve # ROC Curves
from sklearn.metrics import auc # AUC 
from sklearn.model_selection import KFold, cross_val_score #cross validation 
from sklearn import cross_validation  #cross validation 
from urllib.request import urlopen # Get data from UCI Machine Learning Repository

import plotly.graph_objs as go
import plotly.plotly as py
import plotly.tools as pt



df = pd.read_csv(open("cleanedProjectData.csv"))

for item in df: #converts everything to floats
    df[item] = pd.to_numeric(df[item])

def normalize(heartDisease, toNormalize): #normalizes 
    result = heartDisease.copy()
    for item in heartDisease.columns:
        if (item in toNormalize):
            max_value = heartDisease[item].max()
            min_value = heartDisease[item].min()
            result[item] = (heartDisease[item] - min_value) / (max_value - min_value)
    return result
toNormalize = ['age', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak'] #columns to normalize
df = normalize(df, toNormalize)

for i in range(1,5):
    df['target'] = df['target'].replace(i,1)

print(df)
train, test = train_test_split(df, test_size = 0.50, random_state = 42)
# Create the training test omitting the diagnosis

training_set = train.ix[:, train.columns != 'target']
# Next we create the class set 
class_set = train.ix[:, train.columns == 'target']

# Next we create the test set doing the same process as the training set
test_set = test.ix[:, test.columns != 'target']
test_class_set = test.ix[:, test.columns == 'target']

dt = tree.DecisionTreeClassifier()
dt = dt.fit(train[['age', 'gender', 'cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']], train['target'])
predictions_dt = dt.predict(test[['age', 'gender', 'cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']])
predictright = 0
for i in range(0,predictions_dt.shape[0]-1):
    if (predictions_dt[i]== test.iloc[i][10]):
        predictright +=1
accuracy = predictright/predictions_dt.shape[0]
print(accuracy)

fitRF = RandomForestClassifier(random_state = 42, 
                                criterion='gini',
                                n_estimators = 500,
                                max_features = 5)
fitRF.fit(training_set, class_set['target'])
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=5, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=500, n_jobs=1, oob_score=False, random_state=42,
            verbose=0, warm_start=False)
importancesRF = fitRF.feature_importances_
indicesRF = np.argsort(importancesRF)[::-1]
predictions_RF = fitRF.predict(test_set)
accuracy_RF = fitRF.score(test_set, test_class_set['target'])

print("Here is our mean accuracy on the test set:\n",
     '%.3f' % (accuracy_RF * 100), '%')