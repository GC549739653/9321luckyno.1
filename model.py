from sklearn.feature_selection import RFE
from sklearn import model_selection
from sklearn.model_selection import train_test_split as split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import pickle
from sklearn import tree
from io import StringIO
from IPython.display import Image
import pydotplus

df = pd.read_csv('processed.cleveland.data', names=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope",
         "ca", "thal", "target"])

df.replace("?", np.nan, inplace=True)
df.dropna(axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
df['target'].replace(to_replace=[1, 2, 3, 4], value=1, inplace=True)


df[['age', 'sex', 'fbs', 'exang', 'ca']] = df[['age', 'sex', 'fbs', 'exang', 'ca']].astype(int)
df.cp = df.cp.astype('category')
df.restecg = df.restecg.astype('category')
df.slope = df.slope.astype('category')
df.thal = df.thal.astype('category')
df = pd.get_dummies(df)
df[['trestbps', 'chol', 'thalach', 'oldpeak']] = df[['trestbps', 'chol', 'thalach', 'oldpeak']].astype(float)


df.rename(columns={'cp_1.0': 'cp_typical_angina', 'cp_2.0': 'cp_atypical_angina','cp_3.0': 'cp_non_angina', 'cp_4.0': 'cp_asymptomatic_angina'}, inplace=True)
df.rename(columns={'restecg_0.0': 'restecg_normal', 'restecg_1.0': 'restecg_wave_abnorm','restecg_2.0': 'restecg_ventricular_ht'}, inplace=True)
df.rename(columns={'slope_1.0': 'slope_upsloping', 'slope_2.0': 'slope_flat','slope_3.0': 'slope_downsloping'}, inplace=True)
df.rename(columns={'thal_3.0': 'thal_normal', 'thal_6.0': 'thal_fixed_defect','thal_7.0': 'thal_reversible_defect'}, inplace=True)

df = shuffle(df)

df_X = df.drop('target', axis=1)
df_y = df['target']

selected_features = []
rfe = RFE(LogisticRegression())
rfe.fit(df_X.values, df_y.values)

for i, feature in enumerate(df_X.columns.values):
    if rfe.support_[i]:
        selected_features.append(feature)

selected_X = df_X[selected_features]
selected_y = df_y

train_x, test_x, train_y, test_y = split(selected_X, selected_y, test_size=0.3, random_state=40)

linearRegression = LogisticRegression()
linearRegression.fit(train_x, train_y)
print(f"Accuracy: {linearRegression.score(test_x, test_y)}")

svm = svm.SVC(kernel='rbf', C=1, gamma=0.01)
svm.fit(train_x, train_y)
print(f"SVM Accuracy: {svm.score(test_x, test_y)}")
print()

models = [('Linear regression', linearRegression), ('Support vector machine', svm)]
results = model_selection.cross_val_score(linearRegression,train_x,train_y,cv=5,scoring='accuracy')
print(f"Cross validated : Linear regression, 'Accuracy: {results.mean()}")
results = model_selection.cross_val_score(svm, train_x, train_y, cv=3, scoring='accuracy')
print(f"Cross validated : Support vector machine, 'Accuracy: {results.mean()}")

print()
print("LinearRegression features importance")
print(np.std(train_x, 0) * linearRegression.coef_[0])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(selected_X, df_y)
feat_importance = clf.tree_.compute_feature_importances(normalize=False)
print("feat importance = " + str(feat_importance))
dot_data = tree.export_graphviz(clf, out_file=None,
                            feature_names=selected_X.columns,
                            class_names=['No', 'Yes'],
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("tree.png")
Image(graph.create_png())

save_file = 'trained_model.sav'
pickle.dump(svm, open(save_file, 'wb'))

def prediction(sex,exang,ca,cp,restecg,slope,thal):
    svm = pickle.load(open(save_file, 'rb'))
    input = [0 for i in range(11)]
    input[0] = sex
    input[1] = exang
    input[2] = ca
    if cp == 1.0:
        input[3] = 1
    elif cp == 3.0:
        input[4] = 1
    elif cp == 4.0:
        input[5] = 1
    if restecg == 0.0:
        input[6] = 1
    if slope == 1.0:
        input[7] = 1
    if thal == 3.0:
        input[8] = 1
    elif thal == 6.0:
        input[9] = 1
    elif thal == 7.0:
        input[10] = 1

    results = svm.predict([input])

    return results[0]
