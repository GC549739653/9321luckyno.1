import warnings
from sklearn.feature_selection import RFE
from sklearn import model_selection
from sklearn.model_selection import train_test_split as split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
from sklearn import tree
from io import StringIO
from IPython.display import Image
import pydotplus
save_file = 'trained_model.sav'

def run_model():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

    # df = pd.read_csv('processed.cleveland.data',
    #                  names=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak",
    #                         "slope",
    #                         "ca", "thal", "target"])

    # df.replace("?", np.nan, inplace=True)
    # df.dropna(axis=0, inplace=True)
    # df.reset_index(drop=True, inplace=True)
    # df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
    # df[['age', 'sex', 'fbs', 'exang', 'ca']] = df[['age', 'sex', 'fbs', 'exang', 'ca']].astype(int)
    # df[['trestbps', 'chol', 'thalach', 'oldpeak']] = df[['trestbps', 'chol', 'thalach', 'oldpeak']].astype(float)
    # df['target'].replace(to_replace=[1, 2, 3, 4], value=1, inplace=True)

    # cp_dummy = pd.get_dummies(df['cp'])
    # cp_dummy.rename(columns={1: 'cp_typical_angina', 2: 'cp_atypical_angina',
    #                          3: 'cp_non_angina', 4: 'cp_asymptomatic_angina'}, inplace=True)
    # restecg_dummy = pd.get_dummies(df['restecg'])
    # restecg_dummy.rename(columns={0: 'restecg_normal', 1: 'restecg_wave_abnorm',
    #                               2: 'restecg_ventricular_ht'}, inplace=True)
    # slope_dummy = pd.get_dummies(df['slope'])
    # slope_dummy.rename(columns={1: 'slope_upsloping', 2: 'slope_flat',
    #                             3: 'slope_downsloping'}, inplace=True)
    # thal_dummy = pd.get_dummies(df['thal'])
    # thal_dummy.rename(columns={'3.0': 'thal_normal', '6.0': 'thal_fixed_defect',
    #                            '7.0': 'thal_reversible_defect'}, inplace=True)
    # df = pd.concat([df, cp_dummy, restecg_dummy, slope_dummy, thal_dummy], axis=1)

    # df.drop(['slope', 'cp', 'restecg', 'thal'], axis=1, inplace=True)
    # df = shuffle(df)

    # df_X = df.drop('target', axis=1)
    # df_y = df['target']

    # # print(df_X.columns.values.tolist())
    # selected_features = []
    # rfe = RFE(LogisticRegression())

    # rfe.fit(df_X.values, df_y.values)

    # for i, feature in enumerate(df_X.columns.values):
    #     if rfe.support_[i]:
    #         selected_features.append(feature)

    # selected_X = df_X[selected_features]
    # selected_y = df_y
    # # print(selected_X.columns.values.tolist())


    # selected_X_train, selected_X_test, selected_y_train, selected_y_test = split(selected_X, selected_y, test_size=0.3,
    #                                                                              random_state=40)
    # lr = LogisticRegression()
    # lr.fit(selected_X_train, selected_y_train)

    # print(f"LogisticRegression Accuracy: {lr.score(selected_X_test, selected_y_test):0.3f}")

    # parameters = [{'kernel': ['rbf'],
    #                'gamma': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5],
    #                'C': [1, 10, 100]},
    #               {'kernel': ['linear'],
    #                'C': [1, 10, 100]}]
    # grid = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=4)
    # grid.fit(selected_X_train, selected_y_train)
    # grid_means = grid.cv_results_['mean_test_score']
    # grid_stds = grid.cv_results_['std_test_score']

    # for mean, std, params in zip(grid_means, grid_stds, grid.cv_results_['params']):
    #     print(f"{mean:.3f} (+/-{std * 2:.03f}) for {params}")
    # svm_linear = svm.SVC(kernel='rbf', C=1, gamma=0.01)
    # svm_linear.fit(selected_X_train, selected_y_train)

    # print(f"SVM Accuracy: {svm_linear.score(selected_X_test, selected_y_test):.3f}")
    # print()
    # kfold = model_selection.KFold(n_splits=10, random_state=7)
    # models = [('Linear regression', lr),
    #           ('Support vector machine', svm_linear)]
    # for model in models:
    #     results = model_selection.cross_val_score(model[1],
    #                                               selected_X_train,
    #                                               selected_y_train,
    #                                               cv=kfold,
    #                                               scoring='accuracy')
    #     print(f"Cross validated', {model[0]}, 'Accuracy: {results.mean():.3f}")

    # print()
    # print("LogisticRegression features importance")
    # print(np.std(selected_X_train, 0) * lr.coef_[0])
    # lr_acc, grid_acc, svm_acc = lr.score(selected_X_test, selected_y_test),\
    #     model_selection.cross_val_score(lr,
    #                                               selected_X_train,
    #                                               selected_y_train,
    #                                               cv=kfold,
    #                                               scoring='accuracy')

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
    #df.sex = df.sex.astype('category')
    
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
    linearRegression_acc = linearRegression.score(test_x, test_y)
    print(f"linearRegressionAccuracy: {linearRegression_acc}")
    
    my_svm = svm.SVC(kernel='rbf', C=1, gamma=0.01)
    my_svm.fit(train_x, train_y)
    svm_acc = my_svm.score(test_x, test_y)
    print(f"SVM Accuracy: {svm_acc}")
    print()
    
    models = [('Linear regression', linearRegression), ('Support vector machine', my_svm)]
    results = model_selection.cross_val_score(linearRegression,train_x,train_y,cv=5,scoring='accuracy')
    print(f"Cross validated : Linear regression, 'Accuracy: {results.mean()}")
    results = model_selection.cross_val_score(my_svm, train_x, train_y, cv=3, scoring='accuracy')
    print(f"Cross validated : Support vector machine, 'Accuracy: {results.mean()}")

    better_model = linearRegression if linearRegression_acc>svm_acc else my_svm

    print(f"{better_model} is saved.")

    pickle.dump(better_model, open(save_file, 'wb'))
    






def prediction(sex,exang,ca,cp,restecg,slope,thal):
    model = pickle.load(open(save_file, 'rb'))
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

    results = model.predict([input])

    return results[0]

