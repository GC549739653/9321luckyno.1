import pandas as pd
import csv

def load_diet(diet_path):
    cleanData()
    df = pd.read_csv(diet_path,)
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs','restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca','thal','target']

    df.ca = df.ca.astype('category')
    df.thal = df.thal.astype('category')
    df.sex = df.sex.astype('category')
    df.cp = df.cp.astype('category')
    df.fbs = df.fbs.astype('category')
    df.restecg = df.restecg.astype('category')
    df.exang = df.exang.astype('category')
    
    return df

def cleanData():
    filename = 'processed.cleveland.data'
    targetName = 'cleanedProjectData.csv'
    a = open(filename)
    try:
        file = csv.reader(a)
    except:
        print(f'Sorry, input file does not store valid data.')
    with open(targetName, "w+") as tmp:
        for row in file:
            if '?' in row:
                continue
            if int(row[-1]) > 1:
                row[-1] = '1';
            tmp.write(','.join(row)+'\n')
        a.close()