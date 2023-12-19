import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_breast_cancer_data():
    # Load the data set
    column_names = ['id', 'diagnosis', 'radius_mean', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave points', 'symmetry', 'fractal dimensions']
    data = pd.read_csv('wdbc.data', names=column_names, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], delimiter=",")
    x = data.drop(columns=['id', 'diagnosis'])
    y = data['diagnosis']
    return x, y

def clean_data_for_pca(x):
    # Handle missing values (if any)
    x.fillna(x.mean(), inplace=True)

    # Standardize the features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    return x_scaled