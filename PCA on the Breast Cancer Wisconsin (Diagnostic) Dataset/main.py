# Import libraries
import pandas as pd  # For data manipulation
from clean_data import clean_data_for_pca  # Functions for data cleaning
from perform_pca import perform_pca_with_eigen  # Function for PCA
from visualize_data import visualize_data  # Function for data visualization

# Load the data set
def load_breast_cancer_data():
    column_names = ['id', 'diagnosis', 'radius_mean', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave points', 'symmetry', 'fractal dimensions']
    data = pd.read_csv('wdbc.data', names=column_names, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], delimiter=",")
    y = data['diagnosis']
    x = data.drop(columns=['id', 'diagnosis'])
    return x, y
  
def main():
    # Load the data set
    x, y = load_breast_cancer_data()

    # Clean the data set
    x_cleaned_for_pca = clean_data_for_pca(x)

    # Perform PCA
    principal_components = perform_pca_with_eigen(x_cleaned_for_pca)
    
    # Visualize PCA results
    visualize_data(principal_components, y.values)

if __name__ == '__main__':
    main()