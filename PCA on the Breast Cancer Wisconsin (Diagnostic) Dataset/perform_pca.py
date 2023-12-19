import numpy as np

def perform_pca_with_eigen(x_cleaned_for_pca):
    # Calculate the covariance matrix
    cov_matrix = np.cov(x_cleaned_for_pca.T)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Choose the number of components (e.g., 2)
    num_components = 2
    selected_eigenvectors = eigenvectors[:, :num_components]

    # Project the data onto the selected principal components
    principal_components = x_cleaned_for_pca.dot(selected_eigenvectors)

    return principal_components


# As a note, the scikit-learn library provides a PCA class that can be used to perform PCA.