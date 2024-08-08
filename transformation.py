# PCA finds a new set of dimensions such that all the dimensions are orthogonal (linearly independent) and ranked according to the variance of data along them
# The transformed features are linearly independent
# Dimensionality can be reduced by taking only the dimensions with highest importance
# Those newly found dimensions should minimize the projection error
# The projected points should have maximum spread (variance)
#
# Eigenvectors and Eigenvalues
# The eigenvectors point in the direction of the maximum variance, and the corresponding eigenvalues indicate the importance of its corresponding eigenvector
#
# Steps
# Subtract the mean from X
# Calculate cov(X, X)
# Calculate eigenvectors and eigenvalues of covariance matrix
# Sort the eigenvectors according to their eigenvalues in decreasing order
# Choose first k eigenvectors and that will be the new k dimensions
# Transform the original n-dimensional data points into k-dimensions

import numpy as np

class PCA:
    def __init__(self, n_components) -> None:
        self.n_components = n_components
        self.components = None
    
    def fit_transform(self, X):
        X = X - np.mean(X, axis=0)
        cov = np.cov(X.T)

        eigenvectors, eigenvalues = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T

        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[: self.n_components]

        return np.dot(X, self.components.T)