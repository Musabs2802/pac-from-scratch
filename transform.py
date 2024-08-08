from transformation import PCA
from sklearn.datasets import load_iris

iris_data = load_iris()
X, y = iris_data.data, iris_data.target

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_pca.shape)