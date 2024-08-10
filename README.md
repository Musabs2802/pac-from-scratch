# Principal Component Analysis (PCA) from Scratch

## What is PCA?
Principal Component Analysis (PCA) is a dimensionality reduction technique used to transform a large set of variables into a smaller one that still contains most of the information in the original set. It achieves this by finding new variables, called principal components, which are linear combinations of the original variables and are orthogonal to each other. The first principal component accounts for the most variance in the data, with each subsequent component accounting for the remaining variance under the constraint of being orthogonal to the previous components.

## Properties of PCA
- **Orthogonality**: Principal components are orthogonal (uncorrelated) to each other.
- **Variance Maximization**: The first principal component captures the maximum variance in the data, with each subsequent component capturing the remaining variance.
- **Dimensionality Reduction**: PCA reduces the dimensionality of the data while retaining most of the variation present in the dataset.
- **Feature Extraction**: PCA generates new features (principal components) that are linear combinations of the original features.

## How to Calculate PCA from Scratch
PCA seeks to find the principal components by solving the eigenvalue problem. The steps to derive the principal components are as follows:

### Step 1: Standardize the Data
Standardize the dataset $$\ X \$$ to have a mean of zero and a standard deviation of one for each feature:

$$\[
X_{\text{standardized}} = \frac{X - \mu}{\sigma}
\]$$

### Step 2: Compute the Covariance Matrix
Compute the covariance matrix of the standardized data:

$$\[
\Sigma = \frac{1}{n-1} X_{\text{standardized}}^T X_{\text{standardized}}
\]$$

### Step 3: Perform Eigenvalue Decomposition
Compute the eigenvalues and eigenvectors of the covariance matrix by solving the equation:

$$\[
\Sigma v_i = \lambda_i v_i
\]$$

### Step 4: Sort Eigenvalues and Eigenvectors
Sort the eigenvalues in descending order and sort the eigenvectors accordingly:
$$\[
\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_p
\]$$

Correspondingly, the sorted eigenvectors:
$$\[
v_1, v_2, \ldots, v_p
\]$$

### Step 5: Select Top $$\ k \$$ Principal Components
Select the top $$\ k \$$ eigenvectors to form the projection matrix $$\ W \$$:

$$\[
W = [v_1, v_2, \ldots, v_k]
\]$$

### Step 6: Transform the Data
Transform the original data to the new subspace using the projection matrix $$\ W \$$:

$$\[
X' = X_{\text{standardized}} W
\]$$

## When to Use PCA

- **High-Dimensional Data**: Useful for reducing the number of features in datasets with many variables.
- **Noise Reduction**: Helps filter out noise by focusing on the principal components that capture the most variance.
- **Visualization**: Assists in visualizing high-dimensional data in 2D or 3D plots.
- **Feature Extraction**: Creates new features that better represent the data.
- **Preprocessing for Machine Learning**: Improves the performance and training time of machine learning algorithms.

## Advantages of PCA

- **Dimensionality Reduction**: Simplifies models and reduces computational cost by preserving most of the data's variance.
- **Noise Filtering**: Improves model performance by focusing on components with higher variance.
- **Improved Visualization**: Helps in understanding high-dimensional data by reducing it to 2D or 3D.
- **Feature Orthogonality**: Produces uncorrelated features, beneficial for algorithms sensitive to multicollinearity.
- **Speed and Efficiency**: Speeds up machine learning algorithms and makes them more efficient.

## Disadvantages of PCA

- **Loss of Interpretability**: The new principal components are harder to interpret in terms of the original variables.
- **Variance-Based**: May lead to suboptimal results by focusing on variance instead of relevance for the predictive task.
- **Linear Assumption**: May not capture complex nonlinear relationships effectively.
- **Data Centering Required**: Requires data to be centered around the mean, which might not be suitable for all data types.
- **Sensitive to Scaling**: Results can be significantly influenced by the scaling of features.

