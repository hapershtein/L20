# Linear Discriminant Analysis (LDA) from a Linear Algebra Perspective

Linear Discriminant Analysis (LDA) is a dimensionality reduction technique that is commonly used for supervised classification problems. From a linear algebra perspective, LDA aims to find a linear combination of features that characterizes or separates two or more classes of objects or events. The goal is to project a dataset onto a lower-dimensional space with good class separability to avoid the curse of dimensionality and reduce computational costs.

## Mean Vectors

The first step in LDA is to compute the mean vectors for each class and for the entire dataset.

- **Class Mean Vector (μ_k):** For each class `k`, the mean vector `μ_k` is the average of all the data points belonging to that class. It represents the centroid of the class.

  `μ_k = (1/n_k) * Σ(x_i)` for all `x_i` in class `k`, where `n_k` is the number of samples in class `k`.

- **Overall Mean Vector (μ):** The overall mean vector `μ` is the average of all data points in the dataset.

  `μ = (1/N) * Σ(x_i)` for all `x_i` in the dataset, where `N` is the total number of samples.

## Scatter Matrices

LDA uses two scatter matrices to measure the separability of the classes:

- **Within-Class Scatter Matrix (S_W):** This matrix measures the scatter of data points within each class. It is the sum of the individual scatter matrices for each class. A small `S_W` indicates that the data points within each class are tightly clustered.

  `S_W = Σ(S_k)` for all classes `k`, where `S_k = Σ((x_i - μ_k) * (x_i - μ_k)^T)` for all `x_i` in class `k`.

- **Between-Class Scatter Matrix (S_B):** This matrix measures the scatter of the class means around the overall mean. A large `S_B` indicates that the class means are well-separated.

  `S_B = Σ(n_k * (μ_k - μ) * (μ_k - μ)^T)` for all classes `k`.

## Optimization Problem

The core of LDA is to find a projection matrix `W` that maximizes the ratio of the between-class scatter to the within-class scatter. This is known as Fisher's criterion:

`J(W) = det(W^T * S_B * W) / det(W^T * S_W * W)`

To find the optimal `W`, we solve the generalized eigenvalue problem:

`S_B * w = λ * S_W * w`

This can be rewritten as:

`inv(S_W) * S_B * w = λ * w`

The columns of the optimal projection matrix `W` are the eigenvectors corresponding to the largest eigenvalues `λ` of the matrix `inv(S_W) * S_B`.

## Projection

Once the optimal projection matrix `W` is found, the original data `X` can be projected onto a lower-dimensional space `Y` using the following equation:

`Y = X * W`

The number of dimensions in the new feature space is at most `C-1`, where `C` is the number of classes.

## Summary

In summary, LDA from a linear algebra perspective involves the following steps:

1.  Compute the mean vectors for each class and the overall dataset.
2.  Compute the within-class and between-class scatter matrices.
3.  Solve the generalized eigenvalue problem to find the optimal projection matrix.
4.  Project the data onto a lower-dimensional space using the projection matrix.

The result is a new feature space that maximizes the separability of the classes, making it easier to perform classification.
