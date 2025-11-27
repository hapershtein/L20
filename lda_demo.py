import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_data():
    np.random.seed(0)
    # Generate three classes of 2D data
    mean1 = np.array([0, 0])
    cov1 = np.array([[1, 0.5], [0.5, 1]])
    class1_data = np.random.multivariate_normal(mean1, cov1, 100)

    mean2 = np.array([5, 5])
    cov2 = np.array([[1, -0.5], [-0.5, 1]])
    class2_data = np.random.multivariate_normal(mean2, cov2, 100)

    mean3 = np.array([0, 5])
    cov3 = np.array([[1, 0.5], [0.5, 1]])
    class3_data = np.random.multivariate_normal(mean3, cov3, 100)

    return class1_data, class2_data, class3_data

def lda_demo():
    # 1. Generate random data
    class1_data, class2_data, class3_data = generate_data()
    all_data = np.concatenate((class1_data, class2_data, class3_data), axis=0)

    # 2. Compute mean vectors
    mean_class1 = np.mean(class1_data, axis=0)
    mean_class2 = np.mean(class2_data, axis=0)
    mean_class3 = np.mean(class3_data, axis=0)
    overall_mean = np.mean(all_data, axis=0)

    # 3. Compute scatter matrices
    # Within-class scatter matrix
    s_w = np.zeros((2, 2))
    for data, mean in [(class1_data, mean_class1), (class2_data, mean_class2), (class3_data, mean_class3)]:
        class_scatter = np.zeros((2, 2))
        for row in data:
            row, mean = row.reshape(2, 1), mean.reshape(2, 1)
            class_scatter += (row - mean).dot((row - mean).T)
        s_w += class_scatter

    # Between-class scatter matrix
    s_b = np.zeros((2, 2))
    for n, mean in [(len(class1_data), mean_class1), (len(class2_data), mean_class2), (len(class3_data), mean_class3)]:
        mean_vec_diff = (mean.reshape(2, 1) - overall_mean.reshape(2, 1))
        s_b += n * (mean_vec_diff).dot(mean_vec_diff.T)

    # 4. Solve the generalized eigenvalue problem
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(s_w).dot(s_b))

    # Sort eigenvectors by decreasing eigenvalues
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    # Choose the top eigenvector as the projection matrix
    w = eig_pairs[0][1].reshape(2, 1)

    # 5. Project the data
    projected_data = all_data.dot(w)

    # Plotting
    # 2D Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(class1_data[:, 0], class1_data[:, 1], alpha=0.8, label='Class 1')
    plt.scatter(class2_data[:, 0], class2_data[:, 1], alpha=0.8, label='Class 2')
    plt.scatter(class3_data[:, 0], class3_data[:, 1], alpha=0.8, label='Class 3')
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(projected_data[:100], bins=30, alpha=0.7, label='Class 1')
    plt.hist(projected_data[100:200], bins=30, alpha=0.7, label='Class 2')
    plt.hist(projected_data[200:], bins=30, alpha=0.7, label='Class 3')
    plt.title('Projected Data (LDA)')
    plt.xlabel('LD1')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lda_demo.png')

    # 3D Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(class1_data[:, 0], class1_data[:, 1], projected_data[:100], label='Class 1')
    ax.scatter(class2_data[:, 0], class2_data[:, 1], projected_data[100:200], label='Class 2')
    ax.scatter(class3_data[:, 0], class3_data[:, 1], projected_data[200:], label='Class 3')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Projected Value (LD1)')
    ax.set_title('3D View of LDA Projection')
    ax.legend()

    ax.text2D(0.05, 0.95, f'Projection Matrix (W):\n{w}', transform=ax.transAxes,
              fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('lda_demo_3d.png')
    plt.show()

if __name__ == '__main__':
    lda_demo()