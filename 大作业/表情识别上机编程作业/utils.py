import torch
import numpy as np
import matplotlib.pyplot as plt


def PCA(X: torch.Tensor, k: int):  
    """
    X: data
    k: dims which you want  
    """
    X = X.cpu().numpy()
    n_samples, n_features = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    norm_X = X - mean
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    eig_pairs.sort(reverse=True, key=lambda ls: ls[0])
    feature = np.array([elem[1] for elem in eig_pairs[:k]])
    data = np.dot(norm_X, np.transpose(feature))
    return data


def PCA_draw(data, label):
    features = PCA(data, 2)
    label = list(label.cpu().numpy())
    plt.scatter(features[:, 0], features[:, 1], c=label, cmap='rainbow')