"""
可视化数据
"""
import torch
import numpy as np


# PCA
# def visualize(x, y):
#     x = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
#     y = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
#     from sklearn.decomposition import PCA
#     from matplotlib import pyplot as plt
#     pca = PCA(2, whiten=True)
#     new_x = pca.fit_transform(x)
#     for i in np.unique(y):
#         tmp = new_x[y == i]
#         c = plt.cm.Set1(i)
#         plt.scatter(tmp[:, 0], tmp[:, 1], color=c, marker='*')
#     plt.show()


# PCA
def reduce_dimension(x):
    x_ = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    from sklearn.decomposition import PCA
    pca = PCA(2, whiten=True)
    new_x = pca.fit_transform(x_)
    reduction = torch.tensor(new_x, dtype=x.dtype, device=x.device)
    return reduction


# TSNE
def visualize(x, y):
    x = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    y = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
    from sklearn.manifold import TSNE
    from matplotlib import pyplot as plt
    pca = TSNE(2, random_state=666)
    new_x = pca.fit_transform(x)
    for i in np.unique(y):
        tmp = new_x[y == i]
        c = plt.cm.Set1(i)
        plt.scatter(tmp[:, 0], tmp[:, 1], color=c, marker='*')
    plt.show()



# TSNE
def reduce_dimension_(x):
    x_ = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    from sklearn.manifold import TSNE
    pca = TSNE(2, random_state=666)
    new_x = pca.fit_transform(x_)
    return torch.tensor(new_x, dtype=x.dtype, device=x.device) \
        if isinstance(x, torch.Tensor) else new_x

# from sklearn.datasets import load_iris
# data = load_iris()
# visualize(data.data, data.target)