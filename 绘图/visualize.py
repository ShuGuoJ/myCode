"""
可视化数据
"""
import torch
import numpy as np
import pylab
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import argparse
import os
from scipy.io import loadmat
from sklearn.preprocessing import scale
INFO = {'PaviaU': {'data_key': 'paviaU',
                   'gt_key': 'paviaU_gt'},
        'Salinas': {'data_key': 'salinas_corrected',
                    'gt_key': 'salinas_gt'},
        'KSC': {'data_key': 'KSC',
                'gt_key': 'KSC_gt'},
        'gf5': {'data_key': 'gf5',
                'gt_key': 'gf5_gt'}}


# TSNE
# def visualize(x, y):
#     x = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
#     y = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
#     tsne = TSNE(2, random_state=666)
#     new_x = tsne.fit_transform(x)
#     cm = pylab.get_cmap('tab20')
#     # y从0开始编码
#     t = y.max()
#     for i in np.unique(y):
#         tmp = new_x[y == i]
#         c = cm(1.*i/t)
#         plt.scatter(tmp[:, 0], tmp[:, 1], color=c, marker='*', label=str(i))
#     plt.legend()


# PCA
def visualize(x, y):
    x = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    y = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
    pca = PCA(n_components=2, random_state=666)
    new_x = pca.fit_transform(x)
    cm = pylab.get_cmap('tab20')
    # y从0开始编码
    t = y.max()
    for i in np.unique(y):
        tmp = new_x[y == i]
        c = cm(1.*i/t)
        plt.scatter(tmp[:, 0], tmp[:, 1], color=c, marker='*', label=str(i))
    plt.legend()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='绘画数据集的二维分布')
    parser.add_argument('--root', type=str, default='data',
                        help='ROOT DIRECTORY')
    parser.add_argument('--name', type=str, default='PaviaU',
                        help='DATASET NAME')
    arg = parser.parse_args()
    root = arg.root
    dataset_name = arg.name
    data_key = INFO[dataset_name]['data_key']
    gt_key = INFO[dataset_name]['gt_key']
    # 读取数据
    data_path = '{0}/{1}/{1}.mat'.format(root, dataset_name)
    assert os.path.exists(data_path)
    m = loadmat(data_path)
    data = m[data_key]
    data = data.astype(np.float)
    h, w, c = data.shape
    data = data.reshape((h*w, c))
    data = scale(data)
    # 读取ground truth
    gt_path = '{0}/{1}/{1}_gt.mat'.format(root, dataset_name)
    assert os.path.exists(gt_path)
    m = loadmat(gt_path)
    gt = m[gt_key]
    gt = gt.astype(np.int32)
    gt = gt.flatten()
    indices = np.nonzero(gt)
    plt.figure(figsize=(8, 6))
    visualize(data[indices], gt[indices])
    plt.savefig('{}.pdf'.format(dataset_name), dpi=1)
    plt.clf()
    # plt.show()
