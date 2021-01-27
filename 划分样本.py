"""
该代码主要用于对高光谱数据集中的样本进行划分
"""
import random
import numpy as np


def train_test_split(gt: np.ndarray, size: int):
    """
    :param gt: 高光谱图像的groundtruth
    :param size: 每类样本的大小
    :return: train_gt(训练样本标签), test_gt(测试样本标签)
    """
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    test_gt[:] = gt[:]
    for i in np.unique(gt):
        if i != 0:
            indices = list(zip(*np.nonzero(gt == i)))
            samples = random.sample(indices, size)
            indices = list(zip(*indices))
            train_gt[indices] = gt[indices]
            test_gt[indices] = 0
    return train_gt, test_gt