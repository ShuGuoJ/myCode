"""
原始图像->超像素分割->图
"""
import numpy as np
import torch
import cv2 as cv
from torch_scatter import scatter


"""
该算法与暴力递归算法相比，能够获得1.5的加速比，而且在极端情况下也能运行。
如如下矩阵：
0 1 2
3 4 5
6 7 8
"""


# 获得节点之间的邻接关系
def get_edge_index(segment):
    # 扩张
    img = segment.astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    expansion = cv.dilate(img, kernel)
    mask = segment == expansion
    mask = np.invert(mask)
    # 构图
    h, w = segment.shape
    edge_index = set()
    directions = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    indices = list(zip(*np.nonzero(mask)))
    for x, y in indices:
        for dx, dy in directions:
            adj_x, adj_y = x + dx, y + dy
            if -1 < adj_x < h and -1 < adj_y < w:
                source, target = segment[x, y], segment[adj_x, adj_y]
                if source != target:
                    edge_index.add((source, target))
                    edge_index.add((target, source))
    return torch.tensor(list(edge_index), dtype=torch.long).T, edge_index


# 获得节点
def get_node(x, segment, mode='mean'):
    assert x.ndim == 3 and segment.ndim == 2
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(segment, np.ndarray):
        segment = torch.from_numpy(segment).to(torch.long)
    c = x.shape[2]
    x = x.reshape((-1, c))
    mask = segment.flatten()
    nodes = scatter(x, mask, dim=0, reduce=mode)
    return nodes


# x = np.arange(4).reshape((2,2,1))
# print(x)
# segment = np.array([[0, 0], [0, 1]])
# print(get_node(x, segment))

