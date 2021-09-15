"""
原始图像->超像素分割->图
"""
import numpy as np
import torch
import cv2 as cv
from torch_scatter import scatter
from torch_geometric.data import Data


"""
该算法与暴力递归算法相比，能够获得1.5的加速比，而且在极端情况下也能运行。
如如下矩阵：
0 1 2
3 4 5
6 7 8
"""


# 获得节点之间的邻接关系
def get_edge_index(segment):
    if isinstance(segment, torch.Tensor):
        segment = segment.numpy()
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
    return nodes.to(torch.float32)


# 绘制网格数据中点的4领域关系
# 双重for循环遍历图中每一个节点的4领域'
# def get_grid_adj(grid):
#     DETA = [[-1,0],[1,0],[0,-1],[0,1]]
#     h, w = grid.shape
#     edge_index = list()
#     for i in range(h):
#         for j in range(w):
#             if grid[i, j] == -1:
#                 continue
#             for deta in DETA:
#                 i_adj = i + deta[0]
#                 j_adj = j + deta[1]
#                 if 0 <= i_adj < h and 0 <= j_adj < w and grid[i_adj, j_adj] != -1:
#                     edge_index.append((grid[i, j], grid[i_adj, j_adj]))
#     return edge_index


# 采用图像偏移的方法来构造图中每一个节点的4领域
def get_grid_adj(grid):
    edge_index = list()
    # 上偏移
    a = np.full_like(grid, -1, dtype=np.int32)
    a[:-1] = grid[1:]
    adj = np.stack([grid, a], axis=-1)
    mask = adj != -1
    mask = np.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    # 下偏移
    a = np.full_like(grid, -1, dtype=np.int32)
    a[1:] = grid[:-1]
    adj = np.stack([grid, a], axis=-1)
    mask = adj != -1
    mask = np.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    # 左偏移
    a = np.full_like(grid, -1, dtype=np.int32)
    a[:, :-1] = grid[:, 1:]
    adj = np.stack([grid, a], axis=-1)
    mask = adj != -1
    mask = np.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    # 右偏移
    a = np.full_like(grid, -1, dtype=np.int32)
    a[:, 1:] = grid[:, :-1]
    adj = np.stack([grid, a], axis=-1)
    mask = adj != -1
    mask = np.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    return edge_index


# 获取graph list
def get_graph_list(data, seg):
    graph_node_feature = []
    graph_edge_index = []
    for i in np.unique(seg):
        # 获取节点特征
        graph_node_feature.append(data[seg == i])
        # 获取邻接信息
        x, y = np.nonzero(seg == i)
        n = len(x)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        grid = np.full((x_max - x_min + 1, y_max - y_min + 1), -1, dtype=np.int32)
        x_hat, y_hat = x - x_min, y - y_min
        grid[x_hat, y_hat] = np.arange(n)
        graph_edge_index.append(get_grid_adj(grid))
    graph_list = []
    # 数据变换
    for node, edge_index in zip(graph_node_feature, graph_edge_index):
        node = torch.tensor(node, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        graph_list.append(Data(node, edge_index=edge_index))
    return graph_list


# data = np.arange(12).reshape((4, 3, 1))
# seg = [[0, 0, 2], [0, 1, 2], [1, 1, 2], [1, 2, 2]]
# seg = np.array(seg)
# print(get_graph_list(data, seg))


