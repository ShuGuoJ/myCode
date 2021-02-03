import torch
from torch.nn import functional as F


# 根据中间的簇分配矩阵生成最终的簇分配矩阵
def assgin_cluster(clusters: list):
    # 将簇分配概率分布转换为one-hot
    clusters_one_hot = []
    for cluster in clusters:
        c = cluster.shape[-1]
        indices = cluster.argmax(-1)
        cluster_one_hot = F.one_hot(indices, c).to(torch.float).to(cluster.device)
        clusters_one_hot.append(cluster_one_hot)
    # 簇分配矩阵连乘
    ans = clusters_one_hot[0]
    for i in range(1, len(clusters_one_hot)):
        ans = torch.matmul(ans, clusters_one_hot[i])
    return ans


# cluster_1 = torch.tensor([[[0.3, 0.2, 0.5], [0.6, 0.3, 0.1], [0.1, 0.7, 0.2]]])
# # cluster_2 = torch.tensor([[[0.4, 0.6],[0.7, 0.3],[0.8, 0.2]]])
# # print(assgin_cluster([cluster_1]))