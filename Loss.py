import torch
from torch import nn
from torch.nn import functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    # # 当样本数量过大的时候，计算tmp时会导致内存溢出
    # def forward(self, features: torch.Tensor, target: torch.Tensor):
    #     assert features.ndim == 2
    #     # 计算距离
    #     n = features.shape[0]
    #     n_1 = features.unsqueeze(1) # [n, 1, n_f]
    #     n_2 = features.unsqueeze(0) # [1, n, n_f]
    #     tmp = n_1 - n_2
    #     distance = tmp.norm(dim=-1)
    #     # one_hot编码
    #     nc = target.max().item() + 1
    #     target_one_hot = F.one_hot(target, nc)
    #     target_one_hot = target_one_hot.to(torch.float)
    #     # 计算样本之间的匹配程度，若标签相同则为1；反之，则为0。
    #     label_consistency = torch.matmul(target_one_hot, target_one_hot.T)
    #     # 去掉样本自身配对的样本对，即对角线为0
    #     eye_mask = torch.logical_not(torch.eye(n), out=torch.empty((n, n), dtype=torch.float))
    #     label_consistency = label_consistency * eye_mask
    #     mask = label_consistency.to(torch.bool)
    #     # 匹配样本的损失
    #     positive_loss = distance[mask].pow(2).sum()
    #     not_mask = torch.logical_not(mask)
    #     # 去掉样本自身配对的样本对，即对角线为0
    #     not_mask = not_mask & eye_mask.to(torch.bool)
    #     # 不匹配样本的损失
    #     tmp = self.margin - distance[not_mask]
    #     negative_loss = torch.where(tmp>0, tmp, torch.zeros_like(tmp)).pow(2).sum()
    #     # n个样本的匹配样本数，有序
    #     t = n * (n - 1)
    #     return (positive_loss + negative_loss) / (2 * t)

    # 通过逐一计算每一个特征之间的差异来避免同时计算而导致的内存溢出，虽然在面对较大的数据量时也溢出，但优于第一种方法
    def forward(self, features: torch.Tensor, target: torch.Tensor):
        assert features.ndim == 2
        # 计算距离
        n, c = features.shape
        n_1 = features.unsqueeze(1)  # [n, 1, n_f]
        n_2 = features.unsqueeze(0)  # [1, n, n_f]
        # 扩展特征，避免广播导致的内存溢出
        n_1_expand = n_1.expand([-1, n, -1])
        n_2_expand = n_2.expand([n, -1, -1])
        distance = None
        for i in range(c):
            tmp = n_1_expand[..., i] - n_2_expand[..., i]
            tmp = tmp.pow(2)
            distance = tmp if distance is None else distance + tmp
        # one_hot编码
        nc = target.max().item() + 1
        target_one_hot = F.one_hot(target, nc)
        target_one_hot = target_one_hot.to(torch.float)
        # 计算样本之间的匹配程度，若标签相同则为1；反之，则为0。
        label_consistency = torch.matmul(target_one_hot, target_one_hot.T)
        # 去掉样本自身配对的样本对，即对角线为0
        eye_mask = torch.logical_not(torch.eye(n, device=distance.device), out=torch.empty_like(distance))
        label_consistency = label_consistency * eye_mask
        mask = label_consistency.to(torch.bool)
        # 匹配样本的损失
        positive_loss = distance[mask].sum()
        not_mask = torch.logical_not(mask)
        # 去掉样本自身配对的样本对，即对角线为0
        not_mask = not_mask & eye_mask.to(torch.bool)
        # 不匹配样本的损失
        tmp = self.margin - distance[not_mask].sqrt()
        negative_loss = torch.where(tmp > 0, tmp, torch.zeros_like(tmp)).pow(2).sum()
        # n个样本的匹配样本数，有序
        t = n * (n - 1)
        return (positive_loss + negative_loss) / (2 * t)

    # 采用暴力的方法来计算，但是时间较慢
    # def forward(self, features: torch.Tensor, gt: torch.Tensor):
    #     assert features.ndim == 2
    #     gt = gt.to(torch.long)
    #     n, c = features.shape
    #     # 计算每个样本与其它样本的匹配损失
    #     positive_loss, negative_loss = None, None
    #     for i in range(n - 1):
    #         # 剩余样本量
    #         r = n - i - 1
    #         source = features[i].unsqueeze(0)
    #         r_source = source.expand([r, -1])
    #         target = features[i+1:]
    #         diff = r_source - target
    #         d = diff.norm(dim=-1)
    #         source_gt = gt[i]
    #         r_source_gt = source_gt.expand([r, ])
    #         target_gt = gt[i+1:]
    #         mask = r_source_gt == target_gt
    #         positive_loss = d[mask].pow(2).sum() if positive_loss is None \
    #             else positive_loss + d[mask].pow(2).sum()
    #         not_mask = torch.logical_not(mask)
    #         tmp = self.margin - d[not_mask]
    #         v = torch.where(tmp > 0, tmp, torch.zeros_like(tmp)).sum()
    #         negative_loss = v if negative_loss is None else negative_loss + v
    #     return (positive_loss + negative_loss) / (n * (n - 1))

    # 以向量的点成来代替欧式距离
    # def forward(self, features: torch.Tensor, target: torch.Tensor):
    #     assert features.ndim == 2
    #     # 计算距离
    #     n = features.shape[0]
    #     distance = torch.matmul(features, features.T).exp()
    #     # one_hot编码
    #     nc = target.max().item() + 1
    #     target_one_hot = F.one_hot(target, nc)
    #     target_one_hot = target_one_hot.to(torch.float)
    #     # 计算样本之间的匹配程度，若标签相同则为1；反之，则为0。
    #     label_consistency = torch.matmul(target_one_hot, target_one_hot.T)
    #     # 去掉样本自身配对的样本对，即对角线为0
    #     eye_mask = torch.logical_not(torch.eye(n, device=distance.device), out=torch.empty_like(distance))
    #     label_consistency = label_consistency * eye_mask
    #     mask = label_consistency.to(torch.bool)
    #     # 匹配样本的损失
    #     positive_loss = distance[mask].sum()
    #     not_mask = torch.logical_not(mask)
    #     # 去掉样本自身配对的样本对，即对角线为0
    #     not_mask = not_mask & eye_mask.to(torch.bool)
    #     # 不匹配样本的损失
    #     tmp = self.margin - distance[not_mask]
    #     negative_loss = torch.where(tmp>0, tmp, torch.zeros_like(tmp)).sum()
    #     # n个样本的匹配样本数，有序
    #     t = n * (n - 1)
    #     return (positive_loss + negative_loss) / (2 * t)




# positive_loss: 10; negative_loss: 2
# criterion = ContrastiveLoss(1.)
# features = torch.tensor([1., 2., 3., 1.]).reshape((-1, 1))
# target = torch.tensor([0, 0, 1, 1], dtype=torch.long)
# loss = criterion(features, target)
# print(loss)


# 测试运行时间
# torch.manual_seed(666)
# feature = torch.rand(100, 100)
# target = torch.randint(10, (100, ))
# criterion = ContrastiveLoss(1.)
# import time
# import numpy as np
# times = []
# for i in range(1000):
#     begin = time.time()
#     criterion(feature, target)
#     end = time.time()
#     times.append(end - begin)
# print(np.mean(times))
# print(criterion(feature, target))
