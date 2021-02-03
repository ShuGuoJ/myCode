import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import optimizer as optimizer_

class Trainer(object):
    r"""模型训练器
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    # 训练过程
    def train(self, data_loader: DataLoader, optimizer: optimizer_, criterion, device: torch.device, monitor=None):
        self.model.train()
        self.model.to(device)
        criterion.to(device)
        losses = []
        grads = []
        for step, (x, target) in enumerate(data_loader):
            x, target = x.to(device), target.to(device)
            logits = self.model(x)  # 前向传播
            loss = criterion(logits, target)  # 计算损失函数值
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 监控器
            if monitor is not None:
                grads.append(monitor())
            if (step+1) % 5 == 0:
                print('batch:{} loss:{:.6f}'.format(step, loss.item()))
            losses.append(loss.item())
        if len(grads) > 0:
            grads_mean = np.mean(grads)
        else:
            grads_mean = None
        return np.mean(losses), grads_mean

    # 验证过程
    def evaluate(self, data_loader: DataLoader, criterion, device: torch.device):
        self.model.eval()
        self.model.to(device)
        criterion.to(device)
        correct = 0
        losses = []
        for x, target in data_loader:
            x, target = x.to(device), target.to(device)
            logits = self.model(x)
            with torch.no_grad():
                loss = criterion(logits, target)
            pred = logits.argmax(-1)
            correct += pred.eq(target).sum().item()
            losses.append(loss.item())
        return np.mean(losses), correct / len(data_loader.dataset)

    # 获得分类结果
    def infer(self, data, device: torch.device):
        self.model.eval()
        self.model.to(device)
        ans = []
        if isinstance(data, torch.Tensor):
            x = data.to(device)
            with torch.no_grad():
                logits = self.model(x)
            pred = logits.argmax(-1)
            ans.append(pred)
        else:
            for x, _ in data:
                x = x.to(device)
                with torch.no_grad():
                    logits = self.model(x)
                pred = logits.argmax(-1)
                ans.append(pred)
        return torch.cat(ans, dim=0)

    def get_parameters(self):
        return self.model.parameters()