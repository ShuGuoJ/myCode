import torch
from torch import nn
from torchvision import models, datasets, transforms
from visdom import Visdom
from torch.utils.data import DataLoader
from Trainer import Trainer
# 加载数据
transform = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root='/home/jsg/dataset', transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='/home/jsg/dataset', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)
# 加载模型
net = models.resnet18()
net.fc = nn.Linear(512, 10)
trainer = Trainer(net)
# 定义损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化器
optimizer = torch.optim.SGD(trainer.get_parameters(), lr=1e-1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
viz = Visdom(port=17000)
viz.line([[0., 0.]], [0], win='train&eval loss', opts={'title': 'train&eval loss',
                                                       'legend': ['train', 'eval']})
viz.line([0.], [0], win='accuracy', opts={'title': 'accuracy'})
for i in range(1000):
    print('*'*5 + str(i) + '*'*5)
    train_loss = trainer.train(train_loader, optimizer, criterion, device)
    eval_loss, acc = trainer.evaluate(test_loader, criterion, device)
    print('epoch: {} train_loss: {:.6f} eval_loss: {:.6f} acc: {:.2%}'.format(i, train_loss, eval_loss, acc))
    viz.line([[train_loss, eval_loss]], [i], win='train&eval loss', update='append')
    viz.line([acc], [i], win='accuracy', update='append')
    print('*'*10)