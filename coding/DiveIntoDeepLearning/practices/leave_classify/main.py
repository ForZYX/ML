import torch
from torch.utils.data import DataLoader
from torch import nn
from MyDataset import MyDataset
from Net import Model
import d2l.torch as d2l
import utils
import torchvision.models as models

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


epochs, batch_size = 100, 4

train_iter = DataLoader(MyDataset(), batch_size=batch_size, shuffle=True)
test_iter = DataLoader(MyDataset(train=False), batch_size=batch_size, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Model()
model.apply(init_weights)
model.to(device)

animator = d2l.Animator(xlabel='epoch', xlim=[1, epochs],
                        legend=['train loss', 'train acc', 'test acc'])#动画需要

optim = torch.optim.SGD(model.parameters(), lr=0.01)

loss = nn.CrossEntropyLoss()

timer, num_batches = d2l.Timer(), len(train_iter)
for epoch in range(epochs):
    metric = d2l.Accumulator(3)
    model.train()
    for i, (x, y) in enumerate(train_iter):
        timer.start()

        optim.zero_grad()
        x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.long)
        y_pred = model(x)
        l = loss(y_pred, y)
        l.backward()
        optim.step()

        with torch.no_grad():
            metric.add(l * x.shape[0], utils.accuracy(y_pred, y), x.shape[0])  # 训练损失之和，训练准确率之和，范例数
        timer.stop()
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            animator.add(epoch + (i + 1) / num_batches,
                         (train_l, train_acc, None))
    test_acc = utils.evaluate_acc_gpu(model, test_iter)  # 评估测试集的精度
    animator.add(epoch + 1, (None, None, test_acc))
print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
      f'test acc {test_acc:.3f}')
print(f'{metric[2] * epochs / timer.sum():.1f} examples/sec '
      f'on {str(device)}')
