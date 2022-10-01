import torch
from d2l import torch as d2l
from torch import nn


def accuracy(y_hat, y):
    """
    sum of right label predicted
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        idx = y_hat.argmax(axis=1)
    cmp = idx == y
    return float(cmp.sum())


def evaluate_accuracy(net, data_iter):
    """Compute the accuracy for a model on a dataset.
    """
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_acc_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for x, y in data_iter:
            if isinstance(x, list):
                x = [t.to(device) for t in x]
            else:
                x = x.to(device)
            y = y.to(device)
            metric.add(accuracy(net(x), y), y.numel())
    return metric[0] / metric[1]
