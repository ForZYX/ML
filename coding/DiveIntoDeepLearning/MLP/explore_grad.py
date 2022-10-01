import torch

num_input, num_hidden, num_output = 784, 128, 1

lr, batch_size, num_epochs = 0.1, 1, 2

w0 = torch.normal(0, 0.01, size=(num_input, num_hidden), requires_grad=True)
b0 = torch.zeros((num_hidden,), requires_grad=True)

w1 = torch.normal(0, 0.01, size=(num_hidden, num_output), requires_grad=True)
b1 = torch.zeros((num_output,), requires_grad=True)


def loss(y_pred, y):
    return (y_pred - y.reshape(y_pred.shape)) ** 2 / 2


def net(X):
    t = X @ w0 + b0
    return t @ w1 + b1


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


x = torch.normal(0, 1, (1, 784))
y = torch.tensor([1])

for epoch in range(num_epochs):
    y_pred = net(x)
    l = loss(y_pred, y)
    l.sum().backward()
    sgd([w0, b0, w1, b1], lr, batch_size)

l = loss(net(x), y)