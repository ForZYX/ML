import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def loadData():
    data = pd.read_excel('..//data//iris.xlsx', sheet_name='dataset')
    data = data[['sepal length', 'sepal width', 'class']].iloc[:100]

    x = data._drop_axis('class', 1)
    y = np.array(data['class'])
    # 数据标准化处理
    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(x)
    return X, y


def logisticRegression(trainData, trainLabel, iters=200):
    ones = np.ones(trainData.shape[0])
    trainData = np.insert(trainData, 2, ones, axis=1)
    w = np.zeros(trainData.shape[1])

    h = 0.001

    for i in range(iters):
        for j in range(trainData.shape[0]):
            wx = np.dot(w, trainData[j])
            yi = trainLabel[j]
            xi = trainData[j]
            w += h * (xi * yi - (np.exp(wx) * xi) / (1 + np.exp(wx)))
    return w


def predict(w, x):
    wx = np.dot(w, x)
    p1 = np.exp(wx) / (1 + np.exp(wx))

    if p1 >= 0.5:
        return 1
    else:
        return 0


def draw(x, w):
    fig = plt.figure()
    ax = Axes3D(fig)
    X1 = np.linspace(-3, 3, 1000)
    X2 = np.linspace(-3, 3, 1000)
    x1, x2 = np.meshgrid(X1, X2)
    h = w[2] + w[0] * x1 + w[1] * x2
    ax.plot_surface(x1, x2, h, cmap=plt.cm.jet)
    ax.scatter(x[:50, 0], x[:50, 1], 0, marker='*', c='red', label='Iris-setosa')
    ax.scatter(x[50:, 0], x[50:, 1], 1, marker=',', c='blue', label='Iris-versicolor')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    x, y = loadData()
    plt.scatter(x[:50, 0], x[:50, 1], marker='v', c='red', label='Iris-setosa')
    plt.scatter(x[50:, 0], x[50:, 1], marker=',', c='blue', label='Iris-versicolor')
    plt.legend(loc='upper right')
    plt.show()
    w = logisticRegression(x, y)
    draw(x, w)
