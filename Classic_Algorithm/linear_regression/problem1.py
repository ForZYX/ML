import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

data = pd.read_excel('../data/experimentdata.xlsx')


def SingleLinearRegression(x, y):
    plt.scatter(x, y)

    w = np.sum(y * (x - np.mean(x))) / (np.sum(x * x) - 1 / len(x) * ((np.sum(x)) ** 2))
    b = 1 / len(x) * np.sum(y - w * x)

    print('Function expression is: y = {}x + {}'.format(w, b))
    print('Error square sum is ', np.sum((w * x + b - y) ** 2))

    z = np.linspace(0, 500, 1000)

    plt.title('Function expression is: y = {:.5e}x + {:.5e}\nError square sum is: {:.5e}'.format
              (float(w), float(b), float(np.sum((w * x + b - y) ** 2))))
    plt.plot(z, (w * z + b), color='red')
    plt.show()


def polyRegression(x, y, n):
    plt.scatter(x, y)
    coef = np.zeros((n+1, n+1))
    result = np.zeros((n+1, 1))
    for i in range(n+1):
        result[i] = np.sum(y * (x ** i))
    coef[0][0] = n
    for i in range(n+1):
        for j in range(i, n+1):
            if i == 0 and j == 0:
                coef[0][0] = n
            else:
                coef[i][j] = np.sum(x ** (i+j))
                coef[j][i] = coef[i][j]
    ans = linalg.solve(coef, result)
    expression = 'y = {:.4e}'.format(ans[0][0])
    for i in range(1, n+1):
        expression += '+({:.4e})x^{}'.format(ans[i][0], i)
    print(expression)
    pre = np.zeros(y.shape)
    for i in range(n+1):
        pre += ans[i] * (x**i)
    error_square_sum = np.sum((pre - y) ** 2)
    print('Error square sum is {:.4e}'.format(error_square_sum))

    z = np.linspace(0, 500, 1000).reshape(-1, 1)
    pre = np.zeros(z.shape)
    for i in range(n + 1):
        pre += ans[i] * (z**i)

    plt.title(expression+'\nError square sum is {:.4e}'.format(error_square_sum))
    # draw one point's acceleration
    # x = 200
    # y = ans[0][0] + x * ans[1][0] + (x**2) * ans[2][0] + (x**3) * ans[3][0]
    # y1 = ans[1][0] + 2 * x * ans[2][0] + 3 * (x**2) * ans[3][0]
    # y1 = round(y1, 9)
    # plt.annotate(text=y1, xy=(200, y), xytext=(300, 0.2), weight='bold', color='black',
    #              arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red'),
    #              bbox=dict(boxstyle='round,pad=0.5', fc='blue', ec='k', lw=1, alpha=0.4))
    plt.plot(z, pre, color='red')
    plt.show()


def logRegression(x, y):
    plt.scatter(x, y)

    x = np.reshape(x, (1, -1))[0]
    y = np.reshape(y, (1, -1))[0]
    coef = np.polyfit(np.log(x), y, 1)
    print(coef)

    print('Function expression is: y = {}log x + ({})'.format(coef[0], coef[1]))
    print('Error square sum is ', np.sum(((coef[0]*np.log(x) + coef[1]) - y) ** 2))

    z = np.linspace(0, 500, 1000).reshape(-1, 1)
    plt.title('Function: y = {:.5e}log x + ({:.5e})\nError square sum is: {:.5e}'.format(
        float(coef[0]), float(coef[1]), np.sum(((coef[0]*np.log(x)+coef[1])-y)**2)))
    plt.plot(z, coef[0]*np.log(z) + coef[1], color='red')
    plt.show()


def indexRegression(x, y):
    plt.scatter(x, y)
    y[0] += 0.0000001

    x = np.reshape(x, (1, -1))[0]
    y = np.reshape(y, (1, -1))[0]
    coef = np.polyfit(x, np.log(y), 1)
    print(coef)

    print('Function expression is: y = e^({}x+{})'.format(coef[0], coef[1]))
    print('Error square sum is ', np.sum((np.exp(coef[0] * x + coef[1]) - y) ** 2))

    z = np.linspace(0.5, 500, 1000).reshape(-1, 1)
    plt.title('Function: y = e^({:.5e}x+{:.5e})\nError square sum is: {:.5e}'.format(
        float(coef[0]), float(coef[1]), np.sum((np.exp(coef[0] * x + coef[1]) - y) ** 2)))
    plt.plot(z, np.exp(coef[0] * z + coef[1]), color='red')
    plt.show()


if __name__ == '__main__':
    x = np.array(data['时间']).reshape(-1, 1)
    y = np.array(data['速度']).reshape(-1, 1)
    SingleLinearRegression(x, y)
    # polyRegression(x, y, 3)
    # logRegression(x, y)
    # indexRegression(x, y)
