import random
import numpy as np
import matplotlib.pyplot as plt

x = np.zeros(100)
y = np.zeros(100)
for i in range(100):
    x[i] = random.random()
    y[i] = random.random()*0.2-0.1+x[i]*x[i]+x[i]+1

a1 = random.random()
a2 = random.random()
a3 = random.random()

h = np.zeros(100)
A = 0.3
for i in range(1000):
    sum_1, sum_2, sum_3 = 0, 0, 0
    for i in range(100):
        h[i] = a1 + a2 * x[i] + a3 * x[i]**2
        sum_1 += (h[i] - y[i])
        sum_2 += (h[i] - y[i]) * x[i]
        sum_3 += (h[i] - y[i]) * x[i]**2
    a1 -= A * (sum_1 / 100)
    a2 -= A * (sum_3 / 100)
    a3 -= A * (sum_3 / 100)
print(a1, a2, a3)
z = np.linspace(0,1,100)
plt.plot(z, a1 + a2 * z + a3 * (z**2), 'r')
plt.scatter(x, y)
plt.show()
