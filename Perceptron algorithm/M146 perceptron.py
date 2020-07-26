import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("data2.csv")
x = df['xs']
y = df['ys']
label = df['label']


def train():
    w = [0.0, 0.0, 0.0]
    iteration = 100000
    i = 0
    k = 0
    updates = 0
    while i < iteration:
        k = k % 100
        lin_comb = w[0] + w[1] * x[k] + w[2] * y[k]
        if label[k] * lin_comb <= 0:
            w[0] = w[0] + label[k]
            w[1] = w[1] + label[k] * x[k]
            w[2] = w[2] + label[k] * y[k]
            print(w)
            updates += 1
        k += 1
        i += 1
    return w, updates


def margin(w):
    print(w)
    i = 0
    minimum = 1000
    while i < 100:
        pred = 0 - w[0] / w[2] - w[1] / w[2] * (x[i])
        minimum = min(minimum, abs(abs(y[i]) - abs(pred)))
        i += 1
    print(minimum)


part1 = df[df.label > 0]
x1 = part1['xs']
y1 = part1['ys']
part2 = df[df.label < 0]
x2 = part2['xs']
y2 = part2['ys']

weights, updates = train()
print(weights)
print(updates)
input = np.linspace(-0.5, 0.6, 100)
line = 0 - weights[0] / weights[2] - weights[1] / weights[2] * input
plt.plot(input, line, '-g')
margin(weights)

plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Visualization for data3.csv')
plt.show()

A = np.array([[2], [3]])
B = np.array([[1, 1], [1, 1]])
print(A*B)
