"""
Simple experiments on optimizing IOU directly.
Experiments on optimizing UOI and IOU. As reported in [1], it turns out that 
optimizing UOI is much faster.

Author: Mengye Ren (mren@cs.toronto.edu)

Reference:
[1] M. Cogswell, X. Lin, S. Purushwalkam, D. Batra. Combining the best of 
graphical models and ConvNets for semantic segmentation. arXiv preprint 
arXiv:1412.4313, 2014.
"""

from utils import logger
import matplotlib.pyplot as plt
import numpy as np

log = logger.get()


def sigmoid(x, eps=1e-7):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(dl, x):
    s = sigmoid(x)
    return dl * s * (1 - s)


def inter(a, b):
    return (a * b).sum()


def inter_grad(dl, a, b):
    return dl * b, dl * a


def union(a, b):
    return (a + b - a * b).sum()


def union_grad(dl, a, b):
    return dl * (np.ones(b.shape) - b), dl * (np.ones(a.shape) - a)


def divide(i, u):
    return i / u


def uoi(i, u):
    return u / i


def divide_grad(i, u):
    return 1 / u, -i / (u * u)


def optimize(x, y, use_uoi=True, num_steps=2000):
    uoi = []
    iou = []
    for step in xrange(num_steps):
        z = sigmoid(x)
        i = inter(z, y)
        u = union(z, y)
        uoi.append(divide(u, i))
        iou.append(divide(i, u))

        log.info('step: {}, UOI: {:.2f}, IOU: {:.2f}'.format(
            step, uoi[-1], iou[-1]))
        if use_uoi:
            du, di = divide_grad(u, i)
        else:
            di, du = divide_grad(i, u)
        diz, diy = inter_grad(di, z, y)
        duz, duy = union_grad(du, z, y)

        z = diz + duz
        dx = sigmoid_grad(z, x)
        if use_uoi:
            x -= lr * dx
        else:
            x += lr * dx

    return x, uoi, iou


if __name__ == '__main__':
    y = np.eye(5)
    lr = 1e-1

    for use_uoi in [True, False]:
        random = np.random.RandomState(2)
        x = random.uniform(-1, 1, [5, 5])
        x, uoi, iou = optimize(x, y, use_uoi=use_uoi, num_steps=10000)
        plt.plot(np.arange(len(iou)), iou)

    plt.legend(['by UOI', 'by IOU'])
    plt.show()
