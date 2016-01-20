from utils import logger
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os

log = logger.get()

im_width = 28
im_height = 28
filter_size = 10


def sel_attn(img, g_x, g_y, delta, lg_var):
    # [W, 1]
    span_x = np.arange(im_width).reshape([-1, 1])
    # [H, 1]
    span_y = np.arange(im_height).reshape([-1, 1])
    # [1, F]
    span_f = np.arange(filter_size).reshape([1, -1])

    # [1, F]
    mu_x = g_x + delta * (span_f - filter_size / 2.0 - 0.5)
    log.info('Mu x: {}'.format(mu_x))
    # [1, F]
    mu_y = g_y + delta * (span_f - filter_size / 2.0 - 0.5)
    log.info('Mu y: {}'.format(mu_y))

    # [W, F]
    filter_x = (
        1 / np.sqrt(np.exp(lg_var)) / np.sqrt(2 * np.pi) *
        np.exp(-0.5 * (span_x - mu_x) * (span_x - mu_x) /
               np.exp(lg_var)))
    # [H, F]
    filter_y = (
        1 / np.sqrt(np.exp(lg_var)) / np.sqrt(2 * np.pi) *
        np.exp(-0.5 * (span_y - mu_y) * (span_y - mu_y) /
               np.exp(lg_var)))

    read = filter_y.transpose().dot(img).dot(filter_x)

    return read


if not os.path.exists('img.npy'):
    from data_api import mnist
    dataset = mnist.read_data_sets("../MNIST_data/", one_hot=True)
    img = dataset.train.images[0].reshape([im_height, im_width])
    np.save('img.npy', img)
else:
    img = np.load('img.npy')
    # Controller variables.
    num_img = 8
    f, axarr = plt.subplots(2, num_img)

    for ii in xrange(num_img):
        gg_x = 0.0
        gg_y = 0.0
        ddelta = 0.1 * (ii + 1)
        log.info('ddelta: {}'.format(ddelta))
        lg_var = 0.01

        g_x = (gg_x + 1) * (im_width + 1) / 2.0
        g_y = (gg_y + 1) * (im_height + 1) / 2.0
        delta = (max(im_width, im_height) - 1) / \
            float(filter_size - 1) * ddelta
        log.info('delta: {}'.format(delta))

        read = sel_attn(img, g_x, g_y, delta, lg_var)

        axarr[0, ii].imshow(img, cmap=cm.Greys_r)
        axarr[0, ii].add_patch(
            patches.Rectangle(
                (g_x - delta * (filter_size - 1) / 2.0,
                 g_y - delta * (filter_size - 1) / 2.0),
                delta * (filter_size - 1),
                delta * (filter_size - 1),
                fill=False,
                color='r')
        )
        axarr[0, ii].set_axis_off()
        axarr[1, ii].imshow(read, cmap=cm.Greys_r)
        axarr[1, ii].set_axis_off()

    plt.show()
