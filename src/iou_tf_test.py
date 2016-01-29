"""
Simple experiments on optimizing IOU directly.
Experiments on optimizing UOI and IOU. It turns out that if we optimize using
Adam then there is no difference.

Author: Mengye Ren (mren@cs.toronto.edu)
"""

from utils import logger
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

log = logger.get()


def get_reduction_indices(a):
    if len(a.get_shape()) == 4:
        reduction_indices = [2, 3]
    elif len(a.get_shape()) == 3:
        reduction_indices = [1, 2]
    elif len(a.get_shape()) == 2:
        reduction_indices = [0, 1]

    return reduction_indices


def inter(a, b):
    """
    Computes intersection.

    Args:
        a: [B, N, H, W]
        b: [B, M, H, W]
    """
    reduction_indices = get_reduction_indices(a)
    return tf.reduce_sum(a * b, reduction_indices=reduction_indices)


def union(a, b):
    """
    Computes union.

    Args:
        a: [B, N, H, W]
        b: [B, M, H, W]
    """
    reduction_indices = get_reduction_indices(a)
    return tf.reduce_sum(a + b - (a * b), reduction_indices)


def f_iou(a, b):
    return inter(a, b) / union(a, b)


def f_uoi(a, b):
    return union(a, b) / inter(a, b)

if __name__ == '__main__':
    lr = 1e-1
    for use_uoi in [True, False]:
        log.info('Optimize: {}'.format('UOI' if use_uoi else 'IOU'))
        initial = tf.truncated_normal([3, 10, 5, 5], stddev=0.01)
        w = tf.Variable(initial)
        x = tf.sigmoid(w)
        y = tf.placeholder('float', [3, 10, 5, 5])
        iou = f_iou(x, y)
        uoi = f_uoi(x, y)
        avg_uoi = tf.reduce_mean(uoi)
        avg_iou = tf.reduce_mean(iou)

        if use_uoi:
            train_step = tf.train.AdamOptimizer(lr).minimize(avg_uoi)
        else:
            train_step = tf.train.AdamOptimizer(lr).minimize(-avg_iou)

        random = np.random.RandomState(2)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            iou_list = []
            for step in xrange(1000):

                random = np.random.RandomState(2)
                y_val = np.round(random.uniform(0, 1, [3, 10, 5, 5]))
                results = sess.run([avg_iou, avg_uoi, iou, uoi, train_step, x], feed_dict={
                    y: y_val})
                iou_list.append(results[0])
                log.info('step: {}, UOI: {:.2f}, IOU: {:.2f}'.format(
                    step, results[1], results[0]))

            print results[2], results[2].shape
            print results[5][0, 0]
            print y_val[0, 0]
            plt.plot(np.arange(len(iou_list)), iou_list)

    plt.legend(['by UOI', 'by IOU'])
    plt.show()
