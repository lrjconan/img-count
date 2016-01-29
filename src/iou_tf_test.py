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
    """
    Gets the list of axes to sum over.
    """
    dim = len(a.get_shape())

    return [dim - 2, dim - 1]


def inter(a, b):
    """
    Computes intersection.

    Args:
        a: [B, N, H, W], or [N, H, W], or [H, W]
        b: [B, M, H, W], or [M, H, W], or [H, W]
    """
    reduction_indices = get_reduction_indices(a)
    return tf.reduce_sum(a * b, reduction_indices=reduction_indices)


def union(a, b):
    """
    Computes union.

    Args:
        a: [B, N, H, W], or [N, H, W], or [H, W]
        b: [B, M, H, W], or [M, H, W], or [H, W]
    """
    reduction_indices = get_reduction_indices(a)
    return tf.reduce_sum(a + b - (a * b), reduction_indices=reduction_indices)


def f_iou(a, b, pairwise=False):
    """
    Computes IOU score.

    Args:
        a: [B, N, H, W], or [N, H, W], or [H, W]
        b: [B, N, H, W], or [N, H, W], or [H, W]
           in pairwise mode, the second dimension can be different,
           e.g. [B, M, H, W], or [M, H, W], or [H, W]
        pariwise: whether the inputs are already aligned, outputs [B, N] or 
                  the inputs are orderless, outputs [B, N, M].
    """
    if pairwise:
        b_shape = tf.shape(b)
        # [1, 1, M, 1, 1]
        a_shape2 = tf.concat(0, [tf.constant([1]), tf.constant([1]), b_shape[1: 2],
                                 tf.constant([1]), tf.constant([1])])
        # [B, N, H, W] => [B, N, 1, H, W] => [B, N, M, H, W]
        a = tf.tile(tf.expand_dims(a, 2), a_shape2)
        # [B, M, H, W] => [B, 1, M, H, W]
        b = tf.expand_dims(b, 1)
    return inter(a, b) / union(a, b)


def f_uoi(a, b, pairwise=False):
    """
    Computes UOI score.

    Args:
        a: [B, N, H, W], or [N, H, W], or [H, W]
        b: [B, N, H, W], or [N, H, W], or [H, W]
           in pairwise mode, the second dimension can be different,
           e.g. [B, M, H, W], or [M, H, W], or [H, W]
        pariwise: whether the inputs are already aligned, outputs [B, N] or 
                  the inputs are orderless, outputs [B, N, M].
    """
    # [B, N, H, W] => [B, N, 1, H, W]
    b_shape = tf.shape(b)
    a_shape2 = tf.concat(0, [tf.constant([1]), tf.constant([1]), b_shape[1: 2],
                             tf.constant([1]), tf.constant([1])])
    a = tf.tile(tf.expand_dims(a, 2), a_shape2)
    # [B, M, H, W] => [B, 1, M, H, W]
    b = tf.expand_dims(b, 1)
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

                avg_iou_val = results[0]
                avg_uoi_val = results[1]
                iou_list.append(avg_iou_val)

                log.info('step: {}, UOI: {:.2f}, IOU: {:.2f}'.format(
                    step, avg_uoi_val, avg_iou_val))

            iou_val = results[2]
            print iou_val, iou_val.shape

            x_val = results[5]
            print x_val[0, 0]
            print y_val[0, 0]
            plt.plot(np.arange(len(iou_list)), iou_list)

    plt.legend(['by UOI', 'by IOU'])
    plt.show()
