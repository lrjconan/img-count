import cslab_environ

from utils import logger
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import unittest

log = logger.get()


@ops.RegisterGradient("CumMin")
def _cum_min_grad(op, grad):
    """The gradients for `cum_min`.

    Args:
      op: The `cum_min` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
      grad: Gradient with respect to the output of the `cum_min` op.

    Returns:
      Gradients with respect to the input of `cum_min`.
    """
    x = op.inputs[0]
    return [tf.user_ops.cum_min_grad(grad, x)]


class CumMinTests(unittest.TestCase):

    def test_1(self):
        x = np.array([
            [1., 2., 0., 4.],
            [2., 1., -1., 3.]
        ])
        y_t = np.array([
            [1., 1., 0., 0.],
            [2., 1., -1., -1.]
        ])

        dy = np.array([
            [1., 2., 3., 4.],
            [5., 6., 7., 8.]
        ])
        dx_t = np.array([
            [3., 0., 7., 0.],
            [5., 6., 15., 0.]
        ])
        y = tf.user_ops.cum_min(x)
        dx = tf.user_ops.cum_min_grad(dy, x)
        with tf.Session() as sess:
            y = y.eval()
            dx = dx.eval()
            np.testing.assert_array_almost_equal(y, y_t)
            np.testing.assert_array_almost_equal(dx, dx_t)

        pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(CumMinTests)
    unittest.TextTestRunner(verbosity=2).run(suite)

    lr = 1e-1
    initial = tf.truncated_normal([2, 4], stddev=0.01)
    w = tf.Variable(initial)
    x = tf.user_ops.cum_min(w)
    y = tf.placeholder('float', [2, 4])
    mse = tf.reduce_mean((y - x) * (y - x))
    train_step = tf.train.AdamOptimizer(lr).minimize(mse)
    # train_step = tf.train.GradientDescentOptimizer(lr).minimize(mse)

    y_val = np.array([
        [1., 1., 0., 0.],
        [2., 1., -1., -1.]
    ])
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        w_val, x_val = sess.run([w, x])
        mse_list = []
        for step in xrange(1000):
            results = sess.run([mse, x, w, train_step], feed_dict={y: y_val})
            mse_val = results[0]
            x_val = results[1]
            w_val = results[2]
            mse_list.append(mse_val)
            log.info('Step: {}, MSE: {}'.format(step, mse_val))

        print 'Y:'
        print y_val
        print
        print
        print 'X:'
        print x_val
        print
        print
        print 'W:'
        print w_val

        plt.plot(np.arange(len(mse_list)), mse_list)

    plt.show()
