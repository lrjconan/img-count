import cslab_environ

import numpy as np
import tensorflow as tf
import unittest


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
        y = tf.user_ops.cum_min(x)
        with tf.Session() as sess:
            y = y.eval()
            np.testing.assert_array_almost_equal(y, y_t)

        pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(CumMinTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
