import cslab_environ

import numpy as np
import tensorflow as tf
import unittest


class BatchIouTests(unittest.TestCase):

    def test_1(self):
        pred = np.array([[
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]]
        ]])
        gt = np.array([[
            [[1.0, 1.0], [0.0, 1.0]]
        ]])
        score_t = np.array([[[2.0 / 3.0], [1.0 / 4.0]]])
        score = tf.user_ops.batch_iou(pred, gt)
        with tf.Session() as sess:
            score = score.eval()
            np.testing.assert_array_almost_equal(score, score_t)

        pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(BatchIouTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
