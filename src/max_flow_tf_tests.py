import numpy as np
import tensorflow as tf
import unittest


class MaxFlowTests(unittest.TestCase):

    def test_max_flow_1(self):
        # x1 -> y1
        # x2 -> y1
        # x2 -> y2
        #             [s  x1 x2 y1 y2 t]
        G = np.array([[0, 1, 2, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 0, 0, 2],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0]])

        with tf.Session() as sess:
            F = tf.user_ops.max_flow(G).eval()

        self.assertTrue((F == G).all())

        pass

    def test_max_flow_2(self):
        # s  -3-> v1
        # s  -3-> v2
        # v1 -3-> v3
        # v1 -2-> v2
        # v2 -2-> v4
        # v3 -4-> v4
        # v3 -2-> t
        # v4 -3-> t
        #             [s  v1 v2 v3 v4 t]
        G = np.array([[0, 3, 3, 0, 0, 0],
                      [0, 0, 2, 3, 0, 0],
                      [0, 0, 0, 0, 2, 0],
                      [0, 0, 0, 0, 4, 2],
                      [0, 0, 0, 0, 0, 3],
                      [0, 0, 0, 0, 0, 0]])
        G = np.zeros([6, 5])

        with tf.Session() as sess:
            F = tf.user_ops.max_flow(G).eval()

        Ft = np.array([[0, 3, 2, 0, 0, 0],
                       [0, 0, 0, 3, 0, 0],
                       [0, 0, 0, 0, 2, 0],
                       [0, 0, 0, 0, 1, 2],
                       [0, 0, 0, 0, 0, 3],
                       [0, 0, 0, 0, 0, 0]])

        self.assertTrue((F == Ft).all())

if __name__ == '__main__':
    # unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(MaxFlowTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
