import numpy as np
import tensorflow as tf
import unittest


class MaxBPMatchTests(unittest.TestCase):

    def test_max_bp_match_1(self):
        G = np.array([[0, 1, 1],
                      [0, 0, 1],
                      [0, 0, 0]])
        with tf.Session() as sess:
          M = tf.user_ops.max_bipartite_matching(G).eval()
        Mt = np.array([[0, 1, 0],
                       [0, 0, 1],
                       [0, 0, 0]])
        self.assertTrue((M == Mt).all())

        pass

if __name__ == '__main__':
    # unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(MaxBPMatchTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
