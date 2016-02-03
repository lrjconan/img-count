import cslab_environ

import numpy as np
import tensorflow as tf
import unittest


class HungarianTests(unittest.TestCase):
    
    def test_min_weighted_bp_cover_1(self):
        W = np.array([[3, 2, 2],
                      [1, 2, 0],
                      [2, 2, 1]])
        M, c_0, c_1 = tf.user_ops.hungarian(W)
        with tf.Session() as sess:
            M = M.eval()
            c_0 = c_0.eval()
            c_1 = c_1.eval()
        c_0_t = np.array([2, 1, 1])
        c_1_t = np.array([1, 1, 0])
        M_t = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
        print M
        print c_0
        print c_1
        self.assertTrue((c_0.flatten() == c_0_t.flatten()).all())
        self.assertTrue((c_1.flatten() == c_1_t.flatten()).all())
        self.assertTrue((M == M_t).all())

        pass

    def test_min_weighted_bp_cover_2(self):
        W = np.array([[5, 0, 4, 0],
                      [0, 4, 6, 8],
                      [4, 0, 5, 7]])
        M, c_0, c_1 = tf.user_ops.hungarian(W)
        with tf.Session() as sess:
            M = M.eval()
            c_0 = c_0.eval()
            c_1 = c_1.eval()
        c_0_t = np.array([5, 6, 5])
        c_1_t = np.array([0, 0, 0, 2])
        M_t = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        print M
        print c_0
        print c_1
        self.assertTrue((c_0.flatten() == c_0_t.flatten()).all())
        self.assertTrue((c_1.flatten() == c_1_t.flatten()).all())
        self.assertTrue((M == M_t).all())

        pass

    def test_min_weighted_bp_cover_3(self):
        W = np.array([[5, 0, 2],
                      [3, 1, 0],
                      [0, 5, 0]])
        M, c_0, c_1 = tf.user_ops.hungarian(W)
        with tf.Session() as sess:
            M = M.eval()
            c_0 = c_0.eval()
            c_1 = c_1.eval()
        c_0_t = np.array([2, 0, 4])
        c_1_t = np.array([3, 1, 0])
        M_t = np.array([[0, 0, 1],
                        [1, 0, 0],
                        [0, 1, 0]])
        print M
        print c_0
        print c_1
        self.assertTrue((c_0.flatten() == c_0_t.flatten()).all())
        self.assertTrue((c_1.flatten() == c_1_t.flatten()).all())
        self.assertTrue((M == M_t).all())

        pass

    def test_min_weighted_bp_cover_4(self):
        W = np.array([
                      [[5, 0, 2],
                       [3, 1, 0],
                       [0, 5, 0]],

                      [[3, 2, 2],
                       [1, 2, 0],
                       [2, 2, 1]]
                    ])
        M, c_0, c_1 = tf.user_ops.hungarian(W)
        with tf.Session() as sess:
            M = M.eval()
            c_0 = c_0.eval()
            c_1 = c_1.eval()
        c_0_t = np.array([[2, 0, 4], [2, 1, 1]])
        c_1_t = np.array([[3, 1, 0], [1, 1, 0]])
        M_t = np.array([[[0, 0, 1],
                         [1, 0, 0],
                         [0, 1, 0]],
                        [[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]]])
        print M
        print c_0
        print c_1
        self.assertTrue((c_0.flatten() == c_0_t.flatten()).all())
        self.assertTrue((c_1.flatten() == c_1_t.flatten()).all())
        self.assertTrue((M == M_t).all())

        pass


    def test_real_values_1(self):
        # Test the while loop terminates with real values.
        W = np.array(
[[0.90, 0.70, 0.30, 0.20, 0.40, 0.001, 0.001, 0.001, 0.001, 0.001],
 [0.80, 0.75, 0.92, 0.10, 0.15, 0.001, 0.001, 0.001, 0.001, 0.001],
 [0.78, 0.85, 0.66, 0.29, 0.21, 0.001, 0.001, 0.001, 0.001, 0.001],
 [0.42, 0.55, 0.23, 0.43, 0.33, 0.002, 0.001, 0.001, 0.001, 0.001],
 [0.64, 0.44, 0.33, 0.33, 0.34, 0.001, 0.002, 0.001, 0.001, 0.001],
 [0.22, 0.55, 0.43, 0.43, 0.14, 0.001, 0.001, 0.002, 0.001, 0.001],
 [0.43, 0.33, 0.34, 0.22, 0.14, 0.001, 0.001, 0.001, 0.002, 0.001],
 [0.33, 0.42, 0.23, 0.13, 0.43, 0.001, 0.001, 0.001, 0.001, 0.002],
 [0.39, 0.24, 0.53, 0.56, 0.89, 0.001, 0.001, 0.001, 0.001, 0.001],
 [0.12, 0.34, 0.82, 0.82, 0.77, 0.001, 0.001, 0.001, 0.001, 0.001]])
        M, c_0, c_1 = tf.user_ops.hungarian(W)
        with tf.Session() as sess:
            M = M.eval()

        M_t = np.array(
[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])

        print M
        self.assertTrue((M == M_t).all())

        pass

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(HungarianTests)
    # suite = unittest.TestSuite()
    # suite.addTest(HungarianTests('test_real_values'))
    unittest.TextTestRunner(verbosity=2).run(suite)