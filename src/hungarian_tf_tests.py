import sys
sys.path.insert(0, '/pkgs/tensorflow-gpu-0.5.0/lib/python2.7/site-packages')
sys.path.insert(0, '/u/mren/code/img-count/third_party/tensorflow/_python_build/')

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
        self.assertTrue((c_0.flatten() == c_0_t.flatten()).all())
        self.assertTrue((c_1.flatten() == c_1_t.flatten()).all())
        self.assertTrue((M == M_t).all())

        pass

    def test_min_weighted_bp_cover_3(self):
        W = np.array([[-5, -3, -4, -4],
                      [-2, -4, -6, -8],
                      [-4, -5, -5, -7]])
        M, c_0, c_1 = tf.user_ops.hungarian(W)
        with tf.Session() as sess:
            M = M.eval()
            c_0 = c_0.eval()
            c_1 = c_1.eval()
        c_0_t = np.array([-3, -3, -5])
        c_1_t = np.array([1, 0, 0, 0])
        M_t = np.array([[0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 1, 0]])
        self.assertTrue((c_0.flatten() == c_0_t.flatten()).all())
        self.assertTrue((c_1.flatten() == c_1_t.flatten()).all())
        self.assertTrue((M == M_t).all())

        pass

    def test_min_weighted_bp_cover_4(self):
        W = np.array([[5, 0, 2],
                      [3, 1, -2],
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
        self.assertTrue((c_0.flatten() == c_0_t.flatten()).all())
        self.assertTrue((c_1.flatten() == c_1_t.flatten()).all())
        self.assertTrue((M == M_t).all())

        pass

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(HungarianTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
