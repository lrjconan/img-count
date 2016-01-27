import cslab_environ

import numpy as np
import rec_ins_segm as model_lib
import tensorflow as tf
import unittest


class RecurrentInstanceSegmentationTests(unittest.TestCase):

    def test_ins_segm_loss(self):
        H = 5
        W = 5
        T = 10
        m = {}
        y = tf.placeholder('float', [None, None, H, W])
        y_ = tf.placeholder('float', [None, None, H, W])
        s = tf.placeholder('float', [None, None])
        s_ = tf.placeholder('float', [None, None])
        m['y'] = y
        m['y_'] = y_
        m['s'] = s
        m['s_'] = s_
        model_lib._add_ins_segm_loss(m, y, y_, s, s_, 1.0)

        # [1, 2, 3, 3]
        yv = np.array([
            [
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],

                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]]
            ]
        ])
        y_v = np.array([
            [
                [[1, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],

                [[0, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0]]
            ]
        ])
        with tf.Session() as sess:
            iou = sess.run(
                m['iou'], feed_dict={
                    m['y']: yv,
                    m['y_']: y_v
                }
            )
        print iou

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(
        RecurrentInstanceSegmentationTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
