import hungarian
import numpy as np
import unittest


class HungarianTests(unittest.TestCase):

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
        F = hungarian.max_flow(G)
        self.assertTrue((F == G).all())

        pass

    def test_max_flow_2(self):
        # Another example.
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
        F = hungarian.max_flow(G)
        Ft = np.array([[0, 3, 2, 0, 0, 0],
                       [0, 0, 0, 3, 0, 0],
                       [0, 0, 0, 0, 2, 0],
                       [0, 0, 0, 0, 1, 2],
                       [0, 0, 0, 0, 0, 3],
                       [0, 0, 0, 0, 0, 0]])
        self.assertTrue((F == Ft).all())

        pass

    def test_max_bp_match_1(self):
        G = np.array([[0, 1, 1],
                      [0, 0, 1],
                      [0, 0, 0]])
        M = hungarian.max_bp_match(G)
        Mt = np.array([[0, 1, 0],
                       [0, 0, 1],
                       [0, 0, 0]])
        self.assertTrue((M == Mt).all())

        pass

    def test_is_bp_match_saturate_1(self):
        M = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
        self.assertFalse(hungarian.is_bp_match_saturate(M))

        pass

    def test_is_bp_match_saturate_2(self):
        M = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0],
                      [1, 0, 0]])
        self.assertTrue(hungarian.is_bp_match_saturate(M))

        pass

    def test_min_weighted_bp_cover_1(self):
        W = np.array([[3, 2, 2],
                      [1, 2, 0],
                      [2, 2, 1]])
        c_0, c_1, M = hungarian.min_weighted_bp_cover(W)
        c_0_t = np.array([2, 1, 1])
        c_1_t = np.array([1, 1, 0])
        M_t = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
        self.assertTrue((c_0 == c_0_t).all())
        self.assertTrue((c_1 == c_1_t).all())
        self.assertTrue((M == M_t).all())

        pass

    def test_min_weighted_bp_cover_2(self):
        W = np.array([[5, 0, 0],
                      [0, 5, 0],
                      [4, 0, 3],
                      [0, 4, 0]])
        c_0, c_1, M = hungarian.min_weighted_bp_cover(W)
        c_0_t = np.array([4, 4, 3, 3])
        c_1_t = np.array([1, 1, 0])
        M_t = np.array([[1, 0, 0],
                        [0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0]])
        self.assertTrue((c_0 == c_0_t).all())
        self.assertTrue((c_1 == c_1_t).all())
        self.assertTrue((M == M_t).all())

        pass


    def test_min_weighted_bp_cover_2(self):
        W = np.array([[5, 0, 4, 0],
                      [0, 4, 6, 8],
                      [4, 0, 5, 7]])
        c_0, c_1, M = hungarian.min_weighted_bp_cover(W)
        c_0_t = np.array([5, 6, 5])
        c_1_t = np.array([0, 0, 0, 2])
        M_t = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        self.assertTrue((c_0 == c_0_t).all())
        self.assertTrue((c_1 == c_1_t).all())
        self.assertTrue((M == M_t).all())

        pass


    def test_min_weighted_bp_cover_3(self):
        W = np.array([[-5, -3, -4, -4],
                      [-2, -4, -6, -8],
                      [-4, -5, -5, -7]])
        c_0, c_1, M = hungarian.min_weighted_bp_cover(W)
        c_0_t = np.array([-3, -3, -5])
        c_1_t = np.array([1, 0, 0, 0])
        M_t = np.array([[0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 1, 0]])
        self.assertTrue((c_0 == c_0_t).all())
        self.assertTrue((c_1 == c_1_t).all())
        self.assertTrue((M == M_t).all())

        pass

    def test_min_weighted_bp_cover_4(self):
        W = np.array([[5, 0, 2],
                      [3, 1, -2],
                      [0, 5, 0]])
        c_0, c_1, M = hungarian.min_weighted_bp_cover(W)
        c_0_t = np.array([2, 0, 4])
        c_1_t = np.array([3, 1, 0])
        M_t = np.array([[0, 0, 1],
                        [1, 0, 0],
                        [0, 1, 0]])
        self.assertTrue((c_0 == c_0_t).all())
        self.assertTrue((c_1 == c_1_t).all())
        self.assertTrue((M == M_t).all())

        pass

if __name__ == '__main__':
    # unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(HungarianTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
