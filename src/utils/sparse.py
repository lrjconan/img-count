import scipy.sparse


def make_sparse_list(mat_list, batch_size=1000):
    pass


def make_sparse_batch(mat, batch_size=10000):
    N = mat.shape[0]
    num_bat = int(np.ceil(N / float(batch_size)))
    feat_sparse = []
    for i in xrange(num_bat):
        start = bat * i
        end = min(bat * (i + 1), N)
        feat_copy = feat[start: end]
        feat_sparse_tmp = scipy.sparse.csr_matrix(feat_copy)
        feat_sparse.append(feat_sparse_tmp)
    return scipy.sparse.vstack(feat_sparse, format='csr')
