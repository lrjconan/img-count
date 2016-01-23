import numpy as np
from utils import logger

log = logger.get()


def augment(R):
    """Finds an augmenting path using BFS.

    Args:
        R: numpy.ndarray, [n, n], residual flow, will be updated by the
        function.
    Returns:
        dF: numpy.ndarray, [n, n], delta flow (upper triangular).
        success: bool, whether found an augmenting path.
    """
    # Number of vertices.
    n = R.shape[0]
    # Vertex index of s.
    s = 0
    # Vertex index of t.
    t = n - 1

    # BFS queue.
    q = [s]
    # Visited vertices.
    mark = np.zeros([n], dtype='bool')
    # Parent list.
    p = np.zeros([n], dtype='int32') - 1

    found = False

    while len(q) > 0:
        v = q[0]
        del q[0]
        mark[v] = True

        if v == t:
            found = True
            break

        for u in xrange(n):
            if not mark[u] and R[v, u] > 0:
                q.append(u)
                p[u] = v

    dF = np.zeros([n, n])
    if found:
        # Bottleneck.
        b = R.max()
        v = t
        while p[v] != -1:
            b = min(b, R[p[v], v])
            v = p[v]

        # log.info('parents: {}'.format(p))
        v = t
        while p[v] != -1:
            if p[v] < v:
                dF[p[v], v] = b
            else:
                dF[v, p[v]] = -b
            R[p[v], v] -= b
            R[v, p[v]] += b
            v = p[v]

    return dF, found


def residual(G, F):
    """Gets residual flow.

    Args:
        G: numpy.ndarray, [n, n], edge weight matrix (upper triangular).
        F: numpy.ndarray, [n, n], current flow matrix (upper triangular).
    Returns:
        R: numpy.ndarray, [n, n], residual flow.
    """
    # Residual flow in regular direction.
    R = G - F
    # Flow in the reverse direction.
    R += G.T - R.T

    return R


def max_flow(G):
    """Calculates the max flow for any graph.

    Args:
        G: numpy.ndarray, [n, n], edge weight matrix (upper triangular).
    Returns:
        F: numpy.ndarray, [n, n], flow matrix (upper triangular).
    """

    if np.tril(G).sum() > 0:
        log.error('G must be upper triangular')
        raise Exception('G must be upper triangular')

    n = G.shape[0]
    F = np.zeros([n, n], dtype=G.dtype)
    R = residual(G, F)
    ii = 0
    while True:
        dF, success = augment(R)
        F += dF
        if not success:
            break
        # log.info('i: {}, residual'.format(ii))
        # print R
        ii += 1

    return F


def max_bp_match(G):
    """Calculates the max matching for bipartite graphs.

    Args:
        G: numpy.ndarray, [n_X, n_Y], edge weight matrix.
    Returns:
        M: numpy.ndarray, [n_X, n_Y], matched edge 1, unmatched 0.
    """
    # Form a source-sink flow graph.

    # Number of vertices in X
    n_X = G.shape[0]
    # Number of vertices in Y
    n_Y = G.shape[0]
    # Total number of vertices, including source (s) and sink (t).
    n = n_X + n_Y + 2

    # Flow di-graph.
    F = np.zeros([n, n], dtype=G.dtype)

    # Vertex index of s.
    s = 0
    # Vertex index of t.
    t = n_X + n_Y + 1

    # Copy the flow capacity (0 or 1 only).
    x_start = 1
    y_start = n_X + 1
    F[x_start: y_start, y_start: t] = G.astype('int32')

    # Source capacity.
    F[s, x_start: y_start] = 1
    # Sink capacity.
    F[y_start: t, t] = 1
    # log.info('reformed graph')
    # print F

    Fmax = max_flow(F)
    # print Fmax

    return Fmax[x_start: y_start, y_start: t]


def is_bp_match_saturate(M):
    """Checks if a bi-partite matching saturates smaller side, ."""
    max_axis = np.argmax(np.array(M.shape))
    return (M.sum(axis=max_axis) == 1).all()


def get_bp_neighbours(v, G):
    """Gets neighbours of a vertex in a bi-partite graph."""
    return set(G[v, :].nonzero()[0])


def get_set_bp_neighbours(S, G):
    """Gets neighbours of a set of vertices in a bi-partite graph."""
    N = set()
    for v in S:
        N = N.union(get_bp_neighbours(v, G))

    return N


def hungarian(W):
    """Calculates the min weighted bi-partite vertex cover or max weighted
    bi-partite matching.

    Args:
        W: numpy.ndarray, edge weight matrix.
    Returns:
        c_0: numpy.ndarray, [n_X], vertex cover on X.
        c_1: numpy.ndarray, [n_Y], vertex cover on Y.
        M: numpy.ndarray, [n_X, n_Y], max matching, 1 matched, 0 unmatched.
    """

    n_X = W.shape[0]
    n_Y = W.shape[1]

    # Generate initial cover.
    w_max = W.max()
    c_0 = W.max(axis=1)
    c_1 = np.zeros([n_Y], dtype=W.dtype)
    log.info('c_0: {}'.format(c_0))
    log.info('c_1: {}'.format(c_1))
    S = set()
    T = set()
    next_match = True
    i = 0

    while True:
        log.info('---------------------------------------------')
        log.info('iteration {}'.format(i))
        # Update equality graph.
        E = ((c_0.reshape([-1, 1]) + c_1.reshape([1, -1]) - W)
             == 0).astype('uint8')
        log.info('equality graph')
        print E

        if next_match:
            log.info('look for maximum matching in equality graph')
            M = max_bp_match(E)

            log.info('new matching')
            print M
            if is_bp_match_saturate(M):
                log.info('found solution, exit')
                log.info('---------------------------------------------')
                return c_0, c_1, M

            for u in xrange(n_X):
                if M[u, :].sum() == 0:
                    S.clear()
                    S.add(u)
                    T.clear()

        N_S = get_set_bp_neighbours(S, E)
        log.info('S: {}'.format(S))
        log.info('T: {}'.format(T))
        log.info('N_S: {}'.format(N_S))
        if N_S == T:
            log.info('N_S == T')
            log.info('update cover')
            # Update cover.
            a = w_max
            for x in S:
                for y in xrange(n_Y):
                    if y not in T:
                        a = min(a, c_0[x] + c_1[y] - W[x, y])
            log.info('a: {}'.format(a))
            # a = (c_0.reshape([-1, 1]) + c_1.reshape([1, -1]) - W).min()
            for x in S:
                c_0[x] -= a
            for y in T:
                c_1[y] += a
            log.info('c_0: {}'.format(c_0))
            log.info('c_1: {}'.format(c_1))
        else:
            log.info('N_S != T')
            for y in xrange(n_Y):
                if y not in T:
                    log.info('pick y not in T: {}'.format(y))
                    break
            if M[:, y].sum() == 0:
                log.info('y unmatched, look for matching')
                next_match = True
            else:
                log.info('y matched, increase S and T')
                next_match = False
                z = M[:, y].nonzero()[0][0]
                S.add(z)
                T.add(y)
                log.info('S: {}'.format(S))
                log.info('T: {}'.format(T))

        log.info('end of iteration')
        log.info('---------------------------------------------')
        i += 1


if __name__ == '__main__':
    log.info('---------------------------------------------')
    log.info('max flow 1')
    # Simple example.
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
    log.info('original graph')
    print G
    F = max_flow(G)
    log.info('f_max')
    print F
    # assert((F == G).all(), 'check fail')

    log.info('---------------------------------------------')
    log.info('max flow 2')
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
    log.info('original graph ')
    print G
    F = max_flow(G)
    log.info('f_max')
    print F
    Ft = np.array([[0, 3, 2, 0, 0, 0],
                   [0, 0, 0, 3, 0, 0],
                   [0, 0, 0, 0, 2, 0],
                   [0, 0, 0, 0, 1, 2],
                   [0, 0, 0, 0, 0, 3],
                   [0, 0, 0, 0, 0, 0]])
    # assert((F == Ft).all(), 'check fail')

    log.info('---------------------------------------------')
    log.info('max bi-partite matching')
    G = np.array([[0, 1, 1],
                  [0, 0, 1],
                  [0, 0, 0]])
    log.info('original graph')
    print G
    M = max_bp_match(G)
    log.info('m_max')
    print M
    Mt = np.array([[0, 1, 0],
                   [0, 0, 1],
                   [0, 0, 0]])
    # assert((M == Mt).all(), 'check fail')
    log.info('saturated: {}'.format(is_bp_match_saturate(M)))

    log.info('---------------------------------------------')
    log.info('check saturation')
    M = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [0, 0, 0],
                  [1, 0, 0]])
    log.info('matching')
    print M
    log.info('saturated: {}'.format(is_bp_match_saturate(M)))

    log.info('---------------------------------------------')
    log.info('min weighted bi-partite vertex cover')
    W = np.array([[3, 2, 2],
                  [1, 2, 0],
                  [2, 2, 1]])
    log.info('graph')
    print W
    c_0, c_1, M = hungarian(W)
    log.info('c_0: {}'.format(c_0))
    log.info('c_1: {}'.format(c_1))
    log.info('max matching')
    print M
    c_0_t = np.array([2, 1, 1])
    c_1_t = np.array([1, 1, 0])
    M_t = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
    log.info('---------------------------------------------')
