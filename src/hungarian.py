import numpy as np
from utils import logger

log = logger.get()


def augment(G, F, R):
    """Finds an augmenting path using BFS.

    Args:
        G: numpy.ndarray, [n, n], singly directed graph.
        F: numpy.ndarray, [n, n], total flow.
        R: numpy.ndarray, [n, n], residual flow, will be updated by the
        function.
    Returns:
        dF: numpy.ndarray, [n, n], change in flow.
        found: bool, whether an augmenting path is found.
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

    if found:
        # Bottleneck.
        b = R.max()
        v = t
        while p[v] != -1:
            b = min(b, R[p[v], v])
            v = p[v]

        log.info('parents: {}'.format(p), verbose=3)
        v = t
        while p[v] != -1:
            if G[p[v], v] > 0:
                F[p[v], v] += b
            else:
                F[v, p[v]] -= b
            R[p[v], v] -= b
            R[v, p[v]] += b
            v = p[v]

    return found


def max_flow(G):
    """Calculates the max flow for any graph.

    Args:
        G: numpy.ndarray, [n, n], edge weight matrix (upper triangular).
    Returns:
        F: numpy.ndarray, [n, n], flow matrix (upper triangular).
    """

    with log.verbose_level(3):
        if np.tril(G).sum() > 0:
            log.error('G must be upper triangular')
            raise Exception('G must be upper triangular')

        n = G.shape[0]
        R = np.copy(G)
        F = np.zeros([n, n], dtype=G.dtype)
        ii = 0
        while augment(G, F, R):
            log.info('i: {}, residual: \n{}'.format(ii, R))
            ii += 1

    return F


def max_bp_match(G):
    """Calculates the max matching for bi-partite graphs.

    Args:
        G: numpy.ndarray, [n_X, n_Y], edge weight matrix.
    Returns:
        M: numpy.ndarray, [n_X, n_Y], binary matrix, matched 1, unmatched 0.
    """
    with log.verbose_level(3):
        # Form a source-sink flow graph.
        # Number of vertices in X
        n_X = G.shape[0]
        # Number of vertices in Y
        n_Y = G.shape[1]
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
        log.info('reformed graph: \n{}'.format(F))

        Fmax = max_flow(F)
        log.info('max flow: \n{}'.format(Fmax))

    return Fmax[x_start: y_start, y_start: t]


def is_bp_match_saturate(M):
    """Checks if a bi-partite matching saturates the smaller side.

    Args:
        M: numpy.ndarray, [n_X, n_Y], binary matrix, matched 1, unmatched 0.
    Returns:
        saturate: bool, whether the matching saturates the smaller side.
    """
    max_axis = np.argmax(np.array(M.shape))

    return (M.sum(axis=max_axis) == 1).all()


def get_bp_neighbours(v, G):
    """Gets neighbours of a vertex in a bi-partite graph.

    Args:
        v: int, vertex to search from.
        G: numpy.ndarray, [n_X, n_Y], nonzero terms represent edges.
    Returns:
        N_v: set, a set of neighbours (int) of v.
    """
    return set(G[v, :].nonzero()[0])


def get_set_bp_neighbours(S, G):
    """Gets neighbours of a set of vertices in a bi-partite graph.

    Args:
        S: set, a set of vertices (int).
        G: numpy.ndarray, [n_X, n_Y], nonzero terms represent edges.
    Returns:
        N_S: set, a set of neighbours (int) of any vertex in S.
    """
    N = set()
    for v in S:
        N = N.union(get_bp_neighbours(v, G))

    return N


def min_weighted_bp_cover(W):
    """Calculates the min weighted bi-partite vertex cover or max weighted
    bi-partite matching.

    Args:
        W: numpy.ndarray, edge weight matrix.
    Returns:
        c_0: numpy.ndarray, [n_X], vertex cover on X.
        c_1: numpy.ndarray, [n_Y], vertex cover on Y.
        M: numpy.ndarray, [n_X, n_Y], max matching, 1 matched, 0 unmatched.
    """

    with log.verbose_level(2):
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
            log.info('equality graph: \n{}'.format(E))

            # Look for maximum matching in equality graph
            if next_match:
                log.info('look for maximum matching in equality graph')
                M = max_bp_match(E)

                log.info('new matching: \n{}'.format(M))
                if is_bp_match_saturate(M):
                    log.info('found solution, exit')
                    log.info(
                        '---------------------------------------------')
                    return c_0, c_1, M

                for u in xrange(n_X):
                    if M[u, :].sum() == 0:
                        S.add(u)

            N_S = get_set_bp_neighbours(S, E)
            log.info('S: {}'.format(S))
            log.info('T: {}'.format(T))
            log.info('N_S: {}'.format(N_S))
            if N_S == T:
                # Update vertex cover.
                log.info('N_S == T')
                log.info('update cover')
                a = np.abs(c_0).max() + np.abs(c_1).max() + np.abs(W).max()
                for x in S:
                    for y in xrange(n_Y):
                        if y not in T:
                            a = min(a, c_0[x] + c_1[y] - W[x, y])
                log.info('a: {}'.format(a))
                for x in S:
                    c_0[x] -= a
                for y in T:
                    c_1[y] += a
                log.info('c_0: {}'.format(c_0))
                log.info('c_1: {}'.format(c_1))
            else:
                # Update S and T.
                log.info('N_S != T')
                while len(N_S) > len(T):
                    for y in N_S:
                        if y not in T:
                            log.info('pick y in N_S not in T: {}'.format(y))
                            break
                    if M[:, y].sum() == 0:
                        log.info('y unmatched, look for matching')
                        next_match = True
                        break
                    else:
                        log.info('y matched, increase S and T')
                        next_match = False
                        z = M[:, y].nonzero()[0][0]
                        S.add(z)
                        N_S = N_S.union(get_bp_neighbours(z, E))
                        T.add(y)
                        log.info('S: {}'.format(S))
                        log.info('N_S: {}'.format(N_S))
                        log.info('T: {}'.format(T))

            log.info('end of iteration')
            log.info('---------------------------------------------')
            i += 1

    pass
