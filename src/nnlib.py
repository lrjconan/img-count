import cslab_environ

from tensorflow.python import control_flow_ops
import tensorflow as tf

from utils import logger

log = logger.get()


def conv2d(x, w):
    """2-D convolution.

    Args:
        x: input tensor, [B, H, W, D]
        w: filter tensor, [F, F, In, Out]
    """
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, ratio):
    """N x N max pooling.

    Args:
        x: input tensor, [B, H, W, D]
        ratio: N by N pooling ratio
    """
    return tf.nn.max_pool(x, ksize=[1, ratio, ratio, 1],
                          strides=[1, ratio, ratio, 1], padding='SAME')


def weight_variable(shape, init=None, wd=None, name=None):
    """Initialize weights.

    Args:
        shape: shape of the weights, list of int
        wd: weight decay
    """
    if init is None:
        # init = tf.truncated_normal(shape, stddev=0.01)
        init = tf.truncated_normal_initializer(stddev=0.01)
    var = tf.Variable(init(shape), name=name)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def batch_norm(x, n_out, phase_train, scope='bn', affine=True):
    """
    Batch normalization on convolutional maps.
    Args:
        x: input tensor, [B, H, W, D]
        n_out: integer, depth of input maps
        phase_train: boolean tf.Variable, true indicates training phase
        scope: string, variable scope
        affine: whether to affine-transform outputs
    Return:
        normed: batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=affine)

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        batch_mean.set_shape([n_out])
        batch_var.set_shape([n_out])

        ema = tf.train.ExponentialMovingAverage(decay=0.999)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # mean, var = control_flow_ops.cond(phase_train,
        #                                   mean_var_with_update,
        #                                   lambda: (ema_mean, ema_var))
        mean, var = control_flow_ops.cond(phase_train,
                                          mean_var_with_update,
                                          lambda: (batch_mean, batch_var))

        # normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
        # beta, gamma, 1e-3, affine)
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def cnn(f, ch, pool, act, use_bn, phase_train=None, wd=None, scope='cnn'):
    """Add CNN. N = number of layers.

    Args:
        f: filter size, list of N int
        ch: number of channels, list of (N + 1) int
        pool: pooling ratio, list of N int
        act: activation function, list of N function
        use_bn: whether to use batch normalization, list of N bool
        phase_train: whether in training phase, tf bool variable
        wd: weight decay

    Returns:
        run_cnn: a function that runs the CNN
    """
    nlayers = len(f)
    w = [None] * nlayers
    b = [None] * nlayers
    log.info('CNN: {}'.format(scope))
    log.info('Channels: {}'.format(ch))
    log.info('Activation: {}'.format(act))
    # phase_train = tf.constant(True)

    with tf.variable_scope(scope):
        for ii in xrange(nlayers):
            w[ii] = weight_variable([f[ii], f[ii], ch[ii], ch[ii + 1]], wd=wd)
            b[ii] = weight_variable([ch[ii + 1]])
            log.info('Filter: {}'.format([f[ii], f[ii], ch[ii], ch[ii + 1]]))

    def run_cnn(x):
        """
        Run CNN on an input.
        Args:
            x: input image, [B, H, W, D]
        """
        h = [None] * nlayers
        for ii in xrange(nlayers):
            if ii == 0:
                prev_inp = x
            else:
                prev_inp = h[ii - 1]

            h[ii] = conv2d(prev_inp, w[ii]) + b[ii]

            if use_bn[ii]:
                h[ii] = batch_norm(h[ii], ch[ii + 1], phase_train)

            if act[ii] is not None:
                h[ii] = act[ii](h[ii])

            if pool[ii] > 1:
                h[ii] = max_pool(h[ii], pool[ii])

        return h

    return run_cnn


def dcnn(f, ch, pool, act, use_bn, skip_ch=None, phase_train=None, wd=None, scope='dcnn'):
    """Add DCNN. N = number of layers.

    Args:
        f: filter size, list of size N  int
        ch: number of channels, list of (N + 1) int
        pool: pooling ratio, list of N int
        act: activation function, list of N function
        use_bn: whether to use batch normalization, list of N bool
        skip_ch: skip connection, list of N int
        phase_train: whether in training phase, tf bool variable
        wd: weight decay

    Returns:
        run_dcnn: a function that runs the DCNN
    """
    nlayers = len(f)
    w = [None] * nlayers
    b = [None] * nlayers

    log.info('DCNN: {}'.format(scope))
    log.info('Channels: {}'.format(ch))
    log.info('Activation: {}'.format(act))
    log.info('Skip channels: {}'.format(skip_ch))
    # phase_train = tf.constant(True)

    in_ch = ch[0]

    with tf.variable_scope(scope):
        for ii in xrange(nlayers):
            out_ch = ch[ii + 1]

            if skip_ch is not None:
                if skip_ch[ii] is not None:
                    in_ch += skip_ch[ii]

            log.info('Filter: {}'.format([f[ii], f[ii], out_ch, in_ch]))
            w[ii] = weight_variable([f[ii], f[ii], out_ch, in_ch], wd=wd)
            b[ii] = weight_variable([out_ch])
            in_ch = out_ch

    def run_dcnn(x, skip=None):
        """Run DCNN on an input.

        Args:
            x: input image, [B, H, W, D]
            skip: skip connection activation map, list of 4-D tensor
        """
        h = [None] * nlayers
        out_shape = [None] * nlayers
        batch = tf.shape(x)[0: 1]
        inp_size = tf.shape(x)[1: 3]
        cum_pool = 1

        for ii in xrange(nlayers):
            cum_pool *= pool[ii]
            out_ch = ch[ii + 1]

            if ii == 0:
                prev_inp = x
            else:
                prev_inp = h[ii - 1]

            if skip is not None:
                if skip[ii] is not None:
                    if ii == 0:
                        prev_inp = tf.concat(3, [prev_inp, skip[ii]])
                    else:
                        prev_inp = tf.concat(3, [prev_inp, skip[ii]])

            out_shape[ii] = tf.concat(
                0, [batch, inp_size * cum_pool, tf.constant([out_ch])])

            h[ii] = tf.nn.conv2d_transpose(
                prev_inp, w[ii], out_shape[ii],
                strides=[1, pool[ii], pool[ii], 1]) + b[ii]

            if use_bn[ii]:
                h[ii] = batch_norm(h[ii], out_ch, phase_train)

            if act[ii] is not None:
                h[ii] = act[ii](h[ii])

        return h

    return run_dcnn


def dropout(x, keep_prob, phase_train):
    """Add dropout layer"""
    phase_train_f = tf.to_float(phase_train)
    keep_prob = (1.0 - phase_train_f) * 1.0 + phase_train_f * keep_prob
    return tf.nn.dropout(x, keep_prob)


def mlp(dims, act, add_bias=True, dropout_keep=None, phase_train=None, wd=None, scope='mlp'):
    """Add MLP. N = number of layers.

    Args:
        dims: layer-wise dimensions, list of N int
        act: activation function, list of N function
        dropout_keep: keep prob of dropout, list of N float
        phase_train: whether in training phase, tf bool variable
        wd: weight decay
    """
    nlayers = len(dims) - 1
    w = [None] * nlayers
    b = [None] * nlayers

    log.info('MLP: {}'.format(scope))
    log.info('Dimensions: {}'.format(dims))
    log.info('Activation: {}'.format(act))
    log.info('Dropout: {}'.format(dropout_keep))
    log.info('Add bias: {}'.format(add_bias))

    with tf.variable_scope(scope):
        for ii in xrange(nlayers):
            nin = dims[ii]
            nout = dims[ii + 1]
            w[ii] = weight_variable([nin, nout], wd=wd)
            if add_bias:
                b[ii] = weight_variable([nout])

    def run_mlp(x):
        h = [None] * nlayers
        for ii in xrange(nlayers):
            if ii == 0:
                prev_inp = x
            else:
                prev_inp = h[ii - 1]

            if dropout_keep is not None:
                if dropout_keep[ii] is not None:
                    prev_inp = dropout(prev_inp, dropout_keep[ii], phase_train)

            h[ii] = tf.matmul(prev_inp, w[ii])

            if add_bias:
                h[ii] += b[ii]

            if act[ii]:
                h[ii] = act[ii](h[ii])

        return h

    return run_mlp


def conv_lstm(inp_depth, hid_depth, filter_size, wd=None, scope='conv_lstm'):
    """Adds a Conv-LSTM component.

    Args:
        inp_depth: Input image depth
        filter_size: Conv gate filter size
        hid_depth: Hidden state depth
        wd: Weight decay
        name: Prefix
    """
    log.info('ConvLSTM: {}'.format(scope))
    log.info('Input depth: {}'.format(inp_depth))
    log.info('Hidden depth: {}'.format(hid_depth))

    with tf.variable_scope(scope):
        # Input gate
        w_xi = weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                               wd=wd, name='w_xi')
        w_hi = weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                               wd=wd, name='w_hi')
        b_i = weight_variable([hid_depth], name='b_i')

        # Forget gate
        w_xf = weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                               wd=wd, name='w_xf')
        w_hf = weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                               wd=wd, name='w_hf')
        b_f = weight_variable([hid_depth], name='b_f')

        # Input activation
        w_xu = weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                               wd=wd, name='w_xu')
        w_hu = weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                               wd=wd, name='w_hu')
        b_u = weight_variable([hid_depth], name='b_u')

        # Output gate
        w_xo = weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                               wd=wd, name='w_xo')
        w_ho = weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                               wd=wd, name='w_ho')
        b_o = weight_variable([hid_depth], name='b_o')

    def unroll(inp, state):
        c = tf.slice(state, [0, 0, 0, 0], [-1, -1, -1, hid_depth])
        h = tf.slice(state, [0, 0, 0, hid_depth], [-1, -1, -1, hid_depth])
        g_i = tf.sigmoid(conv2d(inp, w_xi) + conv2d(h, w_hi) + b_i)
        g_f = tf.sigmoid(conv2d(inp, w_xf) + conv2d(h, w_hf) + b_f)
        g_o = tf.sigmoid(conv2d(inp, w_xo) + conv2d(h, w_ho) + b_o)
        u = tf.tanh(conv2d(inp, w_xu) + conv2d(h, w_hu) + b_u)
        c = g_f * c + g_i * u
        h = g_o * tf.tanh(c)
        state = tf.concat(3, [c, f])

        return state

    return unroll


def lstm(inp_dim, hid_dim, wd=None, scope='lstm'):
    """Adds an LSTM component.

    Args:
        inp_dim: Input data dim
        hid_dim: Hidden state dim
        wd: Weight decay
        scope: Prefix
    """
    log.info('LSTM: {}'.format(scope))
    log.info('Input dim: {}'.format(inp_dim))
    log.info('Hidden dim: {}'.format(hid_dim))

    with tf.variable_scope(scope):
        # Input gate
        w_xi = weight_variable([inp_dim, hid_dim], wd=wd, name='w_xi')
        w_hi = weight_variable([hid_dim, hid_dim], wd=wd, name='w_hi')
        b_i = weight_variable([hid_dim], init=tf.constant_initializer(0.0),
                              name='b_i')

        # Forget gate
        w_xf = weight_variable([inp_dim, hid_dim], wd=wd, name='w_xf')
        w_hf = weight_variable([hid_dim, hid_dim], wd=wd, name='w_hf')
        b_f = weight_variable([hid_dim], init=tf.constant_initializer(1.0),
                              name='b_f')

        # Input activation
        w_xu = weight_variable([inp_dim, hid_dim], wd=wd, name='w_xu')
        w_hu = weight_variable([hid_dim, hid_dim], wd=wd, name='w_hu')
        b_u = weight_variable([hid_dim], init=tf.constant_initializer(0.0),
                              name='b_u')

        # Output gate
        w_xo = weight_variable([inp_dim, hid_dim], wd=wd, name='w_xo')
        w_ho = weight_variable([hid_dim, hid_dim], wd=wd, name='w_ho')
        b_o = weight_variable([hid_dim], init=tf.constant_initializer(0.0),
                              name='b_o')

    def unroll(inp, state):
        c = tf.slice(state, [0, 0], [-1, hid_dim])
        h = tf.slice(state, [0, hid_dim], [-1, hid_dim])
        g_i = tf.sigmoid(tf.matmul(inp, w_xi) + tf.matmul(h, w_hi) + b_i)
        g_f = tf.sigmoid(tf.matmul(inp, w_xf) + tf.matmul(h, w_hf) + b_f)
        g_o = tf.sigmoid(tf.matmul(inp, w_xo) + tf.matmul(h, w_ho) + b_o)
        u = tf.tanh(tf.matmul(inp, w_xu) + tf.matmul(h, w_hu) + b_u)
        c = g_f * c + g_i * u
        h = g_o * tf.tanh(c)
        state = tf.concat(1, [c, h])

        return state, g_i, g_f, g_o

    return unroll


def gru(inp_dim, hid_dim, wd=None, scope='gru'):
    """Adds a GRU component.

    Args:
        inp_dim: Input data dim
        hid_dim: Hidden state dim
        wd: Weight decay
        scope: Prefix
    """
    log.info('GRU: {}'.format(scope))
    log.info('Input dim: {}'.format(inp_dim))
    log.info('Hidden dim: {}'.format(hid_dim))

    with tf.variable_scope(scope):
        w_xi = weight_variable([inp_dim, hid_dim], wd=wd, name='w_xi')
        w_hi = weight_variable([hid_dim, hid_dim], wd=wd, name='w_hi')
        b_i = weight_variable([hid_dim], name='b_i')

        w_xu = weight_variable([inp_dim, hid_dim], wd=wd, name='w_xu')
        w_hu = weight_variable([hid_dim, hid_dim], wd=wd, name='w_hu')
        b_u = weight_variable([hid_dim], name='b_u')

        w_xr = weight_variable([inp_dim, hid_dim], wd=wd, name='w_xr')
        w_hr = weight_variable([hid_dim, hid_dim], wd=wd, name='w_hr')
        b_r = weight_variable([hid_dim], name='b_r')

    def unroll(inp, state):
        g_i = tf.sigmoid(tf.matmul(inp, w_xi) + tf.matmul(state, w_hi) + b_i)
        g_r = tf.sigmoid(tf.matmul(inp, w_xr) + tf.matmul(state, w_hr) + b_r)
        u = tf.tanh(tf.matmul(inp, w_xu) + g_r * tf.matmul(state, w_hu) + b_u)
        state = state * (1 - g_i) + u * g_i

        return state

    return unroll
