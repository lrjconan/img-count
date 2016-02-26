import cslab_environ

from tensorflow.python import control_flow_ops
import tensorflow as tf

from utils import logger

log = logger.get()


def cnn(model, x, f, ch, pool, act, use_bn, phase_train=None, wd=None):
    """Add CNN, with standard Conv-Relu-MaxPool layers.

    Args:
        model:
        x: input image
        f: filter size, list of size N (N = number of layers)
        ch: number of channels, list of size N + 1
        pool: pooling ratio, list of size N
        wd: weight decay
    """
    nlayers = len(f)
    w = [None] * nlayers
    b = [None] * nlayers
    h = [None] * nlayers
    log.info('CNN')
    log.info('Channels: {}'.format(ch))

    for ii in xrange(nlayers):
        w[ii] = weight_variable([f[ii], f[ii], ch[ii], ch[ii + 1]], wd=wd)
        b[ii] = weight_variable([ch[ii + 1]], wd=wd)
        log.info('Filter: {}'.format([f[ii], f[ii], ch[ii], ch[ii + 1]]))

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


def dcnn(model, x, f, ch, pool, act, use_bn, inp_h, inp_w, skip=None, skip_ch=None, phase_train=None, wd=None):
    """Add D-CNN, with standard DeConv-Relu layers.

    Args:
        model:
        x: input image
        f: filter size, list of size N (N = number of layers)
        ch: number of channels, list of size N + 1
        pool: pooling ratio, list of size N
        skip: skip connection
        wd: weight decay
    """
    nlayers = len(f)
    w = [None] * nlayers
    b = [None] * nlayers
    h = [None] * nlayers
    out_shape = [None] * nlayers

    batch = tf.shape(x)[0: 1]
    inp_size = tf.shape(x)[1: 3]
    cum_pool = 1
    log.info('DCNN')
    log.info('Channels: {}'.format(ch))
    log.info('Skip channels: {}'.format(skip_ch))

    in_ch = ch[0]

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
                in_ch += skip_ch[ii]

        out_shape[ii] = tf.concat(
            0, [batch, inp_size * cum_pool, tf.constant([out_ch])])
        log.info('Filter: {}'.format([f[ii], f[ii], out_ch, in_ch]))
        w[ii] = weight_variable([f[ii], f[ii], out_ch, in_ch], wd=wd)
        b[ii] = weight_variable([out_ch], wd=wd)

        h[ii] = tf.nn.conv2d_transpose(
            prev_inp, w[ii], out_shape[ii],
            strides=[1, pool[ii], pool[ii], 1]) + b[ii]

        inp_h *= pool[ii]
        inp_w *= pool[ii]

        h[ii].set_shape([None, inp_h, inp_w, out_ch])

        if use_bn[ii]:
            h[ii] = batch_norm(h[ii], out_ch, phase_train)

        if act[ii] is not None:
            h[ii] = act[ii](h[ii])

        in_ch = out_ch

        model['dcnn_h_{}'.format(ii)] = h[ii]

    return h


def dropout(x, keep_prob, phase_train):
    phase_train_f = tf.to_float(phase_train)
    keep_prob = (1.0 - phase_train_f) * 1.0 + phase_train_f * keep_prob
    return tf.nn.dropout(x, keep_prob)


def mlp(model, x, dims, act, dropout_keep=None, phase_train=None, wd=None):
    nlayers = len(dims) - 1
    w = [None] * nlayers
    b = [None] * nlayers
    h = [None] * nlayers

    log.info('MLP')
    log.info('Dimensions: {}'.format(dims))
    log.info('Dropout: {}'.format(dropout_keep))

    for ii in xrange(nlayers):
        nin = dims[ii]
        nout = dims[ii + 1]
        w[ii] = weight_variable([nin, nout], wd=wd)
        b[ii] = weight_variable([nout], wd=wd)

        if ii == 0:
            prev_inp = x
        else:
            prev_inp = h[ii - 1]

        if dropout_keep is not None:
            if dropout_keep[ii] is not None:
                prev_inp = dropout(prev_inp, dropout_keep[ii], phase_train)

        h[ii] = tf.matmul(prev_inp, w[ii]) + b[ii]

        if act[ii]:
            h[ii] = act[ii](h[ii])

    return h


def batch_norm(x, n_out, phase_train, scope='bn', affine=True):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Variable, true indicates training phase
        scope:       string, variable scope
        affine:      whether to affine-transform outputs
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=affine)

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = control_flow_ops.cond(phase_train,
                                          mean_var_with_update,
                                          lambda: (ema_mean, ema_var))

        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
                                                            beta, gamma, 1e-3, affine)
    return normed


def conv2d(x, w):
    """2-D convolution."""
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, ratio):
    """2 x 2 max pooling."""
    return tf.nn.max_pool(x, ksize=[1, ratio, ratio, 1],
                          strides=[1, ratio, ratio, 1], padding='SAME')


def weight_variable(shape, wd=None, name=None):
    """Initialize weights."""
    initial = tf.truncated_normal(shape, stddev=0.01)
    var = tf.Variable(initial, name=name)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def conv_lstm(model, timespan, inp_height, inp_width, inp_depth, filter_size, hid_depth, c_init, h_init, wd=None, name=''):
    """Adds a Conv-LSTM component.

    Args:
        model: Model dictionary
        timespan: Maximum length of the LSTM
        inp_height: Input image height
        inp_width: Input image width
        inp_depth: Input image depth
        filter_size: Conv gate filter size
        hid_depth: Hidden state depth
        c_init: Cell state initialization
        h_init: Hidden state initialization
        wd: Weight decay
        name: Prefix
    """
    g_i = [None] * timespan
    g_f = [None] * timespan
    g_o = [None] * timespan
    u = [None] * timespan
    c = [None] * (timespan + 1)
    h = [None] * (timespan + 1)
    c[-1] = c_init
    h[-1] = h_init

    # Input gate
    w_xi = weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                           wd=wd, name='w_xi_{}'.format(name))
    w_hi = weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                           wd=wd, name='w_hi_{}'.format(name))
    b_i = weight_variable([hid_depth],
                          wd=wd, name='b_i_{}'.format(name))

    # Forget gate
    w_xf = weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                           wd=wd, name='w_xf_{}'.format(name))
    w_hf = weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                           wd=wd, name='w_hf_{}'.format(name))
    b_f = weight_variable([hid_depth],
                          wd=wd, name='b_f_{}'.format(name))

    # Input activation
    w_xu = weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                           wd=wd, name='w_xu_{}'.format(name))
    w_hu = weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                           wd=wd, name='w_hu_{}'.format(name))
    b_u = weight_variable([hid_depth],
                          wd=wd, name='b_u_{}'.format(name))

    # Output gate
    w_xo = weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                           wd=wd, name='w_xo_{}'.format(name))
    w_ho = weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                           wd=wd, name='w_ho_{}'.format(name))
    b_o = weight_variable([hid_depth], name='b_o_{}'.format(name))

    def unroll(inp, time):
        t = time
        g_i[t] = tf.sigmoid(conv2d(inp, w_xi) + conv2d(h[t - 1], w_hi) + b_i)
        g_f[t] = tf.sigmoid(conv2d(inp, w_xf) + conv2d(h[t - 1], w_hf) + b_f)
        g_o[t] = tf.sigmoid(conv2d(inp, w_xo) + conv2d(h[t - 1], w_ho) + b_o)
        u[t] = tf.tanh(conv2d(inp, w_xu) + conv2d(h[t - 1], w_hu) + b_u)
        c[t] = g_f[t] * c[t - 1] + g_i[t] * u[t]
        h[t] = g_o[t] * tf.tanh(c[t])

        pass

    model['g_i_{}'.format(name)] = g_i
    model['g_f_{}'.format(name)] = g_f
    model['g_o_{}'.format(name)] = g_o
    model['u_{}'.format(name)] = u
    model['c_{}'.format(name)] = c
    model['h_{}'.format(name)] = h

    return unroll


def lstm(model, timespan, inp_dim, hid_dim, c_init, h_init, wd=None, name=None):
    """Adds an LSTM component.

    Args:
        model: Model dictionary
        timespan: Maximum length of the LSTM
        inp_height: Input image height
        inp_width: Input image width
        inp_depth: Input image depth
        filter_size: Conv gate filter size
        hid_depth: Hidden state depth
        c_init: Cell state initialization
        h_init: Hidden state initialization
        wd: Weight decay
        name: Prefix
    """
    g_i = [None] * timespan
    g_f = [None] * timespan
    g_o = [None] * timespan
    u = [None] * timespan
    c = [None] * (timespan + 1)
    h = [None] * (timespan + 1)
    c[-1] = c_init
    h[-1] = h_init

    # Input gate
    w_xi = weight_variable([inp_dim, hid_dim], wd=wd,
                           name='w_xi_{}'.format(name))
    w_hi = weight_variable([hid_dim, hid_dim], wd=wd,
                           name='w_hi_{}'.format(name))
    b_i = weight_variable([hid_dim], wd=wd, name='b_i_{}'.format(name))

    # Forget gate
    w_xf = weight_variable([inp_dim, hid_dim], wd=wd,
                           name='w_xf_{}'.format(name))
    w_hf = weight_variable([hid_dim, hid_dim], wd=wd,
                           name='w_hf_{}'.format(name))
    b_f = weight_variable([hid_dim], wd=wd, name='b_f_{}'.format(name))

    # Input activation
    w_xu = weight_variable([inp_dim, hid_dim], wd=wd,
                           name='w_xu_{}'.format(name))
    w_hu = weight_variable([hid_dim, hid_dim], wd=wd,
                           name='w_hu_{}'.format(name))
    b_u = weight_variable([hid_dim], wd=wd, name='b_u_{}'.format(name))

    # Output gate
    w_xo = weight_variable([inp_dim, hid_dim], wd=wd,
                           name='w_xo_{}'.format(name))
    w_ho = weight_variable([hid_dim, hid_dim], wd=wd,
                           name='w_ho_{}'.format(name))
    b_o = weight_variable([hid_dim], name='b_o_{}'.format(name))

    def unroll(inp, time):
        t = time
        g_i[t] = tf.sigmoid(tf.matmul(inp, w_xi) +
                            tf.matmul(h[t - 1], w_hi) + b_i)
        g_f[t] = tf.sigmoid(tf.matmul(inp, w_xf) +
                            tf.matmul(h[t - 1], w_hf) + b_f)
        g_o[t] = tf.sigmoid(tf.matmul(inp, w_xo) +
                            tf.matmul(h[t - 1], w_ho) + b_o)
        u[t] = tf.tanh(tf.matmul(inp, w_xu) + tf.matmul(h[t - 1], w_hu) + b_u)
        c[t] = g_f[t] * c[t - 1] + g_i[t] * u[t]
        h[t] = g_o[t] * tf.tanh(c[t])

        pass

    model['g_i_{}'.format(name)] = g_i
    model['g_f_{}'.format(name)] = g_f
    model['g_o_{}'.format(name)] = g_o
    model['u_{}'.format(name)] = u
    model['c_{}'.format(name)] = c
    model['h_{}'.format(name)] = h

    return unroll


def gru(model, timespan, inp_dim, hid_dim, h_init, wd=None, name=''):
    h = [None] * (timespan + 1)
    g_i = [None] * timespan
    g_r = [None] * timespan
    h[-1] = h_init

    w_xi = weight_variable([inp_dim, hid_dim], wd=wd,
                           name='w_xi_{}'.format(name))
    w_hi = weight_variable([hid_dim, hid_dim], wd=wd,
                           name='w_hi_{}'.format(name))
    b_i = weight_variable([hid_dim], wd=wd, name='b_i_{}'.format(name))

    w_xu = weight_variable([inp_dim, hid_dim], wd=wd,
                           name='w_xu_{}'.format(name))
    w_hu = weight_variable([hid_dim, hid_dim], wd=wd,
                           name='w_hu_{}'.format(name))
    b_u = weight_variable([hid_dim], wd=wd, name='b_u_{}'.format(name))

    w_xr = weight_variable([inp_dim, hid_dim], wd=wd,
                           name='w_xr_{}'.format(name))
    w_hr = weight_variable([hid_dim, hid_dim], wd=wd,
                           name='w_hr_{}'.format(name))
    b_r = weight_variable([hid_dim], wd=wd, name='b_r_{}'.format(name))

    def unroll(inp, time):
        t = time

        # [B, H]
        print 'inp', inp.get_shape()
        print 'w_xi', w_xi.get_shape()
        print 'h', h[t - 1].get_shape()
        print 'w_hi', w_hi.get_shape()
        g_i[t] = tf.sigmoid(tf.matmul(inp, w_xi) +
                            tf.matmul(h[t - 1], w_hi) +
                            b_i,
                            name='g_i_{}_{}'.format(name, t))
        # [B, H]
        g_r[t] = tf.sigmoid(tf.matmul(inp, w_xr) +
                            tf.matmul(h[t - 1], w_hr) +
                            b_r,
                            name='g_r_{}_{}'.format(name, t))
        # [B, H]
        u = tf.tanh(tf.matmul(inp, w_xu) + g_r[t] *
                    (tf.matmul(h[t - 1], w_hu)) +
                    b_u,
                    name='u_{}_{}'.format(name, t))
        # [B, He]
        h[t] = tf.add(h[t - 1] * (1 - g_i[t]), u * g_i[t],
                      name='h_{}_{}'.format(name, t))

    model['w_xi_{}'.format(name)] = w_xi
    model['w_hi_{}'.format(name)] = w_hi
    model['b_i_{}'.format(name)] = b_i
    model['w_xu_{}'.format(name)] = w_xu
    model['w_hu_{}'.format(name)] = w_hu
    model['b_u_{}'.format(name)] = b_u
    model['w_xr_{}'.format(name)] = w_xr
    model['w_hr_{}'.format(name)] = w_hr
    model['b_r_{}'.format(name)] = b_r
    model['g_i_{}'.format(name)] = g_i
    model['g_r_{}'.format(name)] = g_i
    model['h_{}'.format(name)] = h

    return unroll
