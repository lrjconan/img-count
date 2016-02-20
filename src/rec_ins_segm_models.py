import cslab_environ

from tensorflow.python import control_flow_ops
from tensorflow.python.framework import ops
import numpy as np
import tensorflow as tf

from utils import logger
from utils.grad_clip_optim import GradientClipOptimizer

log = logger.get()


def get_model(name, opt, device='/cpu:0', train=True):
    """Model router."""
    if name == 'original':
        return get_orig_model(opt, device=device, train=train)
    elif name == 'attention':
        return get_attn_model(opt, device=device, train=train)
    else:
        raise Exception('Unknown model name "{}"'.format(name))

    pass


def _batch_norm(x, n_out, phase_train, scope='bn', affine=True):
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
        mean, var = control_flow_ops.cond(phase_train[0],
                                          mean_var_with_update,
                                          lambda: (ema_mean, ema_var))

        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
                                                            beta, gamma, 1e-3, affine)
    return normed


# Register gradient for Hungarian algorithm.
ops.NoGradient("Hungarian")


# Register gradient for cumulative minimum operation.
@ops.RegisterGradient("CumMin")
def _cum_min_grad(op, grad):
    """The gradients for `cum_min`.

    Args:
        op: The `cum_min` `Operation` that we are differentiating, which we can
        use to find the inputs and outputs of the original op.
        grad: Gradient with respect to the output of the `cum_min` op.

    Returns:
        Gradients with respect to the input of `cum_min`.
    """
    x = op.inputs[0]
    return [tf.user_ops.cum_min_grad(grad, x)]


def _get_device_fn(device):
    """Choose device for different ops."""
    OPS_ON_CPU = set(['ResizeBilinear', 'ResizeBilinearGrad', 'CumMin',
                      'CumMinGrad', 'Hungarian', 'Reverse'])

    def _device_fn(op):
        if op.type in OPS_ON_CPU:
            return "/cpu:0"
        else:
            # Other ops will be placed on GPU if available, otherwise
            # CPU.
            return device

    return _device_fn


def _conv2d(x, w):
    """2-D convolution."""
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def _max_pool(x, ratio):
    """2 x 2 max pooling."""
    return tf.nn.max_pool(x, ksize=[1, ratio, ratio, 1],
                          strides=[1, ratio, ratio, 1], padding='SAME')


def _weight_variable(shape, wd=None, name=None):
    """Initialize weights."""
    initial = tf.truncated_normal(shape, stddev=0.01)
    var = tf.Variable(initial, name=name)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _add_lstm(model, timespan, inp_dim, hid_dim, c_init, h_init, wd=None, name=None):
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
    w_xi = _weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                            wd=wd, name='w_xi_{}'.format(name))
    w_hi = _weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                            wd=wd, name='w_hi_{}'.format(name))
    b_i = _weight_variable([hid_depth],
                           wd=wd, name='b_i_{}'.format(name))

    # Forget gate
    w_xf = _weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                            wd=wd, name='w_xf_{}'.format(name))
    w_hf = _weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                            wd=wd, name='w_hf_{}'.format(name))
    b_f = _weight_variable([hid_depth],
                           wd=wd, name='b_f_{}'.format(name))

    # Input activation
    w_xu = _weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                            wd=wd, name='w_xu_{}'.format(name))
    w_hu = _weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                            wd=wd, name='w_hu_{}'.format(name))
    b_u = _weight_variable([hid_depth],
                           wd=wd, name='b_u_{}'.format(name))

    # Output gate
    w_xo = _weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                            wd=wd, name='w_xo_{}'.format(name))
    w_ho = _weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                            wd=wd, name='w_ho_{}'.format(name))
    b_o = _weight_variable([hid_depth], name='b_o_{}'.format(name))

    def unroll(inp, time):
        t = time
        g_i[t] = tf.sigmoid(_conv2d(inp, w_xi) + _conv2d(h[t - 1], w_hi) + b_i)
        g_f[t] = tf.sigmoid(_conv2d(inp, w_xf) + _conv2d(h[t - 1], w_hf) + b_f)
        g_o[t] = tf.sigmoid(_conv2d(inp, w_xo) + _conv2d(h[t - 1], w_ho) + b_o)
        u[t] = tf.tanh(_conv2d(inp, w_xu) + _conv2d(h[t - 1], w_hu) + b_u)
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


def _add_conv_lstm(model, timespan, inp_height, inp_width, inp_depth, filter_size, hid_depth, c_init, h_init, wd=None, name=''):
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
    w_xi = _weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                            wd=wd, name='w_xi_{}'.format(name))
    w_hi = _weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                            wd=wd, name='w_hi_{}'.format(name))
    b_i = _weight_variable([hid_depth],
                           wd=wd, name='b_i_{}'.format(name))

    # Forget gate
    w_xf = _weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                            wd=wd, name='w_xf_{}'.format(name))
    w_hf = _weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                            wd=wd, name='w_hf_{}'.format(name))
    b_f = _weight_variable([hid_depth],
                           wd=wd, name='b_f_{}'.format(name))

    # Input activation
    w_xu = _weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                            wd=wd, name='w_xu_{}'.format(name))
    w_hu = _weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                            wd=wd, name='w_hu_{}'.format(name))
    b_u = _weight_variable([hid_depth],
                           wd=wd, name='b_u_{}'.format(name))

    # Output gate
    w_xo = _weight_variable([filter_size, filter_size, inp_depth, hid_depth],
                            wd=wd, name='w_xo_{}'.format(name))
    w_ho = _weight_variable([filter_size, filter_size, hid_depth, hid_depth],
                            wd=wd, name='w_ho_{}'.format(name))
    b_o = _weight_variable([hid_depth], name='b_o_{}'.format(name))

    def unroll(inp, time):
        t = time
        g_i[t] = tf.sigmoid(_conv2d(inp, w_xi) + _conv2d(h[t - 1], w_hi) + b_i)
        g_f[t] = tf.sigmoid(_conv2d(inp, w_xf) + _conv2d(h[t - 1], w_hf) + b_f)
        g_o[t] = tf.sigmoid(_conv2d(inp, w_xo) + _conv2d(h[t - 1], w_ho) + b_o)
        u[t] = tf.tanh(_conv2d(inp, w_xu) + _conv2d(h[t - 1], w_hu) + b_u)
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


def _add_cnn(model, x, f, ch, pool, use_bn=False, phase_train=None, wd=None):
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
    hc = [None] * nlayers
    hp = [None] * nlayers

    for ii in xrange(nlayers):
        w[ii] = _weight_variable([f[ii], f[ii], ch[ii], ch[ii + 1]], wd=wd)
        b[ii] = _weight_variable([ch[ii + 1]], wd=wd)
        if ii == 0:
            if use_bn:
                hc[ii] = tf.nn.relu(_batch_norm(
                    _conv2d(x, w[ii]) + b[ii], ch[ii + 1], phase_train))
            else:
                hc[ii] = tf.nn.relu(_conv2d(x, w[ii]) + b[ii])
        else:
            if use_bn:
                hc[ii] = tf.nn.relu(_batch_norm(
                    _conv2d(hp[ii - 1], w[ii]) + b[ii], ch[ii + 1], phase_train))
            else:
                hc[ii] = tf.nn.relu(_conv2d(hp[ii - 1], w[ii]) + b[ii])
        if pool[ii] > 1:
            hp[ii] = _max_pool(hc[ii], pool[ii])
        else:
            hp[ii] = hc[ii]

    return hp[-1]


def _add_dcnn(model, x, f, ch, pool, wd=None):
    """Add D-CNN, with standard DeConv-Relu layers.

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
    out_shape = [None] * nlayers

    batch = tf.shape(x)[0: 1]
    inp_size = tf.shape(x)[1: 3]
    cum_pool = 1

    for ii in xrange(nlayers):
        cum_pool *= pool[ii]
        out_shape[ii] = tf.concat(
            0, [batch, inp_size * cum_pool, tf.constant(ch[ii: ii + 1])])
        w[ii] = _weight_variable([f[ii], f[ii], ch[ii], ch[ii + 1]], wd=wd)
        b[ii] = _weight_variable([ch[ii + 1]], wd=wd)
        if ii == 0:
            # h[ii] = tf.nn.conv2d_transpose(
            #     x, w[ii], out_shape[ii],
            #     strides=[1, pool[ii], pool[ii], 1])
            h[ii] = tf.nn.conv2d_transpose(
                x, w[ii], out_shape[ii],
                strides=[1, pool[ii], pool[ii], 1]) + b[ii]
        else:
            # h[ii] = tf.nn.conv2d_transpose(
            #     h[ii - 1], w[ii], out_shape[ii],
            #     strides=[1, pool[ii], pool[ii], 1])
            h[ii] = tf.nn.conv2d_transpose(
                h[ii - 1], w[ii], out_shape[ii],
                strides=[1, pool[ii], pool[ii], 1]) + b[ii]

    return h[-1]


def _add_attn_controller_rnn(model, timespan, inp_width, inp_height, ctl_inp_dim, filter_size, wd=None, name='', sess=None, train_model=None):
    """Add an attention controller."""
    # Constant for computing filter_x. Shape: [1, W, 1].
    span_x = np.reshape(np.arange(inp_width), [1, inp_width, 1])
    # Constant for computing filter_y. Shape: [1, H, 1].
    span_y = np.reshape(np.arange(inp_height), [1, inp_height, 1])
    # Constant for computing filter_x. Shape: [1, 1, Fr].
    span_filter = np.reshape(np.arange(filter_size) + 1, [1, 1, -1])

    # Control variale. [g_x, g_y, log_var, log_delta, log_gamma].
    # Shape: T * [B, 5].
    ctl = [None] * timespan
    # Attention center x. Shape: T * [B].
    ctr_x = [None] * timespan
    # Attention center y. Shape: T * [B].
    ctr_y = [None] * timespan
    # Attention width. Shape: T * [B].
    delta = [None] * timespan
    # Log of the variance of Gaussian filter. Shape: T * [B].
    lg_var = [None] * timespan
    # Log of the multiplier of Gaussian filter. Shape: T * [B].
    lg_gamma = [None] * timespan
    # Gaussian filter mu_x. Shape: T * [B, W, Fr].
    mu_x = [None] * timespan
    # Gaussian filter mu_y. Shape: T * [B, H, Fr].
    mu_y = [None] * timespan
    # Gaussian filter on x direction. Shape: T * [B, W, Fr].
    filter_x = [None] * timespan
    # Gaussian filter on y direction. Shape: T * [B, H, Fr].
    filter_y = [None] * timespan

    # From hidden decoder to controller variables. Shape: [Hd, 5].
    w_ctl = weight_variable(
        [ctl_inp_dim, 5], wd=wd, name='w_ctl_{}'.format(name),
        sess=sess, train_model=train_model)
    # Controller variable bias. Shape: [5].
    b_ctl = weight_variable([5], wd=wd, name='b_ctl_{}'.format(name),
                            sess=sess, train_model=train_model)

    def unroll(ctl_inp, time):
        """Run controller."""
        t = time
        #########################################################
        # Attention controller
        #########################################################
        # [B, 5]
        ctl[t] = tf.matmul(ctl_inp, w_ctl) + b_ctl
        # [B, 1, 1]
        ctr_x[t] = tf.reshape((inp_width + 1) / 2.0 *
                              (ctl[t][:, 0] + 1),
                              [-1, 1, 1],
                              name='ctr_x_{}_{}'.format(name, t))
        # [B, 1, 1]
        ctr_y[t] = tf.reshape((inp_height + 1) / 2.0 *
                              (ctl[t][:, 1] + 1),
                              [-1, 1, 1],
                              name='ctr_y_{}_{}'.format(name, t))
        # [B, 1, 1]
        delta[t] = tf.reshape((max(inp_width, inp_height) - 1) /
                              ((filter_size - 1) *
                               tf.exp(ctl[t][:, 3])),
                              [-1, 1, 1],
                              name='delta_{}_{}'.format(name, t))
        # [B, 1, 1]
        lg_var[t] = tf.reshape(ctl[t][:, 2], [-1, 1, 1],
                               name='lg_var_{}_{}'.format(name, t))
        # [B, 1, 1]
        lg_gamma[t] = tf.reshape(ctl[t][:, 4], [-1, 1, 1],
                                 name='lg_gamma_{}_{}'.format(name, t))

        #########################################################
        # Gaussian filter
        #########################################################
        # [B, 1, 1] + [B, 1, 1] * [1, Fr, 1] = [B, 1, Fr]
        mu_x[t] = tf.add(ctr_x[t], delta[t] * (
            span_filter - filter_size / 2.0 - 0.5),
            name='mu_x_{}_{}'.format(name, t))
        # [B, 1, 1] + [B, 1, 1] * [1, 1, Fr] = [B, 1, Fr]
        mu_y[t] = tf.add(ctr_y[t], delta[t] * (
            span_filter - filter_size / 2.0 - 0.5),
            name='mu_y_{}_{}'.format(name, t))
        # [B, 1, 1] * [1, W, 1] - [B, 1, Fr] = [B, W, Fr]
        filter_x[t] = tf.mul(
            1 / tf.sqrt(tf.exp(lg_var[t])) / tf.sqrt(2 * np.pi),
            tf.exp(-0.5 * (span_x - mu_x[t]) * (span_x - mu_x[t]) /
                   tf.exp(lg_var[t])),
            name='filter_x_{}_{}'.format(name, t))
        # [1, H, 1] - [B, 1, Fr] = [B, H, Fr]
        filter_y[t] = tf.mul(
            1 / tf.sqrt(tf.exp(lg_var[t])) / tf.sqrt(2 * np.pi),
            tf.exp(-0.5 * (span_y - mu_y[t]) * (span_y - mu_y[t]) /
                   tf.exp(lg_var[t])),
            name='filter_y_{}_{}'.format(name, t))

        pass

    model['w_ctl_{}'.format(name)] = w_ctl
    model['b_ctl_{}'.format(name)] = b_ctl
    model['ctl_{}'.format(name)] = ctl
    model['ctr_x_{}'.format(name)] = ctr_x
    model['ctr_y_{}'.format(name)] = ctr_y
    model['delta_{}'.format(name)] = delta
    model['lg_var_{}'.format(name)] = lg_var
    model['lg_gamma_{}'.format(name)] = lg_gamma
    model['mu_x_{}'.format(name)] = mu_x
    model['mu_y_{}'.format(name)] = mu_y
    model['filter_x_{}'.format(name)] = filter_x
    model['filter_y_{}'.format(name)] = filter_y

    return unroll


def _cum_min(s, d):
    """Calculates cumulative minimum.

    Args:
        s: Input matrix [B, D].
        d: Second dim.

    Returns:
        s_min: [B, D], cumulative minimum accross the second dim.
    """
    s_min_list = [None] * d
    s_min_list[0] = s[:, 0: 1]
    for ii in xrange(1, d):
        s_min_list[ii] = tf.minimum(s_min_list[ii - 1], s[:, ii: ii + 1])

    return tf.concat(1, s_min_list)


def _f_iou(a, b, timespan, pairwise=False):
    """
    Computes IOU score.

    Args:
        a: [B, N, H, W], or [N, H, W], or [H, W]
        b: [B, N, H, W], or [N, H, W], or [H, W]
           in pairwise mode, the second dimension can be different,
           e.g. [B, M, H, W], or [M, H, W], or [H, W]
        pairwise: whether the inputs are already aligned, outputs [B, N] or
                  the inputs are orderless, outputs [B, N, M].
    """
    eps = 1e-5

    def _get_reduction_indices(a):
        """Gets the list of axes to sum over."""
        dim = len(a.get_shape())

        return [dim - 2, dim - 1]

    def _inter(a, b):
        """Computes intersection."""
        reduction_indices = _get_reduction_indices(a)
        return tf.reduce_sum(a * b, reduction_indices=reduction_indices)

    def _union(a, b):
        """Computes union."""
        reduction_indices = _get_reduction_indices(a)
        return tf.reduce_sum(a + b - (a * b) + eps,
                             reduction_indices=reduction_indices)
    if pairwise:
        # b_shape = tf.shape(b)
        # # [1, 1, M, 1, 1]
        # a_shape2 = tf.concat(0, [tf.constant([1, 1]),
        #                          b_shape[1: 2],
        #                          tf.constant([1, 1])])
        # # [B, N, H, W] => [B, N, 1, H, W] => [B, N, M, H, W]
        # a = tf.expand_dims(a, 2)
        # # [B, M, H, W] => [B, 1, M, H, W]
        # b = tf.expand_dims(b, 1)
        # a = tf.tile(a, a_shape2)
        # return _inter(a, b) / _union(a, b)

        y_list = [None] * timespan
        a_list = [None] * timespan
        # [B, N, H, W] => [B, N, 1, H, W]
        a = tf.expand_dims(a, 2)
        # [B, N, 1, H, W] => N * [B, 1, 1, H, W]
        a_list = tf.split(1, timespan, a)
        # [B, M, H, W] => [B, 1, M, H, W]
        b = tf.expand_dims(b, 1)

        for ii in xrange(timespan):
            # [B, 1, M]
            y_list[ii] = _inter(a_list[ii], b) / _union(a_list[ii], b)

        # N * [B, 1, M] => [B, N, M]
        return tf.concat(1, y_list)

    else:
        return _inter(a, b) / _union(a, b)


def _add_ins_segm_loss(model, y_out, y_gt, s_out, s_gt, r, timespan, use_cum_min=True, segm_loss_fn='iou'):
    """
    Instance segmentation loss.

    Args:
        y_out: [B, N, H, W], output segmentations.
        y_gt: [B, M, H, W], groundtruth segmentations.
        s_out: [B, N], output confidence score.
        s_gt: [B. M], groundtruth confidence score.
        r: float, mixing coefficient for combining segmentation loss and
        confidence score loss.
    """

    def _bce(y_out, y_gt):
        """Binary cross entropy."""
        eps = 1e-7
        return -y_gt * tf.log(y_out + eps) - \
            (1 - y_gt) * tf.log(1 - y_out + eps)

    def _match_bce(y_out, y_gt, match, timespan):
        """Binary cross entropy with matching.

        Args:
            y_out: [B, N, H, W]
            y_gt: [B, N, H, W]
            match: [B, N, N]
            match_count: [B]
            num_ex: [1]
            timespan: N
        """
        # N * [B, 1, H, W]
        y_out_list = tf.split(1, timespan, y_out)
        # N * [B, 1, N]
        match_list = tf.split(1, timespan, match)
        bce_list = [None] * timespan
        shape = tf.shape(y_out)
        num_ex = tf.to_float(shape[0])
        height = tf.to_float(shape[2])
        width = tf.to_float(shape[3])

        # [B, N, M] => [B, N]
        match_sum = tf.reduce_sum(match, reduction_indices=[2])
        # [B, N] => [B]
        match_count = tf.reduce_sum(match_sum, reduction_indices=[1])

        for ii in xrange(timespan):
            # [B, 1, H, W] * [B, N, H, W] => [B, N, H, W] => [B, N]
            # [B, N] * [B, N] => [B]
            # [B] => [B, 1]
            bce_list[ii] = tf.expand_dims(tf.reduce_sum(tf.reduce_sum(
                _bce(y_out_list[ii], y_gt), reduction_indices=[2, 3]) *
                tf.reshape(match_list[ii], [-1, timespan]),
                reduction_indices=[1]), 1)

        # N * [B, 1] => [B, N] => [B]
        bce_total = tf.reduce_sum(
            tf.concat(1, bce_list), reduction_indices=[1])

        return tf.reduce_sum(bce_total / match_count) / num_ex / height / width

    # IOU score, [B, N, M]
    iou_soft = _f_iou(y_out, y_gt, timespan, pairwise=True)

    # Matching score, [B, N, M]
    # Add small epsilon because the matching algorithm only accepts complete
    # bipartite graph with positive weights.
    # Mask out the items beyond the total groudntruth count.
    # Mask X, [B, M] => [B, 1, M]
    mask_x = tf.expand_dims(s_gt, dim=1)
    # Mask Y, [B, M] => [B, N, 1]
    mask_y = tf.expand_dims(s_gt, dim=2)
    iou_mask = iou_soft * mask_x * mask_y

    # Keep certain precision so that we can get optimal matching within
    # reasonable time.
    eps = 1e-5
    precision = 1e6
    iou_mask = tf.round(iou_mask * precision) / precision
    match_eps = tf.user_ops.hungarian(iou_mask + eps)[0]

    # [1, N, 1, 1]
    y_out_shape = tf.shape(y_out)
    num_segm_out = y_out_shape[1: 2]
    num_segm_out_mul = tf.concat(
        0, [tf.constant([1]), num_segm_out, tf.constant([1])])
    # Mask the graph algorithm output.
    match = match_eps * mask_x * mask_y
    model['match'] = match
    # [B, N, M] => [B, N]
    match_sum = tf.reduce_sum(match, reduction_indices=[2])
    # [B, N] => [B]
    match_count = tf.reduce_sum(match_sum, reduction_indices=[1])

    # Loss for confidence scores.
    if use_cum_min:
        # [B, N]
        s_out_min = _cum_min(s_out, timespan)
        # [B, N]
        s_bce = _bce(s_out_min, match_sum)
        model['s_out_min'] = s_out_min
    else:
        # Try simply do binary xent for matching sequence.
        s_bce = _bce(s_out, match_sum)
    model['s_bce'] = s_bce

    # Loss normalized by number of examples.
    y_gt_shape = tf.shape(y_gt)
    num_ex = tf.to_float(y_gt_shape[0])
    max_num_obj = tf.to_float(y_gt_shape[1])

    # IOU
    iou_hard = _f_iou(tf.to_float(y_out > 0.5), y_gt, timespan, pairwise=True)
    # [B, M, N] * [B, M, N] => [B] * [B] => [1]
    iou_hard = tf.reduce_sum(tf.reduce_sum(
        iou_hard * match, reduction_indices=[1, 2]) / match_count) / num_ex
    iou_soft = tf.reduce_sum(tf.reduce_sum(
        iou_soft * match, reduction_indices=[1, 2]) / match_count) / num_ex
    model['iou_hard'] = iou_hard
    model['iou_soft'] = iou_soft

    # [B, N, M] => scalar
    conf_loss = r * tf.reduce_sum(s_bce) / num_ex / max_num_obj
    if segm_loss_fn == 'iou':
        segm_loss = -iou_soft
    elif segm_loss_fn == 'bce':
        segm_loss = _match_bce(y_out, y_gt, match, timespan)

    model['segm_loss'] = segm_loss
    loss = segm_loss + conf_loss

    model['conf_loss'] = conf_loss
    model['loss'] = loss

    # Counting accuracy
    count_out = tf.reduce_sum(tf.to_float(s_out > 0.5), reduction_indices=[1])
    count_gt = tf.reduce_sum(s_gt, reduction_indices=[1])
    count_acc = tf.reduce_sum(tf.to_float(
        tf.equal(count_out, count_gt))) / num_ex
    model['count_out'] = count_out
    model['count_gt'] = count_gt
    model['count_acc'] = count_acc

    return loss


def get_orig_model(opt, device='/cpu:0', train=True):
    """The original model.
    # Original model:
    #           ---
    #           | |
    # CNN -> -> RNN -> Instances
    """
    model = {}
    timespan = opt['timespan']
    inp_height = opt['inp_height']
    inp_width = opt['inp_width']
    conv_lstm_filter_size = opt['conv_lstm_filter_size']
    conv_lstm_hid_depth = opt['conv_lstm_hid_depth']
    wd = opt['weight_decay']
    store_segm_map = ('store_segm_map' not in opt) or opt['store_segm_map']

    with tf.device(_get_device_fn(device)):
        # Input image, [B, H, W, 3]
        x = tf.placeholder('float', [None, inp_height, inp_width, 3])
        # Whether in training stage, required for batch norm.
        phase_train_f = tf.placeholder('float')
        phase_train = tf.cast(phase_train_f, 'bool')
        # Groundtruth segmentation maps, [B, T, H, W]
        y_gt = tf.placeholder('float', [None, timespan, inp_height, inp_width])
        # Groundtruth confidence score, [B, T]
        s_gt = tf.placeholder('float', [None, timespan])
        y_gt_list = tf.split(1, timespan, y_gt)
        model['x'] = x
        model['phase_train'] = phase_train_f
        model['y_gt'] = y_gt
        model['s_gt'] = s_gt

        # Possibly add random image transformation layers here in training time.
        # Need to combine x and y together to crop.
        # Other operations on x only.
        # x = tf.image.random_crop()
        # x = tf.image.random_flip()

        # CNN
        # [B, H, W, 3] => [B, H / 2, W / 2, 16]
        # [B, H / 2, W / 2, 16] => [B, H / 4, W / 4, 32]
        # [B, H / 4, W / 4, 32] => [B, H / 8, W / 8, 64]
        cnn_filt = [3, 3, 3]
        cnn_channels = [3, 16, 32, 64]
        cnn_pool = [2, 2, 1]
        h_pool3 = _add_cnn(model, x, cnn_filt, cnn_channels, cnn_pool,
                           use_bn=opt['use_bn'], phase_train=phase_train, wd=wd)

        if store_segm_map:
            lstm_inp_depth = cnn_channels[-1] + 1
        else:
            lstm_inp_depth = cnn_channels[-1]

        lstm_depth = 16
        subsample = np.array(cnn_pool).prod()
        lstm_height = inp_height / subsample
        lstm_width = inp_width / subsample

        # ConvLSTM hidden state initialization
        # [B, LH, LW, LD]
        x_shape = tf.shape(x)
        num_ex = x_shape[0: 1]
        c_init = tf.zeros(tf.concat(
            0, [num_ex, tf.constant([lstm_height, lstm_width, lstm_depth])]))
        h_init = tf.zeros(tf.concat(
            0, [num_ex, tf.constant([lstm_height, lstm_width, lstm_depth])]))

        # Segmentation network
        # 4th convolution layer (on ConvLSTM output).
        w_conv4 = _weight_variable([3, 3, lstm_depth, 1], wd=wd)
        b_conv4 = _weight_variable([1], wd=wd)

        # Bias towards segmentation output.
        b_5 = _weight_variable([lstm_height * lstm_width], wd=wd)

        # Confidence network
        # Linear layer for output confidence score.
        w_6 = _weight_variable(
            [lstm_height * lstm_width / 16 * lstm_depth, 1], wd=wd)
        b_6 = _weight_variable([1], wd=wd)

        unroll_conv_lstm = _add_conv_lstm(
            model=model,
            timespan=timespan,
            inp_height=lstm_height,
            inp_width=lstm_width,
            inp_depth=lstm_inp_depth,
            filter_size=conv_lstm_filter_size,
            hid_depth=lstm_depth,
            c_init=c_init,
            h_init=h_init,
            wd=wd,
            name='lstm'
        )
        h_lstm = model['h_lstm']

        h_conv4 = [None] * timespan
        segm_lo = [None] * timespan
        # segm_out = [None] * timespan
        score = [None] * timespan
        h_pool4 = [None] * timespan
        segm_canvas = [None] * timespan
        segm_canvas[0] = tf.zeros(tf.concat(
            0, [num_ex, tf.constant([lstm_height, lstm_width, 1])]))
        lstm_inp = [None] * timespan
        y_out = [None] * timespan

        for t in xrange(timespan):
            # If we also send the cumulative output maps.
            if store_segm_map:
                lstm_inp[t] = tf.concat(3, [h_pool3, segm_canvas[t]])
            else:
                lstm_inp[t] = h_pool3
            unroll_conv_lstm(lstm_inp[t], time=t)

            # Segmentation network
            # [B, LH, LW, 1]
            h_conv4 = tf.nn.relu(_conv2d(h_lstm[t], w_conv4) + b_conv4)
            # [B, LH * LW]
            h_conv4_reshape = tf.reshape(
                h_conv4, [-1, lstm_height * lstm_width])
            # [B, LH * LW] => [B, LH, LW] => [B, 1, LH, LW]
            # [B, LH * LW] => [B, LH, LW] => [B, LH, LW, 1]
            # segm_lo[t] = tf.expand_dims(tf.reshape(tf.sigmoid(
            #     tf.log(tf.nn.softmax(h_conv4_reshape)) + b_5),
            #     [-1, 1, lstm_height, lstm_width]), dim=3)

            # Without sigmoid in the RNN.
            segm_lo[t] = tf.expand_dims(tf.reshape(
                tf.log(tf.nn.softmax(h_conv4_reshape)) + b_5,
                [-1, 1, lstm_height, lstm_width]), dim=3)

            # [B, LH, LW, 1]
            if t != timespan - 1:
                segm_canvas[t + 1] = segm_canvas[t] + segm_lo[t]

            # Objectness network
            # [B, LH, LW, LD] => [B, LLH, LLW, LD] => [B, LLH * LLW * LD]
            h_pool4[t] = tf.reshape(_max_pool(h_lstm[t], 4),
                                    [-1,
                                     lstm_height * lstm_width / 16 * lstm_depth])
            # [B, LLH * LLW * LD] => [B, 1]
            score[t] = tf.sigmoid(tf.matmul(h_pool4[t], w_6) + b_6)

        # [B * T, LH, LW, 1]
        segm_lo_all = tf.reshape(
            tf.concat(1, segm_lo), [-1, lstm_height, lstm_width, 1])

        # [B * T, LH, LW, 1] => [B * T, H, W, 1] => [B, T, H, W]
        # Use deconvolution to upsample.
        if opt['use_deconv']:
            dcnn_filters = [3, 3]
            dcnn_channels = [1, 1, 1]
            dcnn_unpool = [2, 2]
            h_dc_ = _add_dcnn(model, segm_lo_all, dcnn_filters, dcnn_channels,
                              dcnn_unpool,  wd=wd)
            # if opt['use_bn']:
            #     h_dc = _batch_norm(h_dc_, dcnn_channels[-1], phase_train)
            # else:
            #     h_dc = h_dc_
            h_dc = h_dc_

            # Add sigmoid outside RNN.
            y_out = tf.reshape(tf.sigmoid(h_dc),
                               [-1, timespan, inp_height, inp_width])
            # y_out = tf.reshape(h_dc, [-1, timespan, inp_height, inp_width])
        else:
            y_out = tf.reshape(
                tf.image.resize_bilinear(segm_lo_all, [inp_height, inp_width]),
                [-1, timespan, inp_height, inp_width])

        model['y_out'] = y_out

        # T * [B, 1] = [B, T]
        s_out = tf.concat(1, score)
        model['s_out'] = s_out

        model['h_lstm_0'] = h_lstm[0]
        model['h_pool4_0'] = h_pool4[0]
        model['s_0'] = score[0]

        # Loss function
        if train:
            r = opt['loss_mix_ratio']
            lr = opt['learning_rate']
            use_cum_min = ('cum_min' not in opt) or opt['cum_min']
            eps = 1e-7
            loss = _add_ins_segm_loss(
                model, y_out, y_gt, s_out, s_gt, r, timespan,
                use_cum_min=use_cum_min,
                segm_loss_fn=opt['segm_loss_fn'])
            tf.add_to_collection('losses', loss)
            total_loss = tf.add_n(tf.get_collection(
                'losses'), name='total_loss')
            model['total_loss'] = total_loss

            train_step = GradientClipOptimizer(
                tf.train.AdamOptimizer(lr, epsilon=eps),
                clip=1.0).minimize(total_loss)
            model['train_step'] = train_step

    return model


def get_attn_model(opt, device='/cpu:0', train=True):
    """Attention-based model."""
    model = {}
    timespan = opt['timespan']
    inp_height = opt['inp_height']
    inp_width = opt['inp_width']
    conv_lstm_filter_size = opt['conv_lstm_filter_size']
    conv_lstm_hid_depth = opt['conv_lstm_hid_depth']
    wd = opt['weight_decay']
    store_segm_map = ('store_segm_map' not in opt) or opt['store_segm_map']

    with tf.device(_get_device_fn(device)):
        pass

    # Loss function
    if train:
        r = opt['loss_mix_ratio']
        lr = opt['learning_rate']
        use_cum_min = ('cum_min' not in opt) or opt['cum_min']
        eps = 1e-7
        loss = _add_ins_segm_loss(
            model, y_out, y_gt, s_out, s_gt, r, timespan,
            use_cum_min=use_cum_min,
            segm_loss_fn=opt['segm_loss_fn'])
        tf.add_to_collection('losses', loss)
        total_loss = tf.add_n(tf.get_collection(
            'losses'), name='total_loss')
        model['total_loss'] = total_loss

        train_step = GradientClipOptimizer(
            tf.train.AdamOptimizer(lr, epsilon=eps),
            clip=1.0).minimize(total_loss)
        model['train_step'] = train_step

    return model
