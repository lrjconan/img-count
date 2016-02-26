import cslab_environ

from tensorflow.python.framework import ops
import numpy as np
import tensorflow as tf

from utils import logger
from utils.grad_clip_optim import GradientClipOptimizer
import nnlib as nn
import image_ops as img

log = logger.get()


def get_model(name, opt, device='/cpu:0'):
    """Model router."""
    if name == 'original':
        return get_orig_model(opt, device=device)
    elif name == 'attention':
        return get_attn_model(opt, device=device)
    else:
        raise Exception('Unknown model name "{}"'.format(name))

    pass


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
    OPS_ON_CPU = set(['ResizeBilinear', 'ResizeBilinearGrad', 'Mod', 'CumMin',
                      'CumMinGrad', 'Hungarian', 'Reverse', 'SparseToDense'])

    def _device_fn(op):
        if op.type in OPS_ON_CPU:
            return "/cpu:0"
        else:
            # Other ops will be placed on GPU if available, otherwise
            # CPU.
            return device

    return _device_fn


def _add_attn_controller_rnn(model, timespan, inp_width, inp_height, ctl_inp_dim, filter_size, wd=None, name=''):
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
    w_ctl = nn.weight_variable(
        [ctl_inp_dim, 5], wd=wd, name='w_ctl_{}'.format(name))
    # Controller variable bias. Shape: [5].
    b_ctl = nn.weight_variable([5], wd=wd, name='b_ctl_{}'.format(name))

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
        # a_shape = tf.shape(a)
        # b_shape = tf.shape(b)
        # zeros_a = tf.zeros(tf.concat(0, [a_shape[0: 1], b_shape[1: 2], a_shape[1: ]]))
        # # [B, N, H, W] => [B, N, 1, H, W]
        # a = tf.expand_dims(a, 2)
        # a = a + zeros_a
        # # [B, M, H, W] => [B, 1, M, H, W]
        # b = tf.expand_dims(b, 1)
        # return _inter(a, b) / _union(a, b)

        # N * [B, 1, M]
        y_list = [None] * timespan
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
    num_segm_out = y_out_shape[1]
    num_segm_out_mul = tf.pack([1, num_segm_out, 1])
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


def _add_rnd_img_transformation(model, x, y_gt, padding, phase_train):
    """
    Perform random crop, flip, transpose, hue, saturation, brightness, contrast.
    """
    # Random image transformation layers.
    phase_train_f = tf.to_float(phase_train)
    x_shape = tf.shape(x)
    num_ex = x_shape[0]
    full_height = x_shape[1]
    full_width = x_shape[2]
    inp_height = full_height - 2 * padding
    inp_width = full_width - 2 * padding
    inp_depth = x_shape[3]

    # Random crop
    offset = tf.random_uniform([2], dtype='int32', maxval=padding * 2)
    x_rand = tf.slice(x, tf.pack([0, offset[0], offset[1], 0]),
                      tf.pack([-1, inp_height, inp_width, inp_depth]))
    y_rand = tf.slice(y_gt, tf.pack([0, 0, offset[0], offset[1]]),
                      tf.pack([-1, -1, inp_height, inp_width]))

    # Center slices (for inference)
    x_ctr = tf.slice(x, [0, padding, padding, 0],
                     tf.pack([-1, inp_height, inp_width, -1]))
    y_ctr = tf.slice(y_gt, [0, 0, padding, padding],
                     tf.pack([-1, -1, inp_height, inp_width]))

    # Random horizontal & vertical flip & transpose
    rand = tf.random_uniform([3], 0, 1.0)
    mirror_x = tf.pack([1.0, rand[0], rand[1], 1.0]) < 0.5
    mirror_y = tf.pack([1.0, 1.0, rand[0], rand[1]]) < 0.5
    x_rand = tf.reverse(x_rand, mirror_x)
    y_rand = tf.reverse(y_rand, mirror_y)
    do_tr = tf.cast(rand[2] > 0.5, 'int32')
    x_rand = tf.transpose(x_rand, tf.pack([0, 1 + do_tr, 2 - do_tr, 3]))
    y_rand = tf.transpose(y_rand, tf.pack([0, 1, 2 + do_tr, 3 - do_tr]))

    # Random hue, saturation, brightness, contrast
    x_rand = img.random_hue(x_rand, 0.5)
    x_rand = img.random_saturation(x_rand, 0.5, 2.0)
    x_rand = tf.image.random_brightness(x_rand, 0.5)
    x_rand = tf.image.random_contrast(x_rand, 0.5, 2.0)

    x = (1.0 - phase_train_f) * x_ctr + phase_train_f * x_rand
    y_gt = (1.0 - phase_train_f) * y_ctr + phase_train_f * y_rand

    return x, y_gt


def get_orig_model(opt, device='/cpu:0'):
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
    inp_depth = opt['inp_depth']
    padding = opt['padding']
    full_height = inp_height + 2 * padding
    full_width = inp_width + 2 * padding

    rnn_type = opt['rnn_type']
    cnn_filter_size = opt['cnn_filter_size']
    cnn_depth = opt['cnn_depth']
    dcnn_filter_size = opt['dcnn_filter_size']
    dcnn_depth = opt['dcnn_depth']
    conv_lstm_filter_size = opt['conv_lstm_filter_size']
    conv_lstm_hid_depth = opt['conv_lstm_hid_depth']
    rnn_hid_dim = opt['rnn_hid_dim']
    mlp_depth = opt['mlp_depth']
    wd = opt['weight_decay']
    segm_dense_conn = opt['segm_dense_conn']
    use_bn = opt['use_bn']
    use_deconv = opt['use_deconv']
    add_skip_conn = opt['add_skip_conn']
    score_use_core = opt['score_use_core']
    loss_mix_ratio = opt['loss_mix_ratio']
    base_learn_rate = opt['base_learn_rate']
    learn_rate_decay = opt['learn_rate_decay']
    steps_per_decay = opt['steps_per_decay']
    num_mlp_layers = opt['num_mlp_layers']
    mlp_dropout_ratio = opt['mlp_dropout']

    with tf.device(_get_device_fn(device)):
        # Input image, [B, H, W, D]
        x = tf.placeholder('float', [None, full_height, full_width, inp_depth])

        # Whether in training stage, required for batch norm.
        phase_train = tf.placeholder('bool')
        global_step = tf.placeholder('int32')
        
        # Groundtruth segmentation maps, [B, T, H, W]
        y_gt = tf.placeholder(
            'float', [None, timespan, full_height, full_width])
        
        # Groundtruth confidence score, [B, T]
        s_gt = tf.placeholder('float', [None, timespan])
        y_gt_list = tf.split(1, timespan, y_gt)
        
        model['x'] = x
        model['phase_train'] = phase_train
        model['global_step'] = global_step
        model['y_gt'] = y_gt
        model['s_gt'] = s_gt
        
        x_shape = tf.shape(x)
        num_ex = x_shape[0]

        # Random image transformation
        x, y_gt = _add_rnd_img_transformation(
            model, x, y_gt, padding, phase_train)
        model['x_trans'] = x
        model['y_gt_trans'] = y_gt

        # CNN
        cnn_filters = cnn_filter_size
        cnn_channels = [inp_depth] + cnn_depth
        cnn_pool = [2] * len(cnn_filters)
        cnn_act = [tf.nn.relu] * len(cnn_filters)
        cnn_use_bn = [use_bn] * len(cnn_filters)
        h_cnn = nn.cnn(model, x=x, f=cnn_filters, ch=cnn_channels,
                       pool=cnn_pool, act=cnn_act, use_bn=cnn_use_bn,
                       phase_train=phase_train, wd=wd)
        h_pool3 = h_cnn[-1]

        # RNN input size
        subsample = np.array(cnn_pool).prod()
        rnn_h = inp_height / subsample
        rnn_w = inp_width / subsample

        # Low-res segmentation depth
        core_depth = mlp_depth if segm_dense_conn else 1
        core_dim = rnn_h * rnn_w * core_depth

        # RNN
        if rnn_type == 'conv_lstm':
            rnn_depth = conv_lstm_hid_depth
            rnn_dim = rnn_h * rnn_w * rnn_depth
            conv_lstm_inp_depth = cnn_channels[-1]
            rnn_inp = h_pool3
            unroll_rnn = nn.conv_lstm(
                model=model, timespan=timespan,
                inp_height=rnn_h, inp_width=rnn_w,
                inp_depth=conv_lstm_inp_depth,
                filter_size=conv_lstm_filter_size, hid_depth=rnn_depth,
                c_init=tf.zeros(tf.pack([num_ex, rnn_h, rnn_w, rnn_depth])),
                h_init=tf.zeros(tf.pack([num_ex, rnn_h, rnn_w, rnn_depth])),
                wd=wd, name='rnn')
        elif rnn_type == 'lstm' or rnn_type == 'gru':
            rnn_dim = rnn_hid_dim
            rnn_inp_dim = rnn_h * rnn_w * cnn_channels[-1]
            rnn_inp = tf.reshape(
                h_pool3, [-1, rnn_h * rnn_w * cnn_channels[-1]])
            if rnn_type == 'lstm':
                unroll_rnn = nn.lstm(
                    model=model, timespan=timespan,
                    inp_dim=rnn_inp_dim, hid_dim=rnn_hid_dim,
                    c_init=tf.zeros(tf.pack([num_ex, rnn_hid_dim])),
                    h_init=tf.zeros(tf.pack([num_ex, rnn_hid_dim])),
                    wd=wd, name='rnn')
            else:
                unroll_rnn = nn.gru(
                    model=model, timespan=timespan, inp_dim=rnn_inp_dim,
                    hid_dim=rnn_hid_dim,
                    h_init=tf.zeros(tf.pack([num_ex, rnn_hid_dim])),
                    wd=wd, name='rnn')
        else:
            raise Exception('Unknown RNN type: {}'.format(rnn_type))

        for t in xrange(timespan):
            unroll_rnn(rnn_inp, time=t)
        h_rnn = model['h_rnn']
        h_rnn_all = tf.concat(
            1, [tf.expand_dims(h_rnn[tt], 1) for tt in xrange(timespan)])

        # Core segmentation network.
        if segm_dense_conn:
            # Dense segmentation network
            h_rnn_all = tf.reshape(h_rnn_all, [-1, rnn_dim])     
            mlp_dims = [rnn_dim] + [core_dim] * num_mlp_layers
            mlp_act = [tf.nn.relu] * num_mlp_layers
            mlp_dropout = [1.0 - mlp_dropout_ratio] * num_mlp_layers
            mlp = nn.mlp(model,
                         x=h_rnn_all,
                         dims=mlp_dims,
                         act=mlp_act,
                         dropout_keep=mlp_dropout,
                         phase_train=phase_train,
                         wd=wd)
            h_core = tf.reshape(
                mlp[-1], [-1, rnn_h, rnn_w, mlp_depth])
        else:
            # Convolutional segmentation netowrk
            w_segm_conv = nn.weight_variable([3, 3, rnn_depth, 1], wd=wd)
            b_segm_conv = nn.weight_variable([1], wd=wd)
            b_log_softmax = nn.weight_variable([1])
            h_rnn_all = tf.reshape(
                h_rnn_all, [-1, rnn_h, rnn_w, rnn_depth])
            h_core = tf.reshape(tf.log(tf.nn.softmax(tf.reshape(
                nn.conv2d(h_rnn_all, w_segm_conv) + b_segm_conv,
                [-1, rnn_h * rnn_w]))) + b_log_softmax,
                [-1, rnn_h, rnn_w, 1])
        model['h_core'] = h_core

        # Deconv net to upsample
        if use_deconv:
            dcnn_filters = dcnn_filter_size
            dcnn_unpool = [2] * (len(dcnn_filters) - 1) + [1]
            dcnn_act = [tf.nn.relu] * (len(dcnn_filters) - 1) + [tf.sigmoid]
            if segm_dense_conn:
                dcnn_channels = [mlp_depth] + dcnn_depth
            else:
                dcnn_channels = [1] * (len(dcnn_filters) + 1)
            dcnn_use_bn = [use_bn] * len(dcnn_filters)

            if add_skip_conn:
                skip = [None]
                skip_ch = [0]
                for jj, layer in enumerate(h_cnn[-2::-1] + [x]):
                    layer_shape = tf.shape(layer)
                    zeros = tf.zeros(tf.concat(
                        0, [layer_shape[0: 1], tf.constant([timespan]),
                            layer_shape[1:]]))
                    new_shape = tf.concat(
                        0, [layer_shape[0: 1] * timespan, layer_shape[1:]])
                    layer_reshape = tf.reshape(tf.expand_dims(layer, 1) +
                                               zeros, new_shape)
                    skip.append(layer_reshape)
                    ch_idx = len(cnn_channels) - jj - 2
                    skip_ch.append(cnn_channels[ch_idx])
            else:
                skip = None
                skip_ch = None

            h_dcnn = nn.dcnn(model, x=h_core, f=dcnn_filters,
                             ch=dcnn_channels,
                             pool=dcnn_unpool, act=dcnn_act,
                             use_bn=dcnn_use_bn,
                             inp_h=rnn_h, inp_w=rnn_w,
                             skip=skip, skip_ch=skip_ch,
                             phase_train=phase_train, wd=wd)
            y_out = tf.reshape(
                h_dcnn[-1], [-1, timespan, inp_height, inp_width])
        else:
            y_out = tf.reshape(
                tf.image.resize_bilinear(h_core, [inp_height, inp_width]),
                [-1, timespan, inp_height, inp_width])

        model['y_out'] = y_out

        # Scoring network
        if score_use_core:
            # Use core network to predict score
            score_inp = h_core
            score_inp_shape = [-1, core_dim]
            score_inp = tf.reshape(score_inp, score_inp_shape)
            score_dim = core_dim
        else:
            # Use RNN hidden state to predict score
            score_inp = h_rnn_all
            if rnn_type == 'conv_lstm':
                score_inp_shape = [-1, rnn_h, rnn_w, rnn_depth]
                score_inp = tf.reshape(score_inp, score_inp_shape)
                score_maxpool = opt['score_maxpool']
                score_dim = rnn_h * rnn_w / (score_maxpool ** 2) * rnn_depth
                if score_maxpool > 1:
                    score_inp = _max_pool(score_inp, score_maxpool)
                score_inp = tf.reshape(score_inp, [-1, score_dim])
            else:
                score_inp_shape = [-1, rnn_dim]
                score_inp = tf.reshape(score_inp, score_inp_shape)
                score_dim = rnn_dim

        s_out_mlp = nn.mlp(model,
                           x=score_inp,
                           dims=[score_dim, 1],
                           act=[tf.sigmoid],
                           wd=wd)
        s_out = tf.reshape(s_out_mlp[-1], [-1, timespan])
        model['s_out'] = s_out

        # Loss function
        learn_rate = tf.train.exponential_decay(
            base_learn_rate, global_step, steps_per_decay,
            learn_rate_decay, staircase=True)
        model['learn_rate'] = learn_rate
        eps = 1e-7
        loss = _add_ins_segm_loss(
            model, y_out, y_gt, s_out, s_gt, loss_mix_ratio, timespan,
            use_cum_min=True,
            segm_loss_fn=opt['segm_loss_fn'])
        tf.add_to_collection('losses', loss)
        total_loss = tf.add_n(tf.get_collection(
            'losses'), name='total_loss')
        model['total_loss'] = total_loss

        train_step = GradientClipOptimizer(
            tf.train.AdamOptimizer(learn_rate, epsilon=eps),
            clip=1.0).minimize(total_loss)
        model['train_step'] = train_step

    return model


def _get_attn_filter(mu, lg_var, size):
    # [B] => [B, 1, 1]
    mu = tf.reshape(mu, [-1, 1, 1])
    lg_var = tf.reshape(lg_var, [-1, 1, 1])

    # [1, L, 1]
    span = np.reshape(np.arange(size), [1, size, 1])

    # [1, L, 1] - [B, 1, F] = [B, L, F]
    filter = tf.mul(
        1 / tf.sqrt(tf.exp(lg_var)) / tf.sqrt(2 * np.pi),
        tf.exp(-0.5 * (span - mu) * (span - mu) /
               tf.exp(lg_var)))

    return filter


def get_attn_gt_model(opt, device='/cpu:0'):
    """The original model"""
    model = {}

    # timespan = opt['timespan']
    inp_height = opt['inp_height']
    inp_width = opt['inp_width']
    inp_depth = opt['inp_depth']
    # rnn_type = opt['rnn_type']
    # cnn_filter_size = opt['cnn_filter_size']
    # cnn_depth = opt['cnn_depth']
    # dcnn_filter_size = opt['dcnn_filter_size']
    # dcnn_depth = opt['dcnn_depth']
    # conv_lstm_filter_size = opt['conv_lstm_filter_size']
    # conv_lstm_hid_depth = opt['conv_lstm_hid_depth']
    # rnn_hid_dim = opt['rnn_hid_dim']
    # mlp_depth = opt['mlp_depth']
    # wd = opt['weight_decay']
    # segm_dense_conn = opt['segm_dense_conn']
    # use_bn = opt['use_bn']
    # use_deconv = opt['use_deconv']
    # add_skip_conn = opt['add_skip_conn']
    # score_use_core = opt['score_use_core']
    # loss_mix_ratio = opt['loss_mix_ratio']
    # base_learn_rate = opt['base_learn_rate']
    # learn_rate_decay = opt['learn_rate_decay']
    # steps_per_decay = opt['steps_per_decay']

    with tf.device(_get_device_fn(device)):
        # Input image, [B, H, W, D]
        x = tf.placeholder('float', [None, inp_height, inp_width, inp_depth])
        # Attention center, [B, 2]
        attn_mu = tf.placeholder('float', [None, 2])
        # Attention width, [B, 2]
        attn_lg_var = tf.placeholder('float', [None, 2])
        filters_x = _get_attn_filter(attn_mu[:, 0], attn_lg_var[:, 0], )

    pass
