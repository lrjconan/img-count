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
        return get_attn_gt_model(opt, device=device)
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


def _conf_loss(s_out, match, timespan, use_cum_min=True):
    """Loss function for confidence score sequence.

    Args:
        s_out:
        match:
        use_cum_min:
    """
    s_out_shape = tf.shape(s_out)
    num_ex = tf.to_float(s_out_shape[0])
    max_num_obj = tf.to_float(s_out_shape[1])
    match_sum = tf.reduce_sum(match, reduction_indices=[2])

    # Loss for confidence scores.
    if use_cum_min:
        # [B, N]
        s_out_min = _cum_min(s_out, timespan)
        # [B, N]
        s_bce = _bce(s_out_min, match_sum)
    else:
        s_bce = _bce(s_out, match_sum)
    loss = tf.reduce_sum(s_bce) / num_ex / max_num_obj

    return loss


def _segm_match(iou, s_gt):
    """Matching between segmentation output and groundtruth.

    Args:
        y_out: [B, T, H, W], output segmentations
        y_gt: [B, T, H, W], groundtruth segmentations
        s_gt: [B, T], groudtruth score sequence
    """
    # IOU score, [B, N, M]
    # iou_soft = _f_iou(y_out, y_gt, timespan, pairwise=True)
    # Mask X, [B, M] => [B, 1, M]
    mask_x = tf.expand_dims(s_gt, dim=1)
    # Mask Y, [B, M] => [B, N, 1]
    mask_y = tf.expand_dims(s_gt, dim=2)
    iou_mask = iou * mask_x * mask_y

    # Keep certain precision so that we can get optimal matching within
    # reasonable time.
    eps = 1e-5
    precision = 1e6
    iou_mask = tf.round(iou_mask * precision) / precision
    match_eps = tf.user_ops.hungarian(iou_mask + eps)[0]

    # [1, N, 1, 1]
    s_gt_shape = tf.shape(s_gt)
    num_segm_out = s_gt_shape[1]
    num_segm_out_mul = tf.pack([1, num_segm_out, 1])
    # Mask the graph algorithm output.
    match = match_eps * mask_x * mask_y

    return match


def _bce(y_out, y_gt):
    """Binary cross entropy."""
    eps = 1e-5
    return -y_gt * tf.log(y_out + eps) - \
        (1 - y_gt) * tf.log(1 - y_out + eps)


def _match_bce(model, y_out, y_gt, match, timespan):
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
    bce_mat = tf.concat(1, bce_list)
    model['bce_mat'] = bce_mat
    bce_total = tf.reduce_sum(tf.concat(1, bce_list), reduction_indices=[1])
    model['bce_total'] = bce_total

    return tf.reduce_sum(bce_total / match_count) / num_ex / height / width


def _count_acc(s_out, s_gt):
    """Counting accuracy.

    Args:
        s_out:
        s_gt:
    """
    num_ex = tf.to_float(tf.shape(s_out)[0])
    count_out = tf.reduce_sum(tf.to_float(s_out > 0.5), reduction_indices=[1])
    count_gt = tf.reduce_sum(s_gt, reduction_indices=[1])
    count_acc = tf.reduce_sum(tf.to_float(
        tf.equal(count_out, count_gt))) / num_ex

    return count_acc


def _rnd_img_transformation(x, y_gt, padding, phase_train):
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
    # x_rand = img.random_hue(x_rand, 0.5)
    # x_rand = img.random_saturation(x_rand, 0.5, 2.0)
    # x_rand = tf.image.random_brightness(x_rand, 0.5)
    # x_rand = tf.image.random_contrast(x_rand, 0.5, 2.0)

    x = (1.0 - phase_train_f) * x_ctr + phase_train_f * x_rand
    y_gt = (1.0 - phase_train_f) * y_ctr + phase_train_f * y_rand

    return x, y_gt


def _build_skip_conn(cnn_channels, h_cnn, x, timespan):
    """Build skip connection."""
    skip = [None]
    skip_ch = [0]
    for jj, layer in enumerate(h_cnn[-2::-1] + [x]):
        ss = tf.shape(layer)
        zeros = tf.zeros(tf.pack([ss[0], timespan, ss[1], ss[2], ss[3]]))
        new_shape = tf.pack([ss[0] * timespan, ss[1], ss[2], ss[3]])
        layer_reshape = tf.reshape(tf.expand_dims(layer, 1) + zeros, new_shape)
        skip.append(layer_reshape)
        ch_idx = len(cnn_channels) - jj - 2
        skip_ch.append(cnn_channels[ch_idx])

    return skip, skip_ch


def get_orig_model(opt, device='/cpu:0'):
    """CNN -> -> RNN -> DCNN -> Instances"""
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
    segm_loss_fn = opt['segm_loss_fn']

    with tf.device(_get_device_fn(device)):
        # Input image, [B, H, W, D]
        x = tf.placeholder('float', [None, full_height, full_width, inp_depth])

        # Whether in training stage, required for batch norm.
        phase_train = tf.placeholder('bool')

        # Groundtruth segmentation maps, [B, T, H, W]
        y_gt = tf.placeholder(
            'float', [None, timespan, full_height, full_width])

        # Groundtruth confidence score, [B, T]
        s_gt = tf.placeholder('float', [None, timespan])

        model['x'] = x
        model['phase_train'] = phase_train
        model['y_gt'] = y_gt
        model['s_gt'] = s_gt

        x_shape = tf.shape(x)
        num_ex = x_shape[0]

        # Random image transformation
        x, y_gt = _rnd_img_transformation(x, y_gt, padding, phase_train)
        model['x_trans'] = x
        model['y_gt_trans'] = y_gt

        # CNN
        cnn_filters = cnn_filter_size
        cnn_nlayers = len(cnn_filters)
        cnn_channels = [inp_depth] + cnn_depth
        cnn_pool = [2] * cnn_nlayers
        cnn_act = [tf.nn.relu] * cnn_nlayers
        cnn_use_bn = [use_bn] * cnn_nlayers
        cnn = nn.cnn(cnn_filters, cnn_channels, cnn_pool, cnn_act, cnn_use_bn,
                     phase_train=phase_train, wd=wd)
        h_cnn = cnn(x)
        h_pool3 = h_cnn[-1]

        # RNN input size
        subsample = np.array(cnn_pool).prod()
        rnn_h = inp_height / subsample
        rnn_w = inp_width / subsample

        # Low-res segmentation depth
        core_depth = mlp_depth if segm_dense_conn else 1
        core_dim = rnn_h * rnn_w * core_depth
        rnn_state = [None] * (timespan + 1)

        # RNN
        if rnn_type == 'conv_lstm':
            rnn_depth = conv_lstm_hid_depth
            rnn_dim = rnn_h * rnn_w * rnn_depth
            conv_lstm_inp_depth = cnn_channels[-1]
            rnn_inp = h_pool3
            rnn_state[-1] = tf.zeros(tf.pack([num_ex,
                                              rnn_h, rnn_w, rnn_depth * 2]))
            rnn_cell = nn.conv_lstm(conv_lstm_inp_depth, rnn_depth,
                                    conv_lstm_filter_size, wd=wd)
        elif rnn_type == 'lstm' or rnn_type == 'gru':
            rnn_dim = rnn_hid_dim
            rnn_inp_dim = rnn_h * rnn_w * cnn_channels[-1]
            rnn_inp = tf.reshape(
                h_pool3, [-1, rnn_h * rnn_w * cnn_channels[-1]])
            if rnn_type == 'lstm':
                rnn_state[-1] = tf.zeros(tf.pack([num_ex, rnn_hid_dim * 2]))
                rnn_cell = nn.lstm(rnn_inp_dim, rnn_hid_dim, wd=wd)
            else:
                rnn_state[-1] = tf.zeros(tf.pack([num_ex, rnn_hid_dim]))
                rnn_cell = nn.gru(rnn_inp_dim, rnn_hid_dim, wd=wd)
        else:
            raise Exception('Unknown RNN type: {}'.format(rnn_type))

        for tt in xrange(timespan):
            rnn_state[tt] = rnn_cell(rnn_inp, rnn_state[tt - 1])

        if rnn_type == 'conv_lstm':
            h_rnn = [tf.slice(rnn_state[tt], [0, 0, 0, rnn_depth],
                              [-1, -1, -1, rnn_depth])
                     for tt in xrange(timespan)]
        elif rnn_type == 'lstm':
            h_rnn = [tf.slice(rnn_state[tt], [0, rnn_dim], [-1, rnn_dim])
                     for tt in xrange(timespan)]
        elif rnn_type == 'gru':
            h_rnn = state

        h_rnn_all = tf.concat(
            1, [tf.expand_dims(h_rnn[tt], 1) for tt in xrange(timespan)])

        # Core segmentation network.
        if segm_dense_conn:
            # Dense segmentation network
            h_rnn_all = tf.reshape(h_rnn_all, [-1, rnn_dim])
            mlp_dims = [rnn_dim] + [core_dim] * num_mlp_layers
            mlp_act = [tf.nn.relu] * num_mlp_layers
            mlp_dropout = [1.0 - mlp_dropout_ratio] * num_mlp_layers
            segm_mlp = nn.mlp(mlp_dims, mlp_act, mlp_dropout,
                              phase_train=phase_train, wd=wd)
            h_core = segm_mlp(h_rnn_all)[-1]
            h_core = tf.reshape(h_core, [-1, rnn_h, rnn_w, mlp_depth])
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

        # Deconv net to upsample
        if use_deconv:
            dcnn_filters = dcnn_filter_size
            dcnn_nlayers = len(dcnn_filters)
            dcnn_unpool = [2] * (dcnn_nlayers - 1) + [1]
            dcnn_act = [tf.nn.relu] * (dcnn_nlayers - 1) + [tf.sigmoid]
            if segm_dense_conn:
                dcnn_channels = [mlp_depth] + dcnn_depth
            else:
                dcnn_channels = [1] * (dcnn_nlayers + 1)
            dcnn_use_bn = [use_bn] * dcnn_nlayers

            skip = None
            skip_ch = None
            if add_skip_conn:
                skip, skip_ch = _build_skip_conn(
                    cnn_channels, h_cnn, x, timespan)

            dcnn = nn.dcnn(dcnn_filters, dcnn_channels, dcnn_unpool, dcnn_act,
                           dcnn_use_bn, skip_ch=skip_ch,
                           phase_train=phase_train, wd=wd)
            h_dcnn = dcnn(h_core, skip=skip)
            y_out = tf.reshape(
                h_dcnn[-1], [-1, timespan, inp_height, inp_width])
        else:
            y_out = tf.reshape(
                tf.image.resize_bilinear(h_core, [inp_height, inp_wiidth]),
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

        score_mlp = nn.mlp(dims=[score_dim, 1], act=[tf.sigmoid], wd=wd)
        s_out = score_mlp(score_inp)[-1]
        s_out = tf.reshape(s_out, [-1, timespan])
        model['s_out'] = s_out

        # Loss function
        global_step = tf.Variable(0.0)
        learn_rate = tf.train.exponential_decay(
            base_learn_rate, global_step, steps_per_decay,
            learn_rate_decay, staircase=True)
        model['learn_rate'] = learn_rate
        eps = 1e-7

        y_gt_shape = tf.shape(y_gt)
        num_ex = tf.to_float(y_gt_shape[0])
        max_num_obj = tf.to_float(y_gt_shape[1])

        # Pairwise IOU
        iou_soft = _f_iou(y_out, y_gt, timespan, pairwise=True)

        # Matching
        match = _segm_match(iou_soft, s_gt)
        model['match'] = match
        match_sum = tf.reduce_sum(match, reduction_indices=[2])
        match_count = tf.reduce_sum(match_sum, reduction_indices=[1])
        iou_soft = tf.reduce_sum(tf.reduce_sum(
            iou_soft * match, reduction_indices=[1, 2]) / match_count) / num_ex
        model['iou_soft'] = iou_soft

        if segm_loss_fn == 'iou':
            segm_loss = -iou_soft
        elif segm_loss_fn == 'bce':
            segm_loss = _match_bce(model, y_out, y_gt, match, timespan)
        model['segm_loss'] = segm_loss
        conf_loss = _conf_loss(s_out, match, timespan, use_cum_min=True)
        model['conf_loss'] = conf_loss
        loss = loss_mix_ratio * conf_loss + segm_loss
        model['loss'] = loss

        tf.add_to_collection('losses', loss)
        total_loss = tf.add_n(tf.get_collection(
            'losses'), name='total_loss')
        model['total_loss'] = total_loss

        train_step = GradientClipOptimizer(
            tf.train.AdamOptimizer(learn_rate, epsilon=eps),
            clip=1.0).minimize(total_loss, global_step=global_step)
        model['train_step'] = train_step

        # Statistics
        # [B, M, N] * [B, M, N] => [B] * [B] => [1]
        iou_hard = _f_iou(tf.to_float(y_out > 0.5),
                          y_gt, timespan, pairwise=True)
        iou_hard = tf.reduce_sum(tf.reduce_sum(
            iou_hard * match, reduction_indices=[1, 2]) / match_count) / num_ex
        model['iou_hard'] = iou_hard
        model['count_acc'] = _count_acc(s_out, s_gt)

    return model


def _get_attn_filter(center, delta, lg_var, image_size, filter_size):
    """
    Get Gaussian-based attention filter along one dimension
    (assume decomposability).

    Args:
        center:
        lg_d:
        lg_var:
        image_size: L
        filter_size: F
    """
    # [1, 1, F].
    span_filter = tf.to_float(tf.reshape(
        tf.range(filter_size) + 1, [1, 1, -1]))

    # [B, 1, 1]
    center = tf.reshape(center, [-1, 1, 1])
    delta = tf.reshape(delta, [-1, 1, 1])

    # [B, 1, 1] + [B, 1, 1] * [1, F, 1] = [B, 1, F]
    mu = center + delta * (span_filter - filter_size / 2.0 - 0.5)

    # [B, 1, 1]
    lg_var = tf.reshape(lg_var, [-1, 1, 1])

    # [1, L, 1]
    span = tf.to_float(tf.reshape(tf.range(image_size),
                                  tf.pack([1, image_size, 1])))

    # [1, L, 1] - [B, 1, F] = [B, L, F]
    filter = tf.mul(
        1 / tf.sqrt(tf.exp(lg_var)) / tf.sqrt(2 * np.pi),
        tf.exp(-0.5 * (span - mu) * (span - mu) /
               tf.exp(lg_var)))

    return filter


def _extract_patch(x, f_y, f_x, nchannels):
    """
    Args:
        x: [B, H, W, D]
        f_y: [B, H, F]
        f_x: [B, W, F]
        nchannels: D

    Returns:
        patch: [B, F, F]
    """
    patch = [None] * nchannels
    fsize = tf.shape(f_x)[2]
    hh = tf.shape(x)[1]
    ww = tf.shape(x)[2]

    for dd in xrange(nchannels):
        x_ch = tf.reshape(
            tf.slice(x, [0, 0, 0, dd], [-1, -1, -1, 1]),
            tf.pack([-1, hh, ww]))
        patch[dd] = tf.reshape(tf.batch_matmul(
            tf.batch_matmul(f_y, x_ch, adj_x=True),
            f_x), tf.pack([-1, fsize, fsize, 1]))

    return tf.concat(3, patch)


def _get_gt_attn(y_gt, attn_size):
    s = tf.shape(y_gt)
    # [B, T, H, W, 1]
    idx_y = tf.zeros(tf.pack([s[0], s[1], s[2], s[3], 1]), dtype='float')
    idx_x = tf.zeros(tf.pack([s[0], s[1], s[2], s[3], 1]), dtype='float')
    idx_y += tf.reshape(tf.to_float(tf.range(s[2])), [1, 1, -1, 1, 1])
    idx_x += tf.reshape(tf.to_float(tf.range(s[3])), [1, 1, 1, -1, 1])
    # [B, T, H, W, 2]
    idx = tf.concat(4, [idx_y, idx_x])
    idx_min = idx + tf.expand_dims((1.0- y_gt) * tf.to_float(s[2] * s[3]), 4)
    idx_max = idx * tf.expand_dims(y_gt, 4)
    # [B, T, 2]
    top_left = tf.reduce_min(idx_min, reduction_indices=[2, 3])
    bot_right = tf.reduce_max(idx_max, reduction_indices=[2, 3])
    ctr = (bot_right + top_left) / 2
    delta = (bot_right - top_left) / attn_size
    lg_var = tf.zeros(tf.shape(ctr)) + 1.0

    return ctr, delta, lg_var


def get_attn_gt_model(opt, device='/cpu:0'):
    """The original model"""
    model = {}

    timespan = opt['timespan']
    inp_height = opt['inp_height']
    inp_width = opt['inp_width']
    inp_depth = opt['inp_depth']
    padding = opt['padding']
    full_height = inp_height + 2 * padding
    full_width = inp_width + 2 * padding
    attn_size = opt['attn_size']

    cnn_filter_size = opt['cnn_filter_size']
    cnn_depth = opt['cnn_depth']
    dcnn_filter_size = opt['dcnn_filter_size']
    dcnn_depth = opt['dcnn_depth']
    rnn_hid_dim = opt['rnn_hid_dim']
    num_mlp_layers = opt['num_mlp_layers']
    mlp_dropout_ratio = opt['mlp_dropout']
    mlp_depth = opt['mlp_depth']
    wd = opt['weight_decay']
    use_bn = opt['use_bn']
    segm_loss_fn = opt['segm_loss_fn']
    loss_mix_ratio = opt['loss_mix_ratio']
    base_learn_rate = opt['base_learn_rate']
    learn_rate_decay = opt['learn_rate_decay']
    steps_per_decay = opt['steps_per_decay']

    with tf.device(_get_device_fn(device)):
        # Input layer
        # Input image, [B, H, W, D]
        x = tf.placeholder('float', [None, full_height, full_width, inp_depth])
        y_gt = tf.placeholder(
            'float', [None, timespan, full_height, full_width])
        # Groundtruth confidence score, [B, T]
        s_gt = tf.placeholder('float', [None, timespan])

        # Whether in training stage.
        phase_train = tf.placeholder('bool')

        model['x'] = x
        model['y_gt'] = y_gt
        model['s_gt'] = s_gt
        model['phase_train'] = phase_train

        x_shape = tf.shape(x)
        num_ex = x_shape[0]

        # Random image transformation
        x, y_gt = _rnd_img_transformation(x, y_gt, padding, phase_train)
        model['x_trans'] = x
        model['y_gt_trans'] = y_gt

        # Groundtruth bounding box, [B, T, 2]
        attn_ctr, attn_delta, attn_lg_var = _get_gt_attn(y_gt, attn_size)

        # CNN definition
        cnn_filters = cnn_filter_size
        cnn_nlayers = len(cnn_filters)
        cnn_channels = [inp_depth] + cnn_depth
        cnn_pool = [2] * cnn_nlayers
        cnn_act = [tf.nn.relu] * cnn_nlayers
        cnn_use_bn = [use_bn] * cnn_nlayers
        cnn = nn.cnn(cnn_filters, cnn_channels, cnn_pool, cnn_act, cnn_use_bn,
                     phase_train=phase_train, wd=wd)

        # RNN definition
        subsample = np.array(cnn_pool).prod()
        rnn_h = attn_size / subsample
        rnn_w = attn_size / subsample
        rnn_dim = rnn_hid_dim
        rnn_inp_dim = rnn_h * rnn_w * cnn_channels[-1]
        state = [None] * (timespan + 1)
        state[-1] = tf.zeros(tf.pack([num_ex, rnn_hid_dim * 2]))
        rnn_cell = nn.lstm(rnn_inp_dim, rnn_hid_dim, wd=wd)
        filters_y = [None] * timespan
        filters_x = [None] * timespan

        for tt in xrange(timespan):
            # Include controller input CNN here.
            # Include attention controller RNN here.
            # [B, H, A]
            filters_y[tt] = _get_attn_filter(
                attn_ctr[:, tt, 0], attn_delta[
                    :, tt, 0], attn_lg_var[:, tt, 0],
                inp_height, attn_size)
            # [B, W, A]
            filters_x[tt] = _get_attn_filter(
                attn_ctr[:, tt, 1], attn_delta[:, tt, 1], 
                attn_lg_var[:, tt, 1],
                inp_width, attn_size)

            # Attended patch [B, A, A, D]
            x_patch = _extract_patch(
                x, filters_y[tt], filters_x[tt], inp_depth)

            # CNN [B, A, A, D] => [B, RH, RW, RD]
            h_cnn = cnn(x_patch)
            h_cnn_last = h_cnn[-1]
            core_depth = mlp_depth
            core_dim = rnn_h * rnn_w * core_depth

            # RNN [B, T, R]
            rnn_inp = tf.reshape(
                h_cnn_last, [-1, rnn_h * rnn_w * cnn_channels[-1]])
            state[tt] = rnn_cell(rnn_inp, state[tt - 1])

        # Dense segmentation network [B, T, R] => [B, T, M]
        h_rnn = [tf.slice(state[tt], [0, rnn_dim], [-1, rnn_dim])
                 for tt in xrange(timespan)]
        h_rnn_all = tf.concat(1, [tf.expand_dims(h, 1) for h in h_rnn])
        h_rnn_all = tf.reshape(h_rnn_all, [-1, rnn_dim])
        mlp_dims = [rnn_dim] + [core_dim] * num_mlp_layers
        mlp_act = [tf.nn.relu] * num_mlp_layers
        mlp_dropout = [1.0 - mlp_dropout_ratio] * num_mlp_layers
        segm_mlp = nn.mlp(dims=mlp_dims, act=mlp_act, dropout_keep=mlp_dropout,
                          phase_train=phase_train, wd=wd)
        h_core = segm_mlp(h_rnn_all)[-1]
        h_core = tf.reshape(h_core, [-1, rnn_h, rnn_w, mlp_depth])

        # DCNN [B * T, RH, RW, MD] => [B * T, A, A, 1]
        dcnn_filters = dcnn_filter_size
        dcnn_nlayers = len(dcnn_filters)
        dcnn_unpool = [2] * (dcnn_nlayers - 1) + [1]
        # dcnn_act = [tf.nn.relu] * (len(dcnn_filters) - 1) + [tf.sigmoid]
        dcnn_act = [tf.nn.relu] * dcnn_nlayers
        dcnn_channels = [mlp_depth] + dcnn_depth
        dcnn_use_bn = [use_bn] * dcnn_nlayers

        skip, skip_ch = _build_skip_conn(
            cnn_channels, h_cnn, x_patch, timespan)

        dcnn = nn.dcnn(f=dcnn_filters, ch=dcnn_channels, pool=dcnn_unpool,
                       act=dcnn_act, use_bn=dcnn_use_bn, skip_ch=skip_ch,
                       phase_train=phase_train, wd=wd)
        h_dcnn = dcnn(h_core, skip=skip)

        # Inverse attention [B, T, A, A, 1] => [B, T, H, W, 1]
        # Filters T * [B, L, A] => [B * T, L, A]
        filters_y_all = tf.reshape(
            tf.concat(1, [tf.expand_dims(f, 1) for f in filters_y]),
            [-1, inp_height, attn_size])
        filters_x_all = tf.reshape(
            tf.concat(1, [tf.expand_dims(f, 1) for f in filters_x]),
            [-1, inp_width, attn_size])
        y_out = _extract_patch(h_dcnn[-1],
                               tf.transpose(filters_y_all, [0, 2, 1]),
                               tf.transpose(filters_x_all, [0, 2, 1]), 1)
        y_out = tf.sigmoid(y_out)
        y_out = tf.reshape(y_out, [-1, timespan, inp_height, inp_width])
        model['y_out'] = y_out

        # Loss function
        global_step = tf.Variable(0.0)
        learn_rate = tf.train.exponential_decay(
            base_learn_rate, global_step, steps_per_decay,
            learn_rate_decay, staircase=True)
        model['learn_rate'] = learn_rate
        eps = 1e-7

        y_gt_shape = tf.shape(y_gt)
        num_ex = tf.to_float(y_gt_shape[0])
        max_num_obj = tf.to_float(y_gt_shape[1])

        # Pairwise IOU
        iou_soft = _f_iou(y_out, y_gt, timespan, pairwise=True)

        # Matching
        match = _segm_match(iou_soft, s_gt)
        model['match'] = match
        match_sum = tf.reduce_sum(match, reduction_indices=[2])
        match_count = tf.reduce_sum(match_sum, reduction_indices=[1])
        iou_soft = tf.reduce_sum(tf.reduce_sum(
            iou_soft * match, reduction_indices=[1, 2]) / match_count) / num_ex
        model['iou_soft'] = iou_soft

        if segm_loss_fn == 'iou':
            segm_loss = -iou_soft
        elif segm_loss_fn == 'bce':
            segm_loss = _match_bce(model, y_out, y_gt, match, timespan)
        model['segm_loss'] = segm_loss

        loss = segm_loss
        model['loss'] = loss

        tf.add_to_collection('losses', loss)
        total_loss = tf.add_n(tf.get_collection(
            'losses'), name='total_loss')
        model['total_loss'] = total_loss

        train_step = GradientClipOptimizer(
            tf.train.AdamOptimizer(learn_rate, epsilon=eps),
            clip=1.0).minimize(total_loss, global_step=global_step)
        model['train_step'] = train_step

        # Statistics
        # [B, M, N] * [B, M, N] => [B] * [B] => [1]
        iou_hard = _f_iou(tf.to_float(y_out > 0.5),
                          y_gt, timespan, pairwise=True)
        iou_hard = tf.reduce_sum(tf.reduce_sum(
            iou_hard * match, reduction_indices=[1, 2]) / match_count) / num_ex
        model['iou_hard'] = iou_hard

    return model
