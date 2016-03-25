import cslab_environ

from tensorflow.python.framework import ops
import numpy as np
import tensorflow as tf

from utils import logger
from utils.grad_clip_optim import GradientClipOptimizer
import fg_segm_reader
import image_ops as img
import nnlib as nn
import h5py

from rec_ins_segm_base import *

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


# Use region of interest instread of Gaussian filters.
# def _roi(x, top_left, bot_right, max_height, max_width):
# top_left = tf.minimum(max_height, tf.maximum(0,
# tf.cast(tf.round(top_left), 'int'))


def get_orig_model(opt, device='/cpu:0'):
    """CNN -> -> RNN -> DCNN -> Instances"""
    model = {}
    timespan = opt['timespan']
    inp_height = opt['inp_height']
    inp_width = opt['inp_width']
    inp_depth = opt['inp_depth']
    padding = opt['padding']

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
    steps_per_learn_rate_decay = opt['steps_per_learn_rate_decay']
    num_mlp_layers = opt['num_mlp_layers']
    mlp_dropout_ratio = opt['mlp_dropout']
    segm_loss_fn = opt['segm_loss_fn']

    rnd_hflip = opt['rnd_hflip']
    rnd_vflip = opt['rnd_vflip']
    rnd_transpose = opt['rnd_transpose']
    rnd_colour = opt['rnd_colour']

    with tf.device(_get_device_fn(device)):
        # Input image, [B, H, W, D]
        x = tf.placeholder('float', [None, inp_height, inp_width, inp_depth])

        # Whether in training stage, required for batch norm.
        phase_train = tf.placeholder('bool')

        # Groundtruth segmentation maps, [B, T, H, W]
        y_gt = tf.placeholder(
            'float', [None, timespan, inp_height, inp_width])

        # Groundtruth confidence score, [B, T]
        s_gt = tf.placeholder('float', [None, timespan])

        model['x'] = x
        model['phase_train'] = phase_train
        model['y_gt'] = y_gt
        model['s_gt'] = s_gt

        x_shape = tf.shape(x)
        num_ex = x_shape[0]

        # Random image transformation
        x, y_gt = image.random_transformation(
            x, y_gt, padding, phase_train,
            rnd_hflip=rnd_hflip, rnd_vflip=rnd_vflip,
            rnd_transpose=rnd_transpose, rnd_colour=rnd_colour)
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
                     phase_train=phase_train, wd=wd, model=model)
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
            rnn_state[tt], _gi, _gf, _go = rnn_cell(rnn_inp, rnn_state[tt - 1])

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
                           phase_train=phase_train, wd=wd, model=model)
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
            base_learn_rate, global_step, steps_per_learn_rate_decay,
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
            segm_loss = _match_bce(y_out, y_gt, match, timespan)
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
        tf.exp(-0.5 * (span - mu) * (span - mu) / tf.exp(lg_var)))

    return filter


def _extract_patch(x, f_y, f_x, nchannels):
    """
    Args:
        x: [B, H, W, D]
        f_y: [B, H, FH]
        f_x: [B, W, FH]
        nchannels: D

    Returns:
        patch: [B, FH, FW]
    """
    patch = [None] * nchannels
    fsize_h = tf.shape(f_x)[2]
    fsize_w = tf.shape(f_y)[2]
    hh = tf.shape(x)[1]
    ww = tf.shape(x)[2]

    for dd in xrange(nchannels):
        x_ch = tf.reshape(
            tf.slice(x, [0, 0, 0, dd], [-1, -1, -1, 1]),
            tf.pack([-1, hh, ww]))
        patch[dd] = tf.reshape(tf.batch_matmul(
            tf.batch_matmul(f_y, x_ch, adj_x=True),
            f_x), tf.pack([-1, fsize_h, fsize_w, 1]))

    return tf.concat(3, patch)


def _get_gt_attn(y_gt, attn_size, padding_ratio=0.0, center_shift_ratio=0.0):
    """Get groundtruth attention box given segmentation."""
    s = tf.shape(y_gt)
    # [B, T, H, W, 2]
    idx = _get_idx_map(s)
    idx_min = idx + tf.expand_dims((1.0 - y_gt) * tf.to_float(s[2] * s[3]), 4)
    idx_max = idx * tf.expand_dims(y_gt, 4)
    # [B, T, 2]
    top_left = tf.reduce_min(idx_min, reduction_indices=[2, 3])
    bot_right = tf.reduce_max(idx_max, reduction_indices=[2, 3])

    # Enlarge the groundtruth box.
    if padding_ratio > 0:
        # log.info('Pad groundtruth box by {:.2f}'.format(padding_ratio))
        size = bot_right - top_left
        top_left += center_shift_ratio * size
        top_left -= padding_ratio * size
        bot_right += center_shift_ratio * size
        bot_right += padding_ratio * size
        box = _get_filled_box_idx(idx, top_left, bot_right)
    else:
        log.warning('Not padding groundtruth box')

    ctr = (bot_right + top_left) / 2.0
    delta = (bot_right - top_left + 1.0) / attn_size
    lg_var = tf.zeros(tf.shape(ctr)) + 1.0

    return ctr, delta, lg_var, box, top_left, bot_right, idx


def _get_idx_map(shape):
    """Get index map for a image.

    Args:
        shape: [B, T, H, W]
    Returns:
        idx: [B, T, H, W, 2]
    """
    s = shape
    # [B, T, H, W, 1]
    idx_y = tf.zeros(tf.pack([s[0], s[1], s[2], s[3], 1]), dtype='float')
    idx_x = tf.zeros(tf.pack([s[0], s[1], s[2], s[3], 1]), dtype='float')
    idx_y += tf.reshape(tf.to_float(tf.range(s[2])), [1, 1, -1, 1, 1])
    idx_x += tf.reshape(tf.to_float(tf.range(s[3])), [1, 1, 1, -1, 1])
    idx = tf.concat(4, [idx_y, idx_x])

    return idx


def _get_filled_box_idx(idx, top_left, bot_right):
    """Fill a box with top left and bottom right coordinates."""
    # [B, T, H, W]
    idx_y = idx[:, :, :, :, 0]
    idx_x = idx[:, :, :, :, 1]
    top_left_y = tf.expand_dims(tf.expand_dims(top_left[:, :, 0], 2), 3)
    top_left_x = tf.expand_dims(tf.expand_dims(top_left[:, :, 1], 2), 3)
    bot_right_y = tf.expand_dims(tf.expand_dims(bot_right[:, :, 0], 2), 3)
    bot_right_x = tf.expand_dims(tf.expand_dims(bot_right[:, :, 1], 2), 3)
    lower = tf.logical_and(idx_y >= top_left_y, idx_x >= top_left_x)
    upper = tf.logical_and(idx_y <= bot_right_y, idx_x <= bot_right_x)
    box = tf.to_float(tf.logical_and(lower, upper))

    return box


def _unnormalize_attn(ctr, lg_delta, inp_height, inp_width, attn_size):
    ctr_y = ctr[:, 0] + 1.0
    ctr_x = ctr[:, 1] + 1.0
    ctr_y *= (inp_height + 1) / 2.0
    ctr_x *= (inp_width + 1) / 2.0
    ctr = tf.concat(1, [tf.expand_dims(ctr_y, 1), tf.expand_dims(ctr_x, 1)])
    delta = tf.exp(lg_delta)
    delta_y = delta[:, 0]
    delta_x = delta[:, 1]
    delta_y = (inp_height - 1.0) / (attn_size - 1.0) * delta_y
    delta_x = (inp_width - 1.0) / (attn_size - 1.0) * delta_x
    delta = tf.concat(1, [tf.expand_dims(delta_y, 1),
                          tf.expand_dims(delta_x, 1)])

    return ctr, delta


def _get_attn_coord(ctr, delta, attn_size):
    """Get attention coordinates given parameters."""
    a = ctr * 2.0
    b = delta * attn_size - 1.0
    top_left = (a - b) / 2.0
    bot_right = (a + b) / 2.0 + 1.0

    return top_left, bot_right

