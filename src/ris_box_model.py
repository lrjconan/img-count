import cslab_environ

from ris_base import *
from utils import logger
from utils.grad_clip_optim import GradientClipOptimizer
import image_ops as img
import nnlib as nn
import numpy as np
import tensorflow as tf

log = logger.get()


def get_model(opt, device='/:cpu:0'):
    """The box model"""
    model = {}

    timespan = opt['timespan']
    inp_height = opt['inp_height']
    inp_width = opt['inp_width']
    inp_depth = opt['inp_depth']
    padding = opt['padding']

    ctrl_cnn_filter_size = opt['ctrl_cnn_filter_size']
    ctrl_cnn_depth = opt['ctrl_cnn_depth']
    ctrl_cnn_pool = opt['ctrl_cnn_pool']
    ctrl_rnn_hid_dim = opt['ctrl_rnn_hid_dim']

    num_ctrl_mlp_layers = opt['num_ctrl_mlp_layers']
    ctrl_mlp_dim = opt['ctrl_mlp_dim']

    attn_box_padding_ratio = opt['attn_box_padding_ratio']

    wd = opt['weight_decay']
    use_bn = opt['use_bn']
    box_loss_fn = opt['box_loss_fn']
    base_learn_rate = opt['base_learn_rate']
    learn_rate_decay = opt['learn_rate_decay']
    steps_per_learn_rate_decay = opt['steps_per_learn_rate_decay']
    gt_selector = opt['gt_selector']

    rnd_hflip = opt['rnd_hflip']
    rnd_vflip = opt['rnd_vflip']
    rnd_transpose = opt['rnd_transpose']
    rnd_colour = opt['rnd_colour']

    with tf.device(get_device_fn(device)):
        # Input definition
        # Input image, [B, H, W, D]
        x = tf.placeholder('float', [None, inp_height, inp_width, inp_depth])
        x_shape = tf.shape(x)
        num_ex = x_shape[0]
        y_gt = tf.placeholder('float', [None, timespan, inp_height, inp_width])
        # Groundtruth confidence score, [B, T]
        s_gt = tf.placeholder('float', [None, timespan])
        # Whether in training stage.
        phase_train = tf.placeholder('bool')
        phase_train_f = tf.to_float(phase_train)
        model['x'] = x
        model['y_gt'] = y_gt
        model['s_gt'] = s_gt
        model['phase_train'] = phase_train

        # Global step
        global_step = tf.Variable(0.0)

        # Random image transformation
        x, y_gt = img.random_transformation(
            x, y_gt, padding, phase_train,
            rnd_hflip=rnd_hflip, rnd_vflip=rnd_vflip,
            rnd_transpose=rnd_transpose, rnd_colour=rnd_colour)
        model['x_trans'] = x
        model['y_gt_trans'] = y_gt

        # Canvas
        canvas = tf.zeros(tf.pack([num_ex, inp_height, inp_width, 1]))
        ccnn_inp_depth = inp_depth + 1
        acnn_inp_depth = inp_depth + 1

        # Controller CNN definition
        ccnn_filters = ctrl_cnn_filter_size
        ccnn_nlayers = len(ccnn_filters)
        ccnn_channels = [ccnn_inp_depth] + ctrl_cnn_depth
        ccnn_pool = ctrl_cnn_pool
        ccnn_act = [tf.nn.relu] * ccnn_nlayers
        ccnn_use_bn = [use_bn] * ccnn_nlayers
        ccnn_init_w = None
        ccnn_frozen = False
        ccnn = nn.cnn(ccnn_filters, ccnn_channels, ccnn_pool, ccnn_act,
                      ccnn_use_bn, phase_train=phase_train, wd=wd,
                      scope='ctrl_cnn', model=model, init_weights=ccnn_init_w,
                      frozen=ccnn_frozen)
        h_ccnn = [None] * timespan

        # Controller RNN definition
        ccnn_subsample = np.array(ccnn_pool).prod()
        crnn_h = inp_height / ccnn_subsample
        crnn_w = inp_width / ccnn_subsample
        crnn_dim = ctrl_rnn_hid_dim
        canvas_dim = inp_height * inp_width / (ccnn_subsample ** 2)
        crnn_inp_dim = crnn_h * crnn_w * ccnn_channels[-1]
        crnn_state = [None] * (timespan + 1)
        crnn_g_i = [None] * timespan
        crnn_g_f = [None] * timespan
        crnn_g_o = [None] * timespan
        h_crnn = [None] * timespan
        crnn_state[-1] = tf.zeros(tf.pack([num_ex, crnn_dim * 2]))
        crnn_cell = nn.lstm(crnn_inp_dim, crnn_dim, wd=wd, scope='ctrl_lstm')

        # Controller MLP definition
        cmlp_dims = [crnn_dim] + [ctrl_mlp_dim] * \
            (num_ctrl_mlp_layers - 1) + [9]
        cmlp_act = [tf.nn.relu] * (num_ctrl_mlp_layers - 1) + [None]
        cmlp_dropout = None
        # cmlp_dropout = [1.0 - mlp_dropout_ratio] * num_ctrl_mlp_layers
        cmlp = nn.mlp(cmlp_dims, cmlp_act, add_bias=True,
                      dropout_keep=cmlp_dropout,
                      phase_train=phase_train, wd=wd, scope='ctrl_mlp')

        # Score MLP definition
        smlp = nn.mlp([crnn_dim, 1], [tf.sigmoid], wd=wd, scope='score_mlp')
        s_out = [None] * timespan

        # Groundtruth box parameters
        # [B, T, 2]
        box_top_left_gt, box_bot_right_gt, box_map_gt = get_gt_box(
            y_gt, padding_ratio=attn_box_padding_ratio, center_shift_ratio=0.0)
        box_ctr_gt, box_size_gt = get_box_ctr_size(
            box_top_left_gt, box_bot_right_gt)
        box_ctr_norm_gt = get_normalized_center(
            box_ctr_gt, inp_height, inp_width)
        box_size_log_gt = get_normalized_size(
            box_size_gt, inp_height, inp_width)
        box_params_gt = tf.concat(2, [box_ctr_norm_gt, box_size_log_gt])

        # Normalized box parameters
        box_ctr_norm = [None] * timespan
        box_size_log = [None] * timespan

        # Unnormalized box parameters
        box_ctr = [None] * timespan
        box_size = [None] * timespan
        box_top_left = [None] * timespan
        box_bot_right = [None] * timespan
        box_map = [None] * timespan
        box_iou_soft = [None] * timespan

        # Groundtruth mix.
        grd_match_cum = tf.zeros(tf.pack([num_ex, timespan]))
        # Add a bias on every entry so there is no duplicate match
        # [1, N]
        iou_bias_eps = 1e-7
        iou_bias = tf.expand_dims(tf.to_float(
            tf.reverse(tf.range(timespan), [True])) * iou_bias_eps, 0)

        for tt in xrange(timespan):
            # Controller CNN [B, H, W, D] => [B, RH1, RW1, RD1]
            ccnn_inp = tf.concat(3, [x, canvas])
            acnn_inp = ccnn_inp
            h_ccnn[tt] = ccnn(ccnn_inp)
            _h_ccnn = h_ccnn[tt]
            h_ccnn_last = _h_ccnn[-1]
            crnn_inp = tf.reshape(h_ccnn_last, [-1, crnn_inp_dim])

            # Controller RNN [B, R1]
            crnn_state[tt], crnn_g_i[tt], crnn_g_f[tt], crnn_g_o[tt] = \
                crnn_cell(crnn_inp, crnn_state[tt - 1])
            h_crnn[tt] = tf.slice(
                crnn_state[tt], [0, crnn_dim], [-1, crnn_dim])

            ctrl_out = cmlp(h_crnn[tt])[-1]
            # [B, T, 2]
            box_ctr_norm[tt] = tf.slice(ctrl_out, [0, 0], [-1, 2])
            box_size_log[tt] = tf.slice(ctrl_out, [0, 2], [-1, 2])
            box_ctr[tt], box_size[tt] = get_unnormalized_attn(
                box_ctr_norm[tt], box_size_log[tt], inp_height, inp_width)
            box_top_left[tt], box_bot_right[tt] = get_box_coord(
                box_ctr[tt], box_size[tt])
            _idx_map = get_idx_map(tf.pack([num_ex, inp_height, inp_width]))

            # [B, H, W]
            box_map[tt] = get_filled_box_idx(
                _idx_map, box_top_left[tt], box_bot_right[tt])
            # [B, 1, H, W]
            box_map[tt] = tf.expand_dims(box_map[tt], 1)

            # [B, 1, H, W] * [B, T, H, W] = [B, T]
            # box_iou_soft[tt] = f_inter(box_map[tt], box_map_gt) / \
            #     f_union(box_map[tt], box_map_gt, eps=1e-5)
            _top_left = tf.expand_dims(box_top_left[tt], 1)
            _bot_right = tf.expand_dims(box_bot_right[tt], 1)
            box_iou_soft[tt] = f_iou_box(
                _top_left, _bot_right, box_top_left_gt, box_bot_right_gt)
            box_iou_soft[tt] += iou_bias
            grd_match = f_greedy_match(box_iou_soft[tt], grd_match_cum)

            if gt_selector == 'greedy_match':
                # Add in the cumulative matching to not double count.
                grd_match_cum += grd_match

            # [B, N, 1, 1]
            grd_match = tf.expand_dims(tf.expand_dims(grd_match, 2), 3)
            _y_out = tf.expand_dims(tf.reduce_sum(
                grd_match * y_gt, 1), 3)
            # Add independent uniform noise to groundtruth.
            _noise = tf.random_uniform(
                tf.pack([num_ex, inp_height, inp_width, 1]), 0, 0.3)
            _y_out = _y_out - _y_out * _noise
            canvas += tf.stop_gradient(_y_out)

            # Scoring network
            s_out[tt] = smlp(h_crnn[tt])[-1]

        # Scores
        s_out = tf.concat(1, s_out)
        model['s_out'] = s_out

        # Box map
        box_map = tf.concat(1, box_map)
        model['box_map'] = box_map

        # Box unnormalized coordinates
        box_top_left = tf.concat(1, [tf.expand_dims(tmp, 1)
                                     for tmp in box_top_left])
        box_bot_right = tf.concat(1, [tf.expand_dims(tmp, 1)
                                      for tmp in box_bot_right])
        box_ctr = tf.concat(1, [tf.expand_dims(tmp, 1) for tmp in box_ctr])
        box_size = tf.concat(1, [tf.expand_dims(tmp, 1) for tmp in box_size])
        model['box_top_left'] = box_top_left
        model['box_bot_right'] = box_bot_right
        model['box_ctr'] = box_ctr
        model['box_size'] = box_size
        model['box_ctr_norm_gt'] = box_ctr_norm_gt
        model['box_size_log_gt'] = box_size_log_gt
        model['box_top_left_gt'] = box_top_left_gt
        model['box_bot_right_gt'] = box_bot_right_gt

        # Box normalized coordinates
        box_ctr_norm = tf.concat(1, [tf.expand_dims(tmp, 1)
                                     for tmp in box_ctr_norm])
        box_size_log = tf.concat(1, [tf.expand_dims(tmp, 1)
                                     for tmp in box_size_log])
        box_params = tf.concat(2, [box_ctr_norm, box_size_log])
        model['box_ctr_norm'] = box_ctr_norm
        model['box_size_log'] = box_size_log

        # Loss function
        learn_rate = tf.train.exponential_decay(
            base_learn_rate, global_step, steps_per_learn_rate_decay,
            learn_rate_decay, staircase=True)
        model['learn_rate'] = learn_rate
        eps = 1e-7

        y_gt_shape = tf.shape(y_gt)
        num_ex = tf.to_float(y_gt_shape[0])
        max_num_obj = tf.to_float(y_gt_shape[1])

        # Loss for box
        iou_soft_box = tf.concat(1, [tf.expand_dims(box_iou_soft[tt], 1)
                                     for tt in xrange(timespan)])

        model['box_map_gt'] = box_map_gt
        match_box = f_segm_match(iou_soft_box, s_gt)
        model['match_box'] = match_box
        match_sum_box = tf.reduce_sum(match_box, reduction_indices=[2])
        match_count_box = tf.reduce_sum(
            match_sum_box, reduction_indices=[1])
        match_count_box = tf.maximum(1.0, match_count_box)
        iou_soft_box_mask = tf.reduce_sum(iou_soft_box * match_box, [1])
        iou_soft_box = tf.reduce_sum(tf.reduce_sum(iou_soft_box_mask, [1])
                                     / match_count_box) / num_ex
        model['iou_soft_box'] = iou_soft_box
        gt_wt_box = f_coverage_weight(box_map_gt)
        wt_iou_soft_box = tf.reduce_sum(tf.reduce_sum(
            iou_soft_box_mask * gt_wt_box, [1])
            / match_count_box) / num_ex
        model['wt_iou_soft_box'] = wt_iou_soft_box
        if box_loss_fn == 'mse':
            box_loss = f_match_loss(
                box_params, box_params_gt, match_box, timespan, f_squared_err,
                model=model)
        elif box_loss_fn == 'huber':
            box_loss = f_match_loss(
                box_params, box_params_gt, match_box, timespan, f_huber)
        elif box_loss_fn == 'iou':
            box_loss = -iou_soft_box
        elif box_loss_fn == 'wt_iou':
            box_loss = -wt_iou_soft_box
        elif box_loss_fn == 'wt_cov':
            box_loss = -f_weighted_coverage(iou_soft_box, box_map_gt)
        elif box_loss_fn == 'bce':
            box_loss = f_match_loss(
                box_map, box_map_gt, match_box, timespan, f_bce)
        else:
            raise Exception('Unknown box_loss_fn: {}'.format(box_loss_fn))
        model['box_loss'] = box_loss

        box_loss_coeff = tf.constant(1.0)
        tf.add_to_collection('losses', box_loss_coeff * box_loss)

        # Score loss
        conf_loss = f_conf_loss(s_out, match_box, timespan, use_cum_min=True)
        model['conf_loss'] = conf_loss
        conf_loss_coeff = tf.constant(1.0)
        tf.add_to_collection('losses', conf_loss_coeff * conf_loss)

        total_loss = tf.add_n(tf.get_collection(
            'losses'), name='total_loss')
        model['loss'] = total_loss

        train_step = GradientClipOptimizer(
            tf.train.AdamOptimizer(learn_rate, epsilon=eps),
            clip=1.0).minimize(total_loss, global_step=global_step)
        model['train_step'] = train_step

        # Ctrl RNN gate statistics
        crnn_g_i = tf.concat(1, [tf.expand_dims(tmp, 1) for tmp in crnn_g_i])
        crnn_g_f = tf.concat(1, [tf.expand_dims(tmp, 1) for tmp in crnn_g_f])
        crnn_g_o = tf.concat(1, [tf.expand_dims(tmp, 1) for tmp in crnn_g_o])
        crnn_g_i_avg = tf.reduce_sum(
            crnn_g_i) / tf.to_float(num_ex) / timespan / ctrl_rnn_hid_dim
        crnn_g_f_avg = tf.reduce_sum(
            crnn_g_f) / tf.to_float(num_ex) / timespan / ctrl_rnn_hid_dim
        crnn_g_o_avg = tf.reduce_sum(
            crnn_g_o) / tf.to_float(num_ex) / timespan / ctrl_rnn_hid_dim
        model['crnn_g_i_avg'] = crnn_g_i_avg
        model['crnn_g_f_avg'] = crnn_g_f_avg
        model['crnn_g_o_avg'] = crnn_g_o_avg

    return model
