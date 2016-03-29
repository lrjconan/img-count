import cslab_environ

from ris_base import *
from utils import logger
from utils.grad_clip_optim import GradientClipOptimizer
import fg_segm_reader
import h5py
import image_ops as img
import nnlib as nn
import numpy as np
import tensorflow as tf

log = logger.get()


def get_model(opt, device='/cpu:0'):
    """The attention model"""
    model = {}

    timespan = opt['timespan']
    inp_height = opt['inp_height']
    inp_width = opt['inp_width']
    inp_depth = opt['inp_depth']
    padding = opt['padding']
    filter_size = opt['filter_size']

    ctrl_cnn_filter_size = opt['ctrl_cnn_filter_size']
    ctrl_cnn_depth = opt['ctrl_cnn_depth']
    ctrl_cnn_pool = opt['ctrl_cnn_pool']
    ctrl_rnn_hid_dim = opt['ctrl_rnn_hid_dim']

    num_ctrl_mlp_layers = opt['num_ctrl_mlp_layers']
    ctrl_mlp_dim = opt['ctrl_mlp_dim']

    # New parameters for double attention.
    num_ctrl_rnn_iter = opt['num_ctrl_rnn_iter']
    num_glimpse_mlp_layers = opt['num_glimpse_mlp_layers']

    attn_cnn_filter_size = opt['attn_cnn_filter_size']
    attn_cnn_depth = opt['attn_cnn_depth']
    attn_cnn_pool = opt['attn_cnn_pool']
    attn_dcnn_filter_size = opt['attn_dcnn_filter_size']
    attn_dcnn_depth = opt['attn_dcnn_depth']
    attn_dcnn_pool = opt['attn_dcnn_pool']
    attn_rnn_hid_dim = opt['attn_rnn_hid_dim']

    mlp_dropout_ratio = opt['mlp_dropout']

    num_attn_mlp_layers = opt['num_attn_mlp_layers']
    attn_mlp_depth = opt['attn_mlp_depth']
    attn_box_padding_ratio = opt['attn_box_padding_ratio']

    wd = opt['weight_decay']
    use_bn = opt['use_bn']
    use_gt_attn = opt['use_gt_attn']
    segm_loss_fn = opt['segm_loss_fn']
    box_loss_fn = opt['box_loss_fn']
    loss_mix_ratio = opt['loss_mix_ratio']
    base_learn_rate = opt['base_learn_rate']
    learn_rate_decay = opt['learn_rate_decay']
    steps_per_learn_rate_decay = opt['steps_per_learn_rate_decay']
    use_attn_rnn = opt['use_attn_rnn']
    use_knob = opt['use_knob']
    knob_base = opt['knob_base']
    knob_decay = opt['knob_decay']
    steps_per_knob_decay = opt['steps_per_knob_decay']
    use_canvas = opt['use_canvas']
    knob_box_offset = opt['knob_box_offset']
    knob_segm_offset = opt['knob_segm_offset']
    knob_use_timescale = opt['knob_use_timescale']
    gt_selector = opt['gt_selector']
    gt_box_ctr_noise = opt['gt_box_ctr_noise']
    gt_box_pad_noise = opt['gt_box_pad_noise']
    gt_segm_noise = opt['gt_segm_noise']
    downsample_canvas = opt['downsample_canvas']
    pretrain_ccnn = opt['pretrain_ccnn']
    cnn_share_weights = opt['cnn_share_weights']

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
        if use_canvas:
            canvas = tf.zeros(tf.pack([num_ex, inp_height, inp_width, 1]))
            if downsample_canvas:
                ccnn_inp_depth = inp_depth
            else:
                ccnn_inp_depth = inp_depth + 1
            acnn_inp_depth = inp_depth + 1
        else:
            ccnn_inp_depth = inp_depth
            acnn_inp_depth = inp_depth

        # Controller CNN definition
        ccnn_filters = ctrl_cnn_filter_size
        ccnn_nlayers = len(ccnn_filters)
        ccnn_channels = [ccnn_inp_depth] + ctrl_cnn_depth
        ccnn_pool = ctrl_cnn_pool
        ccnn_act = [tf.nn.relu] * ccnn_nlayers
        ccnn_use_bn = [use_bn] * ccnn_nlayers
        if pretrain_ccnn:
            h5f = h5py.File(pretrain_ccnn, 'r')
            ccnn_init_w = [{'w': h5f['cnn_w_{}'.format(ii)][:],
                            'b': h5f['cnn_b_{}'.format(ii)][:]}
                           for ii in xrange(ccnn_nlayers)]
            ccnn_frozen = True
        else:
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
        glimpse_map_dim = crnn_h * crnn_w
        glimpse_feat_dim = ccnn_channels[-1]
        # crnn_inp_dim = crnn_h * crnn_w * ccnn_channels[-1]
        crnn_state = [None] * (timespan + 1)
        crnn_glimpse_map = [None] * timespan
        crnn_g_i = [None] * timespan
        crnn_g_f = [None] * timespan
        crnn_g_o = [None] * timespan
        h_crnn = [None] * timespan
        crnn_cell = nn.lstm(glimpse_feat_dim, crnn_dim,
                            wd=wd, scope='ctrl_lstm')

        # Glimpse MLP definition
        gmlp_dims = [crnn_dim] * num_glimpse_mlp_layers + [glimpse_map_dim]
        gmlp_act = [tf.nn.relu] * \
            (num_glimpse_mlp_layers - 1) + [tf.nn.softmax]
        gmlp_dropout = None
        gmlp = nn.mlp(gmlp_dims, gmlp_act, add_bias=True,
                      dropout_keep=gmlp_dropout,
                      phase_train=phase_train, wd=wd, scope='glimpse_mlp')

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

        # Groundtruth bounding box, [B, T, 2]
        attn_ctr_gt, attn_size_gt, attn_lg_var_gt, attn_box_gt, \
            attn_top_left_gt, attn_bot_right_gt = \
            get_gt_attn(y_gt,
                        padding_ratio=attn_box_padding_ratio,
                        center_shift_ratio=0.0)
        attn_ctr_gt_noise, attn_size_gt_noise, attn_lg_var_gt_noise, \
            attn_box_gt_noise, \
            attn_top_left_gt_noise, attn_bot_right_gt_noise = \
            get_gt_attn(y_gt,
                        padding_ratio=tf.random_uniform(
                            tf.pack([num_ex, timespan, 1]),
                            attn_box_padding_ratio - gt_box_pad_noise,
                            attn_box_padding_ratio + gt_box_pad_noise),
                        center_shift_ratio=tf.random_uniform(
                            tf.pack([num_ex, timespan, 2]),
                            -gt_box_ctr_noise, gt_box_ctr_noise))
        attn_ctr_norm = [None] * timespan
        attn_lg_size = [None] * timespan
        attn_ctr = [None] * timespan
        attn_size = [None] * timespan
        attn_lg_var = [None] * timespan
        attn_lg_gamma = [None] * timespan
        attn_gamma = [None] * timespan
        attn_box_lg_gamma = [None] * timespan
        attn_top_left = [None] * timespan
        attn_bot_right = [None] * timespan

        # Attention CNN definition
        acnn_filters = attn_cnn_filter_size
        acnn_nlayers = len(acnn_filters)
        acnn_channels = [acnn_inp_depth] + attn_cnn_depth
        acnn_pool = attn_cnn_pool
        acnn_act = [tf.nn.relu] * acnn_nlayers
        acnn_use_bn = [use_bn] * acnn_nlayers
        if cnn_share_weights:
            ccnn_shared_weights = []
            for ii in xrange(ccnn_nlayers):
                ccnn_shared_weights.append(
                    {'w': model['ctrl_cnn_w_{}'.format(ii)],
                     'b': model['ctrl_cnn_b_{}'.format(ii)]})
        else:
            ccnn_shared_weights = None
        acnn = nn.cnn(acnn_filters, acnn_channels, acnn_pool, acnn_act,
                      acnn_use_bn, phase_train=phase_train, wd=wd,
                      scope='attn_cnn', model=model, 
                      shared_weights=ccnn_shared_weights)

        x_patch = [None] * timespan
        h_acnn = [None] * timespan
        h_acnn_last = [None] * timespan

        # Attention RNN definition
        acnn_subsample = np.array(acnn_pool).prod()
        arnn_h = filter_size / acnn_subsample
        arnn_w = filter_size / acnn_subsample

        if use_attn_rnn:
            arnn_dim = attn_rnn_hid_dim
            arnn_inp_dim = arnn_h * arnn_w * acnn_channels[-1]
            arnn_state = [None] * (timespan + 1)
            arnn_g_i = [None] * timespan
            arnn_g_f = [None] * timespan
            arnn_g_o = [None] * timespan
            arnn_state[-1] = tf.zeros(tf.pack([num_ex, arnn_dim * 2]))
            arnn_cell = nn.lstm(arnn_inp_dim, arnn_dim,
                                wd=wd, scope='attn_lstm')
            amlp_inp_dim = arnn_dim
        else:
            amlp_inp_dim = arnn_h * arnn_w * acnn_channels[-1]

        # Attention MLP definition
        core_depth = attn_mlp_depth
        core_dim = arnn_h * arnn_w * core_depth
        amlp_dims = [amlp_inp_dim] + [core_dim] * num_attn_mlp_layers
        amlp_act = [tf.nn.relu] * num_attn_mlp_layers
        amlp_dropout = None
        # amlp_dropout = [1.0 - mlp_dropout_ratio] * num_attn_mlp_layers
        amlp = nn.mlp(amlp_dims, amlp_act, dropout_keep=amlp_dropout,
                      phase_train=phase_train, wd=wd, scope='attn_mlp')

        # DCNN [B, RH, RW, MD] => [B, A, A, 1]
        adcnn_filters = attn_dcnn_filter_size
        adcnn_nlayers = len(adcnn_filters)
        adcnn_unpool = attn_dcnn_pool
        adcnn_act = [tf.nn.relu] * adcnn_nlayers
        adcnn_channels = [attn_mlp_depth] + attn_dcnn_depth
        adcnn_use_bn = [use_bn] * adcnn_nlayers
        adcnn_skip_ch = [0] + acnn_channels[::-1][1:]
        adcnn = nn.dcnn(adcnn_filters, adcnn_channels, adcnn_unpool,
                        adcnn_act, use_bn=adcnn_use_bn, skip_ch=adcnn_skip_ch,
                        phase_train=phase_train, wd=wd, model=model,
                        scope='attn_dcnn')
        h_adcnn = [None] * timespan

        # Attention box
        attn_box = [None] * timespan
        iou_soft_box = [None] * timespan
        const_ones = tf.ones(
            tf.pack([num_ex, filter_size, filter_size, 1]))
        attn_box_beta = tf.constant([-5.0])
        attn_box_gamma = [None] * timespan

        # Groundtruth mix.
        grd_match_cum = tf.zeros(tf.pack([num_ex, timespan]))
        # Add a bias on every entry so there is no duplicate match
        # [1, N]
        iou_bias_eps = 1e-7
        iou_bias = tf.expand_dims(tf.to_float(
            tf.reverse(tf.range(timespan), [True])) * iou_bias_eps, 0)

        # Scale mix ratio on different timesteps.
        gt_knob_time_scale = tf.reshape(
            1.0 + tf.log(1.0 + tf.to_float(tf.range(timespan)) * 3.0 *
                         float(knob_use_timescale)), [1, timespan, 1])

        # Mix in groundtruth box.
        global_step_box = tf.maximum(0.0, global_step - knob_box_offset)
        gt_knob_prob_box = tf.train.exponential_decay(
            knob_base, global_step_box, steps_per_knob_decay, knob_decay,
            staircase=False)
        gt_knob_prob_box = tf.minimum(
            1.0, gt_knob_prob_box * gt_knob_time_scale)
        gt_knob_box = tf.to_float(tf.random_uniform(
            tf.pack([num_ex, timespan, 1]), 0, 1.0) <= gt_knob_prob_box)
        model['gt_knob_prob_box'] = gt_knob_prob_box[0, 0, 0]

        # Mix in groundtruth segmentation.
        global_step_segm = tf.maximum(0.0, global_step - knob_segm_offset)
        gt_knob_prob_segm = tf.train.exponential_decay(
            knob_base, global_step_segm, steps_per_knob_decay, knob_decay,
            staircase=False)
        gt_knob_prob_segm = tf.minimum(
            1.0, gt_knob_prob_segm * gt_knob_time_scale)
        gt_knob_segm = tf.to_float(tf.random_uniform(
            tf.pack([num_ex, timespan, 1]), 0, 1.0) <= gt_knob_prob_segm)
        model['gt_knob_prob_segm'] = gt_knob_prob_segm[0, 0, 0]

        # Y out
        y_out = [None] * timespan
        y_out_lg_gamma = [None] * timespan
        y_out_beta = tf.constant([-5.0])

        if not use_canvas:
            h_ccnn = ccnn(x)

        for tt in xrange(timespan):
            # Controller CNN [B, H, W, D] => [B, RH1, RW1, RD1]
            if use_canvas:
                ccnn_inp = tf.concat(3, [x, canvas])
                acnn_inp = ccnn_inp
                h_ccnn[tt] = ccnn(ccnn_inp)
                _h_ccnn = h_ccnn[tt]
            else:
                ccnn_inp = x
                acnn_inp = x
                _h_ccnn = h_ccnn

            h_ccnn_last = _h_ccnn[-1]
            # crnn_inp = tf.reshape(h_ccnn_last, [-1, crnn_inp_dim])
            crnn_inp = tf.reshape(
                h_ccnn_last, [-1, glimpse_map_dim, glimpse_feat_dim])

            crnn_state[tt] = [None] * (num_ctrl_rnn_iter + 1)
            crnn_g_i[tt] = [None] * num_ctrl_rnn_iter
            crnn_g_f[tt] = [None] * num_ctrl_rnn_iter
            crnn_g_o[tt] = [None] * num_ctrl_rnn_iter
            h_crnn[tt] = [None] * num_ctrl_rnn_iter
            crnn_state[tt][-1] = tf.zeros(tf.pack([num_ex, crnn_dim * 2]))
            crnn_glimpse_map[tt] = [None] * num_ctrl_rnn_iter
            crnn_glimpse_map[tt][0] = tf.ones(
                tf.pack([num_ex, glimpse_map_dim, 1])) / glimpse_map_dim
            for tt2 in xrange(num_ctrl_rnn_iter):
                crnn_glimpse = tf.reduce_sum(
                    crnn_inp * crnn_glimpse_map[tt][tt2], [1])
                crnn_state[tt][tt2], crnn_g_i[tt][tt2], crnn_g_f[tt][tt2], \
                    crnn_g_o[tt][tt2] = \
                    crnn_cell(crnn_glimpse, crnn_state[tt][tt2 - 1])
                h_crnn[tt][tt2] = tf.slice(
                    crnn_state[tt][tt2], [0, crnn_dim], [-1, crnn_dim])
                h_gmlp = gmlp(h_crnn[tt][tt2])
                if tt2 < num_ctrl_rnn_iter - 1:
                    crnn_glimpse_map[tt][
                        tt2 + 1] = tf.expand_dims(h_gmlp[-1], 2)

            ctrl_out = cmlp(h_crnn[tt][-1])[-1]
            attn_ctr_norm[tt] = tf.slice(ctrl_out, [0, 0], [-1, 2])
            attn_lg_size[tt] = tf.slice(ctrl_out, [0, 2], [-1, 2])
            attn_ctr[tt], attn_size[tt] = get_unnormalized_attn(
                attn_ctr_norm[tt], attn_lg_size[tt], inp_height, inp_width)
            attn_lg_var[tt] = tf.zeros(tf.pack([num_ex, 2]))
            # attn_lg_var[tt] = tf.slice(ctrl_out, [0, 4], [-1, 2])
            attn_lg_gamma[tt] = tf.slice(ctrl_out, [0, 6], [-1, 1])
            attn_box_lg_gamma[tt] = tf.slice(ctrl_out, [0, 7], [-1, 1])
            y_out_lg_gamma[tt] = tf.slice(ctrl_out, [0, 8], [-1, 1])

            attn_gamma[tt] = tf.reshape(
                tf.exp(attn_lg_gamma[tt]), [-1, 1, 1, 1])
            attn_box_gamma[tt] = tf.reshape(tf.exp(
                attn_box_lg_gamma[tt]), [-1, 1, 1, 1])
            y_out_lg_gamma[tt] = tf.reshape(y_out_lg_gamma[tt], [-1, 1, 1, 1])

            # Initial filters (predicted)
            filter_y = get_gaussian_filter(
                attn_ctr[tt][:, 0], attn_size[tt][:, 0],
                attn_lg_var[tt][:, 0], inp_height, filter_size)
            filter_x = get_gaussian_filter(
                attn_ctr[tt][:, 1], attn_size[tt][:, 1],
                attn_lg_var[tt][:, 1], inp_width, filter_size)
            if tt == 0:
                model['filter_y'] = filter_y
            filter_y_inv = tf.transpose(filter_y, [0, 2, 1])
            filter_x_inv = tf.transpose(filter_x, [0, 2, 1])

            # Attention box
            # attn_box[tt] = extract_patch(const_ones * attn_box_gamma[tt],
            #                              filter_y_inv, filter_x_inv, 1)

            _idx_map = get_idx_map(tf.pack([num_ex, inp_height, inp_width]))
            attn_top_left[tt], attn_bot_right[tt] = get_box_coord(
                attn_ctr[tt], attn_size[tt])
            attn_box[tt] = get_filled_box_idx(
                _idx_map, attn_top_left[tt], attn_bot_right[tt])
            attn_box[tt] = tf.sigmoid(attn_box[tt] + attn_box_beta)
            attn_box[tt] = tf.reshape(attn_box[tt],
                                      [-1, 1, inp_height, inp_width])

            # Here is the knob kick in GT bbox.
            if use_knob:
                # IOU [B, 1, T]
                # [B, 1, H, W] * [B, T, H, W] = [B, T]
                # attn_iou_soft[tt] = f_inter(attn_box[tt], attn_box_gt) / \
                #     f_union(attn_box[tt], attn_box_gt, eps=1e-5)
                _top_left = tf.expand_dims(attn_top_left[tt], 1)
                _bot_right = tf.expand_dims(attn_bot_right[tt], 1)
                iou_soft_box[tt] = f_iou_box(
                    _top_left, _bot_right, attn_top_left_gt, attn_bot_right_gt)
                iou_soft_box[tt] += iou_bias
                grd_match = f_greedy_match(iou_soft_box[tt], grd_match_cum)

                if gt_selector == 'greedy_match':
                    # Add in the cumulative matching to not double count.
                    grd_match_cum += grd_match

                # [B, T, 1]
                grd_match = tf.expand_dims(grd_match, 2)
                attn_ctr_gt_match = tf.reduce_sum(
                    grd_match * attn_ctr_gt_noise, 1)
                attn_size_gt_match = tf.reduce_sum(
                    grd_match * attn_size_gt_noise, 1)

                _gt_knob_box = gt_knob_box
                attn_ctr[tt] = phase_train_f * _gt_knob_box[:, tt, 0: 1] * \
                    attn_ctr_gt_match + \
                    (1 - phase_train_f * _gt_knob_box[:, tt, 0: 1]) * \
                    attn_ctr[tt]
                attn_size[tt] = phase_train_f * _gt_knob_box[:, tt, 0: 1] * \
                    attn_size_gt_match + \
                    (1 - phase_train_f * _gt_knob_box[:, tt, 0: 1]) * \
                    attn_size[tt]

            attn_top_left[tt], attn_bot_right[tt] = get_box_coord(
                attn_ctr[tt], attn_size[tt])

            # [B, H, A]
            filter_y = get_gaussian_filter(
                attn_ctr[tt][:, 0], attn_size[tt][:, 0],
                attn_lg_var[tt][:, 0], inp_height, filter_size)

            # [B, W, A]
            filter_x = get_gaussian_filter(
                attn_ctr[tt][:, 1], attn_size[tt][:, 1],
                attn_lg_var[tt][:, 1], inp_width, filter_size)

            # [B, A, H]
            filter_y_inv = tf.transpose(filter_y, [0, 2, 1])

            # [B, A, W]
            filter_x_inv = tf.transpose(filter_x, [0, 2, 1])

            # Attended patch [B, A, A, D]
            x_patch[tt] = attn_gamma[tt] * extract_patch(
                acnn_inp, filter_y, filter_x, acnn_inp_depth)

            # CNN [B, A, A, D] => [B, RH2, RW2, RD2]
            h_acnn[tt] = acnn(x_patch[tt])
            h_acnn_last[tt] = h_acnn[tt][-1]

            if use_attn_rnn:
                # RNN [B, T, R2]
                arnn_inp = tf.reshape(h_acnn_last[tt], [-1, arnn_inp_dim])
                arnn_state[tt], arnn_g_i[tt], arnn_g_f[tt], arnn_g_o[tt] = \
                    arnn_cell(arnn_inp, arnn_state[tt - 1])

            # Scoring network
            s_out[tt] = smlp(h_crnn[tt][-1])[-1]

            # Dense segmentation network [B, R] => [B, M]
            if use_attn_rnn:
                h_arnn = tf.slice(
                    arnn_state[tt], [0, arnn_dim], [-1, arnn_dim])
                amlp_inp = h_arnn
            else:
                amlp_inp = h_acnn_last[tt]
            amlp_inp = tf.reshape(amlp_inp, [-1, amlp_inp_dim])
            h_core = amlp(amlp_inp)[-1]
            h_core = tf.reshape(h_core, [-1, arnn_h, arnn_w, attn_mlp_depth])

            # DCNN
            skip = [None] + h_acnn[tt][::-1][1:] + [x_patch[tt]]
            h_adcnn[tt] = adcnn(h_core, skip=skip)

            # Output
            y_out[tt] = extract_patch(
                h_adcnn[tt][-1], filter_y_inv, filter_x_inv, 1)
            y_out[tt] = tf.exp(y_out_lg_gamma[tt]) * y_out[tt] + y_out_beta
            y_out[tt] = tf.sigmoid(y_out[tt])
            y_out[tt] = tf.reshape(y_out[tt], [-1, 1, inp_height, inp_width])

            # Here is the knob kick in GT segmentations at this timestep.
            # [B, N, 1, 1]
            if use_canvas:
                if use_knob:
                    _gt_knob_segm = tf.expand_dims(
                        tf.expand_dims(gt_knob_segm[:, tt, 0: 1], 2), 3)
                    # [B, N, 1, 1]
                    grd_match = tf.expand_dims(grd_match, 3)
                    _y_out = tf.expand_dims(tf.reduce_sum(
                        grd_match * y_gt, 1), 3)
                    # Add independent uniform noise to groundtruth.
                    _noise = tf.random_uniform(
                        tf.pack([num_ex, inp_height, inp_width, 1]),
                        0, gt_segm_noise)
                    _y_out = _y_out - _y_out * _noise
                    _y_out = phase_train_f * _gt_knob_segm * _y_out + \
                        (1 - phase_train_f * _gt_knob_segm) * \
                        tf.reshape(y_out[tt], [-1, inp_height, inp_width, 1])
                else:
                    _y_out = tf.reshape(y_out[tt],
                                        [-1, inp_height, inp_width, 1])
                canvas += tf.stop_gradient(_y_out)

        s_out = tf.concat(1, s_out)
        model['s_out'] = s_out
        y_out = tf.concat(1, y_out)
        model['y_out'] = y_out
        attn_box = tf.concat(1, attn_box)
        model['attn_box'] = attn_box
        x_patch = tf.concat(1, [tf.expand_dims(x_patch[tt], 1)
                                for tt in xrange(timespan)])
        model['x_patch'] = x_patch

        # for layer in ['ctrl_cnn', 'attn_cnn', 'attn_dcnn']:
        #     for ii in xrange(len(opt['{}_filter_size'.format(layer)])):
        #         for stat in ['bm', 'bv', 'em', 'ev']:
        #             model['{}_{}_{}'.format(layer, ii, stat)] = tf.add_n(
        #                 [model['{}_{}_{}_{}'.format(layer, ii, stat, tt)]
        #                  for tt in xrange(timespan)]) / timespan

        # Loss function
        learn_rate = tf.train.exponential_decay(
            base_learn_rate, global_step, steps_per_learn_rate_decay,
            learn_rate_decay, staircase=True)
        model['learn_rate'] = learn_rate
        eps = 1e-7

        y_gt_shape = tf.shape(y_gt)
        num_ex = tf.to_float(y_gt_shape[0])
        max_num_obj = tf.to_float(y_gt_shape[1])

        # Loss for attnention box
        if use_knob:
            iou_soft_box = tf.concat(1, [tf.expand_dims(iou_soft_box[tt], 1)
                                         for tt in xrange(timespan)])
        else:
            iou_soft_box = f_iou(attn_box, attn_box_gt,
                                 timespan, pairwise=True)

        model['iou_soft_box'] = iou_soft_box
        model['attn_box_gt'] = attn_box_gt
        match_box = f_segm_match(iou_soft_box, s_gt)
        model['match_box'] = match_box
        match_sum_box = tf.reduce_sum(match_box, reduction_indices=[2])
        match_count_box = tf.reduce_sum(
            match_sum_box, reduction_indices=[1])
        match_count_box = tf.maximum(1.0, match_count_box)
        iou_soft_box_mask = tf.reduce_sum(iou_soft_box * match_box, [1])
        iou_soft_box = tf.reduce_sum(tf.reduce_sum(iou_soft_box_mask, [1])
                                     / match_count_box) / num_ex
        gt_wt_box = f_coverage_weight(attn_box_gt)
        wt_iou_soft_box = tf.reduce_sum(tf.reduce_sum(
            iou_soft_box_mask * gt_wt_box, [1])
            / match_count_box) / num_ex
        if box_loss_fn == 'iou':
            box_loss = -iou_soft_box
        elif box_loss_fn == 'wt_iou':
            box_loss = -wt_iou_soft_box
        elif box_loss_fn == 'wt_cov':
            box_loss = -f_weighted_coverage(iou_soft_box, attn_box_gt)
        elif box_loss_fn == 'bce':
            box_loss = f_match_bce(attn_box, attn_box_gt, match_box, timespan)
        else:
            raise Exception('Unknown box_loss_fn: {}'.format(box_loss_fn))
        model['box_loss'] = box_loss

        box_loss_coeff = tf.constant(1.0)
        tf.add_to_collection('losses', box_loss_coeff * box_loss)

        # Loss for fine segmentation
        iou_soft = f_iou(y_out, y_gt, timespan, pairwise=True)
        match = f_segm_match(iou_soft, s_gt)
        model['match'] = match
        match_sum = tf.reduce_sum(match, reduction_indices=[2])
        match_count = tf.reduce_sum(match_sum, reduction_indices=[1])
        match_count = tf.maximum(1.0, match_count)

        # Weighted coverage (soft)
        wt_cov_soft = f_weighted_coverage(iou_soft, y_gt)
        model['wt_cov_soft'] = wt_cov_soft
        unwt_cov_soft = f_unweighted_coverage(iou_soft, match_count)
        model['unwt_cov_soft'] = unwt_cov_soft

        # IOU (soft)
        iou_soft_mask = tf.reduce_sum(iou_soft * match, [1])
        iou_soft = tf.reduce_sum(tf.reduce_sum(iou_soft_mask, [1]) /
                                 match_count) / num_ex
        model['iou_soft'] = iou_soft
        gt_wt = f_coverage_weight(y_gt)
        wt_iou_soft = tf.reduce_sum(tf.reduce_sum(iou_soft_mask * gt_wt, [1]) /
                                    match_count) / num_ex
        model['wt_iou_soft'] = wt_iou_soft

        if segm_loss_fn == 'iou':
            segm_loss = -iou_soft
        elif segm_loss_fn == 'wt_iou':
            segm_loss = -wt_iou_soft
        elif segm_loss_fn == 'wt_cov':
            segm_loss = -wt_cov_soft
        elif segm_loss_fn == 'bce':
            segm_loss = f_match_bce(y_out, y_gt, match, timespan)
        else:
            raise Exception('Unknown segm_loss_fn: {}'.format(segm_loss_fn))
        model['segm_loss'] = segm_loss
        segm_loss_coeff = tf.constant(1.0)
        tf.add_to_collection('losses', segm_loss_coeff * segm_loss)

        # Score loss
        conf_loss = f_conf_loss(s_out, match, timespan, use_cum_min=True)
        model['conf_loss'] = conf_loss
        tf.add_to_collection('losses', loss_mix_ratio * conf_loss)

        total_loss = tf.add_n(tf.get_collection(
            'losses'), name='total_loss')
        model['loss'] = total_loss

        train_step = GradientClipOptimizer(
            tf.train.AdamOptimizer(learn_rate, epsilon=eps),
            clip=1.0).minimize(total_loss, global_step=global_step)
        model['train_step'] = train_step

        # Statistics
        # [B, M, N] * [B, M, N] => [B] * [B] => [1]
        y_out_hard = tf.to_float(y_out > 0.5)
        iou_hard = f_iou(y_out_hard, y_gt, timespan, pairwise=True)
        wt_cov_hard = f_weighted_coverage(iou_hard, y_gt)
        model['wt_cov_hard'] = wt_cov_hard
        unwt_cov_hard = f_unweighted_coverage(iou_hard, match_count)
        model['unwt_cov_hard'] = unwt_cov_hard
        # [B, T]
        iou_hard_mask = tf.reduce_sum(iou_hard * match, [1])
        iou_hard = tf.reduce_sum(tf.reduce_sum(iou_hard_mask, [1]) /
                                 match_count) / num_ex
        model['iou_hard'] = iou_hard
        wt_iou_hard = tf.reduce_sum(tf.reduce_sum(iou_hard_mask * gt_wt, [1]) /
                                    match_count) / num_ex
        model['wt_iou_hard'] = wt_iou_hard

        dice = f_dice(y_out_hard, y_gt, timespan, pairwise=True)
        dice = tf.reduce_sum(tf.reduce_sum(dice * match, [1, 2]) /
                             match_count) / num_ex
        model['dice'] = dice

        model['count_acc'] = f_count_acc(s_out, s_gt)
        model['dic'] = f_dic(s_out, s_gt, abs=False)
        model['dic_abs'] = f_dic(s_out, s_gt, abs=True)

        # Attention coordinate for debugging [B, T, 2]
        attn_top_left = tf.concat(1, [tf.expand_dims(tmp, 1)
                                      for tmp in attn_top_left])
        attn_bot_right = tf.concat(1, [tf.expand_dims(tmp, 1)
                                       for tmp in attn_bot_right])
        attn_ctr = tf.concat(1, [tf.expand_dims(tmp, 1)
                                 for tmp in attn_ctr])
        attn_size = tf.concat(1, [tf.expand_dims(tmp, 1)
                                  for tmp in attn_size])
        attn_lg_gamma = tf.concat(1, [tf.expand_dims(tmp, 1)
                                      for tmp in attn_lg_gamma])
        attn_box_lg_gamma = tf.concat(1, [tf.expand_dims(tmp, 1)
                                          for tmp in attn_box_lg_gamma])
        y_out_lg_gamma = tf.concat(1, [tf.expand_dims(tmp, 1)
                                       for tmp in y_out_lg_gamma])
        attn_lg_gamma_mean = tf.reduce_sum(attn_lg_gamma) / num_ex / timespan
        attn_box_lg_gamma_mean = tf.reduce_sum(
            attn_box_lg_gamma) / num_ex / timespan
        y_out_lg_gamma_mean = tf.reduce_sum(y_out_lg_gamma) / num_ex / timespan
        model['attn_ctr'] = attn_ctr
        model['attn_size'] = attn_size
        model['attn_top_left'] = attn_top_left
        model['attn_bot_right'] = attn_bot_right
        model['attn_lg_gamma_mean'] = attn_lg_gamma_mean
        model['attn_box_lg_gamma_mean'] = attn_box_lg_gamma_mean
        model['y_out_lg_gamma_mean'] = y_out_lg_gamma_mean

        # # Ctrl RNN gate statistics
        # crnn_g_i = tf.concat(1, [tf.expand_dims(tmp, 1) for tmp in crnn_g_i])
        # crnn_g_f = tf.concat(1, [tf.expand_dims(tmp, 1) for tmp in crnn_g_f])
        # crnn_g_o = tf.concat(1, [tf.expand_dims(tmp, 1) for tmp in crnn_g_o])
        # crnn_g_i_avg = tf.reduce_sum(
        #     crnn_g_i) / tf.to_float(num_ex) / timespan / ctrl_rnn_hid_dim
        # crnn_g_f_avg = tf.reduce_sum(
        #     crnn_g_f) / tf.to_float(num_ex) / timespan / ctrl_rnn_hid_dim
        # crnn_g_o_avg = tf.reduce_sum(
        #     crnn_g_o) / tf.to_float(num_ex) / timespan / ctrl_rnn_hid_dim
        # model['crnn_g_i_avg'] = crnn_g_i_avg
        # model['crnn_g_f_avg'] = crnn_g_f_avg
        # model['crnn_g_o_avg'] = crnn_g_o_avg

    return model
