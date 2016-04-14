import cslab_environ

from utils import logger
from utils.grad_clip_optim import GradientClipOptimizer
import h5py
import image_ops as img
import nnlib as nn
import numpy as np
import tensorflow as tf

import ris_model_base as base

log = logger.get()


def get_model(opt, device='/cpu:0'):
    """The attention model"""
    model = {}

    timespan = opt['timespan']
    inp_height = opt['inp_height']
    inp_width = opt['inp_width']
    inp_depth = opt['inp_depth']
    padding = opt['padding']
    filter_height = opt['filter_height']
    filter_width = opt['filter_width']

    ctrl_cnn_filter_size = opt['ctrl_cnn_filter_size']
    ctrl_cnn_depth = opt['ctrl_cnn_depth']
    ctrl_cnn_pool = opt['ctrl_cnn_pool']
    ctrl_rnn_hid_dim = opt['ctrl_rnn_hid_dim']

    num_ctrl_mlp_layers = opt['num_ctrl_mlp_layers']
    ctrl_mlp_dim = opt['ctrl_mlp_dim']

    attn_cnn_filter_size = opt['attn_cnn_filter_size']
    attn_cnn_depth = opt['attn_cnn_depth']
    attn_cnn_pool = opt['attn_cnn_pool']
    attn_dcnn_filter_size = opt['attn_dcnn_filter_size']
    attn_dcnn_depth = opt['attn_dcnn_depth']
    attn_dcnn_pool = opt['attn_dcnn_pool']

    mlp_dropout_ratio = opt['mlp_dropout']

    num_attn_mlp_layers = opt['num_attn_mlp_layers']
    attn_mlp_depth = opt['attn_mlp_depth']
    attn_box_padding_ratio = opt['attn_box_padding_ratio']

    wd = opt['weight_decay']
    use_bn = opt['use_bn']
    segm_loss_fn = opt['segm_loss_fn']
    box_loss_fn = opt['box_loss_fn']
    loss_mix_ratio = opt['loss_mix_ratio']
    base_learn_rate = opt['base_learn_rate']
    learn_rate_decay = opt['learn_rate_decay']
    steps_per_learn_rate_decay = opt['steps_per_learn_rate_decay']
    use_knob = opt['use_knob']
    knob_base = opt['knob_base']
    knob_decay = opt['knob_decay']
    steps_per_knob_decay = opt['steps_per_knob_decay']
    knob_box_offset = opt['knob_box_offset']
    knob_segm_offset = opt['knob_segm_offset']
    knob_use_timescale = opt['knob_use_timescale']
    gt_box_ctr_noise = opt['gt_box_ctr_noise']
    gt_box_pad_noise = opt['gt_box_pad_noise']
    gt_segm_noise = opt['gt_segm_noise']

    squash_ctrl_params = opt['squash_ctrl_params']
    fixed_order = opt['fixed_order']
    clip_gradient = opt['clip_gradient']
    fixed_gamma = opt['fixed_gamma']
    ctrl_rnn_inp_struct = opt['ctrl_rnn_inp_struct']  # dense or attn
    num_ctrl_rnn_iter = opt['num_ctrl_rnn_iter']
    num_glimpse_mlp_layers = opt['num_glimpse_mlp_layers']
    pretrain_ctrl_net = opt['pretrain_ctrl_net']
    pretrain_attn_net = opt['pretrain_attn_net']
    pretrain_net = opt['pretrain_net']
    freeze_ctrl_cnn = opt['freeze_ctrl_cnn']
    freeze_ctrl_rnn = opt['freeze_ctrl_rnn']
    freeze_attn_net = opt['freeze_attn_net']

    rnd_hflip = opt['rnd_hflip']
    rnd_vflip = opt['rnd_vflip']
    rnd_transpose = opt['rnd_transpose']
    rnd_colour = opt['rnd_colour']

############################
# Input definition
############################
    with tf.device(base.get_device_fn(device)):
        # Input image, [B, H, W, D]
        x = tf.placeholder('float', [None, inp_height, inp_width, inp_depth],
                           name='x')
        x_shape = tf.shape(x)
        num_ex = x_shape[0]

        # Groundtruth segmentation, [B, T, H, W]
        y_gt = tf.placeholder('float', [None, timespan, inp_height, inp_width],
                              name='y_gt')

        # Groundtruth confidence score, [B, T]
        s_gt = tf.placeholder('float', [None, timespan], name='s_gt')

        # Whether in training stage.
        phase_train = tf.placeholder('bool', name='phase_train')
        phase_train_f = tf.to_float(phase_train)
        
        model['x'] = x
        model['y_gt'] = y_gt
        model['s_gt'] = s_gt
        model['phase_train'] = phase_train

        # Global step
        global_step = tf.Variable(0.0, name='global_step')

###############################
# Random input transformation
###############################
        x, y_gt = img.random_transformation(
            x, y_gt, padding, phase_train,
            rnd_hflip=rnd_hflip, rnd_vflip=rnd_vflip,
            rnd_transpose=rnd_transpose, rnd_colour=rnd_colour)
        model['x_trans'] = x
        model['y_gt_trans'] = y_gt

############################
# Canvas: external memory
############################
        canvas = tf.zeros(tf.pack([num_ex, inp_height, inp_width, 1]))
        ccnn_inp_depth = inp_depth + 1
        acnn_inp_depth = inp_depth + 1

############################
# Controller CNN definition
############################
        ccnn_filters = ctrl_cnn_filter_size
        ccnn_nlayers = len(ccnn_filters)
        acnn_nlayers = len(attn_cnn_filter_size)
        ccnn_channels = [ccnn_inp_depth] + ctrl_cnn_depth
        ccnn_pool = ctrl_cnn_pool
        ccnn_act = [tf.nn.relu] * ccnn_nlayers
        ccnn_use_bn = [use_bn] * ccnn_nlayers

        pt = pretrain_net or pretrain_ctrl_net
        if pt:
            log.info('Loading pretrained controller CNN weights from {}'.format(
                pt))
            h5f = h5py.File(pt, 'r')
            ccnn_init_w = [{'w': h5f['ctrl_cnn_w_{}'.format(ii)][:],
                            'b': h5f['ctrl_cnn_b_{}'.format(ii)][:]}
                           for ii in xrange(ccnn_nlayers)]
            for ii in xrange(ccnn_nlayers):
                for tt in xrange(timespan):
                    for w in ['beta', 'gamma']:
                        ccnn_init_w[ii]['{}_{}'.format(w, tt)] = h5f[
                            'ctrl_cnn_{}_{}_{}'.format(ii, tt, w)][:]
            ccnn_frozen = [freeze_ctrl_cnn] * ccnn_nlayers
        else:
            ccnn_init_w = None
            ccnn_frozen = [freeze_ctrl_cnn] * ccnn_nlayers

        ccnn = nn.cnn(ccnn_filters, ccnn_channels, ccnn_pool, ccnn_act,
                      ccnn_use_bn, phase_train=phase_train, wd=wd,
                      scope='ctrl_cnn', model=model, init_weights=ccnn_init_w,
                      frozen=ccnn_frozen)
        h_ccnn = [None] * timespan

############################
# Controller RNN definition
############################
        ccnn_subsample = np.array(ccnn_pool).prod()
        crnn_h = inp_height / ccnn_subsample
        crnn_w = inp_width / ccnn_subsample
        crnn_dim = ctrl_rnn_hid_dim
        canvas_dim = inp_height * inp_width / (ccnn_subsample ** 2)

        glimpse_map_dim = crnn_h * crnn_w
        glimpse_feat_dim = ccnn_channels[-1]
        if ctrl_rnn_inp_struct == 'dense':
            crnn_inp_dim = crnn_h * crnn_w * ccnn_channels[-1]
        elif ctrl_rnn_inp_struct == 'attn':
            crnn_inp_dim = glimpse_feat_dim

        pt = pretrain_net or pretrain_ctrl_net
        if pt:
            log.info('Loading pretrained controller RNN weights from {}'.format(
                pt))
            h5f = h5py.File(pt, 'r')
            crnn_init_w = {}
            for w in ['w_xi', 'w_hi', 'b_i', 'w_xf', 'w_hf', 'b_f', 'w_xu',
                      'w_hu', 'b_u', 'w_xo', 'w_ho', 'b_o']:
                key = 'ctrl_lstm_{}'.format(w)
                crnn_init_w[w] = h5f[key][:]
            crnn_frozen = freeze_ctrl_rnn
        else:
            crnn_init_w = None
            crnn_frozen = freeze_ctrl_rnn

        crnn_state = [None] * (timespan + 1)
        crnn_glimpse_map = [None] * timespan
        crnn_g_i = [None] * timespan
        crnn_g_f = [None] * timespan
        crnn_g_o = [None] * timespan
        h_crnn = [None] * timespan
        crnn_state[-1] = tf.zeros(tf.pack([num_ex, crnn_dim * 2]))
        crnn_cell = nn.lstm(crnn_inp_dim, crnn_dim, wd=wd, scope='ctrl_lstm',
                            init_weights=crnn_init_w, frozen=crnn_frozen,
                            model=model)

############################
# Glimpse MLP definition
############################
        gmlp_dims = [crnn_dim] * num_glimpse_mlp_layers + [glimpse_map_dim]
        gmlp_act = [tf.nn.relu] * \
            (num_glimpse_mlp_layers - 1) + [tf.nn.softmax]
        gmlp_dropout = None

        pt = pretrain_net or pretrain_ctrl_net
        if pt:
            log.info('Loading pretrained glimpse MLP weights from {}'.format(
                pt))
            h5f = h5py.File(pt, 'r')
            gmlp_init_w = [{'w': h5f['glimpse_mlp_w_{}'.format(ii)][:],
                            'b': h5f['glimpse_mlp_b_{}'.format(ii)][:]}
                           for ii in xrange(num_glimpse_mlp_layers)]
            gmlp_frozen = [freeze_ctrl_rnn] * num_glimpse_mlp_layers
        else:
            gmlp_init_w = None
            gmlp_frozen = [freeze_ctrl_rnn] * num_glimpse_mlp_layers

        gmlp = nn.mlp(gmlp_dims, gmlp_act, add_bias=True,
                      dropout_keep=gmlp_dropout,
                      phase_train=phase_train, wd=wd, scope='glimpse_mlp',
                      init_weights=gmlp_init_w, frozen=gmlp_frozen,
                      model=model)

############################
# Controller MLP definition
############################
        cmlp_dims = [crnn_dim] + [ctrl_mlp_dim] * \
            (num_ctrl_mlp_layers - 1) + [9]
        cmlp_act = [tf.nn.relu] * (num_ctrl_mlp_layers - 1) + [None]
        cmlp_dropout = None

        pt = pretrain_net or pretrain_ctrl_net
        if pt:
            log.info('Loading pretrained controller MLP weights from {}'.format(
                pt))
            h5f = h5py.File(pt, 'r')
            cmlp_init_w = [{'w': h5f['ctrl_mlp_w_{}'.format(ii)][:],
                            'b': h5f['ctrl_mlp_b_{}'.format(ii)][:]}
                           for ii in xrange(num_ctrl_mlp_layers)]
            cmlp_frozen = [freeze_ctrl_rnn] * num_ctrl_mlp_layers
        else:
            cmlp_init_w = None
            cmlp_frozen = [freeze_ctrl_rnn] * num_ctrl_mlp_layers

        cmlp = nn.mlp(cmlp_dims, cmlp_act, add_bias=True,
                      dropout_keep=cmlp_dropout,
                      phase_train=phase_train, wd=wd, scope='ctrl_mlp',
                      init_weights=cmlp_init_w, frozen=cmlp_frozen,
                      model=model)

###########################
# Attention CNN definition
###########################
        acnn_filters = attn_cnn_filter_size
        acnn_nlayers = len(acnn_filters)
        acnn_channels = [acnn_inp_depth] + attn_cnn_depth
        acnn_pool = attn_cnn_pool
        acnn_act = [tf.nn.relu] * acnn_nlayers
        acnn_use_bn = [use_bn] * acnn_nlayers

        pt = pretrain_net or pretrain_attn_net
        if pt:
            log.info('Loading pretrained attention CNN weights from {}'.format(
                pt))
            h5f = h5py.File(pt, 'r')
            acnn_init_w = [{'w': h5f['attn_cnn_w_{}'.format(ii)][:],
                            'b': h5f['attn_cnn_b_{}'.format(ii)][:]}
                           for ii in xrange(acnn_nlayers)]
            for ii in xrange(acnn_nlayers):
                for tt in xrange(timespan):
                    for w in ['beta', 'gamma']:
                        key = 'attn_cnn_{}_{}_{}'.format(ii, tt, w)
                        acnn_init_w[ii]['{}_{}'.format(w, tt)] = h5f[key][:]
            acnn_frozen = [freeze_attn_net] * acnn_nlayers
        else:
            acnn_init_w = None
            acnn_frozen = [freeze_attn_net] * acnn_nlayers

        acnn = nn.cnn(acnn_filters, acnn_channels, acnn_pool, acnn_act,
                      acnn_use_bn, phase_train=phase_train, wd=wd,
                      scope='attn_cnn', model=model,
                      init_weights=acnn_init_w,
                      frozen=acnn_frozen)

        x_patch = [None] * timespan
        h_acnn = [None] * timespan
        h_acnn_last = [None] * timespan

############################
# Attention MLP definition
############################
        acnn_subsample = np.array(acnn_pool).prod()
        acnn_h = filter_height / acnn_subsample
        acnn_w = filter_width / acnn_subsample
        amlp_inp_dim = acnn_h * acnn_w * acnn_channels[-1]
        core_depth = attn_mlp_depth
        core_dim = acnn_h * acnn_w * core_depth
        amlp_dims = [amlp_inp_dim] + [core_dim] * num_attn_mlp_layers
        amlp_act = [tf.nn.relu] * num_attn_mlp_layers
        amlp_dropout = None

        pt = pretrain_net or pretrain_attn_net
        if pt:
            log.info('Loading pretrained attention MLP weights from {}'.format(
                pt))
            h5f = h5py.File(pt, 'r')
            amlp_init_w = [{'w': h5f['attn_mlp_w_{}'.format(ii)][:],
                            'b': h5f['attn_mlp_b_{}'.format(ii)][:]}
                           for ii in xrange(num_attn_mlp_layers)]
            amlp_frozen = [freeze_attn_net] * num_attn_mlp_layers
        else:
            amlp_init_w = None
            amlp_frozen = [freeze_attn_net] * num_attn_mlp_layers

        amlp = nn.mlp(amlp_dims, amlp_act, dropout_keep=amlp_dropout,
                      phase_train=phase_train, wd=wd, scope='attn_mlp',
                      init_weights=amlp_init_w,
                      frozen=amlp_frozen,
                      model=model)

##########################
# Score MLP definition
##########################
        pt = pretrain_net
        if pt:
            log.info('Loading score mlp weights from {}'.format(pt))
            h5f = h5py.File(pt, 'r')
            smlp_init_w = [{'w': h5f['score_mlp_w_{}'.format(ii)][:],
                            'b': h5f['score_mlp_b_{}'.format(ii)][:]}
                           for ii in xrange(1)]
        else:
            smlp_init_w = None
        smlp = nn.mlp([crnn_dim + core_dim, 1], [tf.sigmoid], wd=wd,
                      scope='score_mlp', init_weights=smlp_init_w,
                      model=model)
        s_out = [None] * timespan

#############################
# Attention DCNN definition
#############################
        adcnn_filters = attn_dcnn_filter_size
        adcnn_nlayers = len(adcnn_filters)
        adcnn_unpool = attn_dcnn_pool
        adcnn_act = [tf.nn.relu] * adcnn_nlayers
        adcnn_channels = [attn_mlp_depth] + attn_dcnn_depth

        adcnn_bn_nlayers = adcnn_nlayers
        # adcnn_bn_nlayers = adcnn_nlayers - 1
        adcnn_use_bn = [use_bn] * adcnn_bn_nlayers + \
            [False] * (adcnn_nlayers - adcnn_bn_nlayers)
        adcnn_skip_ch = [0] + acnn_channels[:: -1][1:]

        pt = pretrain_net or pretrain_attn_net
        if pt:
            log.info('Loading pretrained attention DCNN weights from {}'.format(
                pt))
            h5f = h5py.File(pt, 'r')
            adcnn_init_w = [{'w': h5f['attn_dcnn_w_{}'.format(ii)][:],
                             'b': h5f['attn_dcnn_b_{}'.format(ii)][:]}
                            for ii in xrange(adcnn_nlayers)]
            for ii in xrange(adcnn_bn_nlayers):
                for tt in xrange(timespan):
                    for w in ['beta', 'gamma']:
                        key = 'attn_dcnn_{}_{}_{}'.format(ii, tt, w)
                        adcnn_init_w[ii]['{}_{}'.format(w, tt)] = h5f[key][:]
            
            adcnn_frozen = [freeze_attn_net] * adcnn_nlayers
        else:
            adcnn_init_w = None
            adcnn_frozen = [freeze_attn_net] * adcnn_nlayers

        adcnn = nn.dcnn(adcnn_filters, adcnn_channels, adcnn_unpool,
                        adcnn_act, use_bn=adcnn_use_bn, skip_ch=adcnn_skip_ch,
                        phase_train=phase_train, wd=wd, model=model,
                        init_weights=adcnn_init_w,
                        frozen=adcnn_frozen,
                        scope='attn_dcnn')
        h_adcnn = [None] * timespan

##########################
# Attention box
##########################
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
        attn_box = [None] * timespan
        iou_soft_box = [None] * timespan
        const_ones = tf.ones(tf.pack([num_ex, filter_height, filter_width, 1]))
        attn_box_beta = tf.constant([-5.0])
        attn_box_gamma = [None] * timespan

#############################
# Groundtruth attention box
#############################
        # [B, T, 2]
        attn_ctr_gt, attn_size_gt, attn_lg_var_gt, attn_box_gt, \
            attn_top_left_gt, attn_bot_right_gt = \
            base.get_gt_attn(y_gt,
                             padding_ratio=attn_box_padding_ratio,
                             center_shift_ratio=0.0,
                             min_padding=padding + 4)
        attn_ctr_gt_noise, attn_size_gt_noise, attn_lg_var_gt_noise, \
            attn_box_gt_noise, \
            attn_top_left_gt_noise, attn_bot_right_gt_noise = \
            base.get_gt_attn(y_gt,
                             padding_ratio=tf.random_uniform(
                                 tf.pack([num_ex, timespan, 1]),
                                 attn_box_padding_ratio - gt_box_pad_noise,
                                 attn_box_padding_ratio + gt_box_pad_noise),
                             center_shift_ratio=tf.random_uniform(
                                 tf.pack([num_ex, timespan, 2]),
                                 -gt_box_ctr_noise, gt_box_ctr_noise),
                             min_padding=padding + 4)
        attn_ctr_norm_gt = base.get_normalized_center(
            attn_ctr_gt, inp_height, inp_width)
        attn_lg_size_gt = base.get_normalized_size(
            attn_size_gt, inp_height, inp_width)

##########################
# Groundtruth mix
##########################
        grd_match_cum = tf.zeros(tf.pack([num_ex, timespan]))

        # Scale mix ratio on different timesteps.
        if knob_use_timescale:
            gt_knob_time_scale = tf.reshape(
                1.0 + tf.log(1.0 + tf.to_float(tf.range(timespan)) * 3.0),
                [1, timespan, 1])
        else:
            gt_knob_time_scale = tf.ones([1, timespan, 1])

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

##########################
# Segmentation output
##########################
        y_out = [None] * timespan
        y_out_lg_gamma = [None] * timespan
        y_out_beta = tf.constant([-5.0])

##########################
# Computation graph
##########################
        for tt in xrange(timespan):
            # Controller CNN
            ccnn_inp = tf.concat(3, [x, canvas])
            acnn_inp = ccnn_inp
            h_ccnn[tt] = ccnn(ccnn_inp)
            _h_ccnn = h_ccnn[tt]
            h_ccnn_last = _h_ccnn[-1]

            # Controller RNN [B, R1]
            if ctrl_rnn_inp_struct == 'dense':
                crnn_inp = tf.reshape(h_ccnn_last, [-1, crnn_inp_dim])
                crnn_state[tt], crnn_g_i[tt], crnn_g_f[tt], crnn_g_o[
                    tt] = crnn_cell(crnn_inp, crnn_state[tt - 1])
                h_crnn[tt] = tf.slice(
                    crnn_state[tt], [0, crnn_dim], [-1, crnn_dim])

                ctrl_out = cmlp(h_crnn[tt])[-1]

            elif ctrl_rnn_inp_struct == 'attn':
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

                # Inner glimpse RNN
                for tt2 in xrange(num_ctrl_rnn_iter):
                    crnn_glimpse = tf.reduce_sum(
                        crnn_inp * crnn_glimpse_map[tt][tt2], [1])
                    crnn_state[tt][tt2], crnn_g_i[tt][tt2], crnn_g_f[tt][tt2], \
                        crnn_g_o[tt][tt2] = crnn_cell(
                            crnn_glimpse, crnn_state[tt][tt2 - 1])
                    h_crnn[tt][tt2] = tf.slice(
                        crnn_state[tt][tt2], [0, crnn_dim], [-1, crnn_dim])
                    h_gmlp = gmlp(h_crnn[tt][tt2])
                    if tt2 < num_ctrl_rnn_iter - 1:
                        crnn_glimpse_map[tt][
                            tt2 + 1] = tf.expand_dims(h_gmlp[-1], 2)

                ctrl_out = cmlp(h_crnn[tt][-1])[-1]

            attn_ctr_norm[tt] = tf.slice(ctrl_out, [0, 0], [-1, 2])
            attn_lg_size[tt] = tf.slice(ctrl_out, [0, 2], [-1, 2])

            # Restrict to (-1, 1), (-inf, 0)
            if squash_ctrl_params:
                attn_ctr_norm[tt] = tf.tanh(attn_ctr_norm[tt])
                attn_lg_size[tt] = -tf.nn.softplus(attn_lg_size[tt])

            attn_ctr[tt], attn_size[tt] = base.get_unnormalized_attn(
                attn_ctr_norm[tt], attn_lg_size[tt], inp_height, inp_width)

            attn_lg_var[tt] = tf.zeros(tf.pack([num_ex, 2]))

            if fixed_gamma:
                attn_lg_gamma[tt] = tf.constant([0.0])
                # attn_box_lg_gamma[tt] = tf.constant([2.0])
                y_out_lg_gamma[tt] = tf.constant([2.0])
            else:
                attn_lg_gamma[tt] = tf.slice(ctrl_out, [0, 6], [-1, 1])
                # attn_box_lg_gamma[tt] = tf.slice(ctrl_out, [0, 7], [-1, 1])
                y_out_lg_gamma[tt] = tf.slice(ctrl_out, [0, 8], [-1, 1])

            attn_box_lg_gamma[tt] = tf.slice(ctrl_out, [0, 7], [-1, 1])
            attn_gamma[tt] = tf.reshape(
                tf.exp(attn_lg_gamma[tt]), [-1, 1, 1, 1])
            attn_box_gamma[tt] = tf.reshape(tf.exp(
                attn_box_lg_gamma[tt]), [-1, 1, 1, 1])
            y_out_lg_gamma[tt] = tf.reshape(y_out_lg_gamma[tt], [-1, 1, 1, 1])

            # Initial filters (predicted)
            filter_y = base.get_gaussian_filter(
                attn_ctr[tt][:, 0], attn_size[tt][:, 0],
                attn_lg_var[tt][:, 0], inp_height, filter_height)
            filter_x = base.get_gaussian_filter(
                attn_ctr[tt][:, 1], attn_size[tt][:, 1],
                attn_lg_var[tt][:, 1], inp_width, filter_width)
            filter_y_inv = tf.transpose(filter_y, [0, 2, 1])
            filter_x_inv = tf.transpose(filter_x, [0, 2, 1])

            # Attention box
            attn_box[tt] = base.extract_patch(
                const_ones * attn_box_gamma[tt],
                filter_y_inv, filter_x_inv, 1)
            attn_box[tt] = tf.sigmoid(attn_box[tt] + attn_box_beta)
            attn_box[tt] = tf.reshape(attn_box[tt],
                                      [-1, 1, inp_height, inp_width])

            # Kick in GT bbox.
            if use_knob:
                if fixed_order:
                    attn_ctr_gtm = attn_ctr_gt_noise[:, tt, :]
                    attn_delta_gtm = attn_delta_gt_noise[:, tt, :]
                    attn_size_gtm = attn_size_gt_noise[:, tt, :]
                else:
                    iou_soft_box[tt] = base.f_inter(
                        attn_box[tt], attn_box_gt) / \
                        base.f_union(attn_box[tt], attn_box_gt, eps=1e-5)
                    grd_match = base.f_greedy_match(
                        iou_soft_box[tt], grd_match_cum)

                    # [B, T, 1]
                    grd_match = tf.expand_dims(grd_match, 2)
                    attn_ctr_gtm = tf.reduce_sum(
                        grd_match * attn_ctr_gt_noise, 1)
                    attn_size_gtm = tf.reduce_sum(
                        grd_match * attn_size_gt_noise, 1)

                attn_ctr[tt] = phase_train_f * gt_knob_box[:, tt, 0: 1] * \
                    attn_ctr_gtm + \
                    (1 - phase_train_f * gt_knob_box[:, tt, 0: 1]) * \
                    attn_ctr[tt]
                attn_size[tt] = phase_train_f * gt_knob_box[:, tt, 0: 1] * \
                    attn_size_gtm + \
                    (1 - phase_train_f * gt_knob_box[:, tt, 0: 1]) * \
                    attn_size[tt]

            attn_top_left[tt], attn_bot_right[tt] = base.get_box_coord(
                attn_ctr[tt], attn_size[tt])

            filter_y = base.get_gaussian_filter(
                attn_ctr[tt][:, 0], attn_size[tt][:, 0],
                attn_lg_var[tt][:, 0], inp_height, filter_height)
            filter_x = base.get_gaussian_filter(
                attn_ctr[tt][:, 1], attn_size[tt][:, 1],
                attn_lg_var[tt][:, 1], inp_width, filter_width)
            filter_y_inv = tf.transpose(filter_y, [0, 2, 1])
            filter_x_inv = tf.transpose(filter_x, [0, 2, 1])
            tf.stop_gradient(filter_y)
            tf.stop_gradient(filter_x)
            # Attended patch [B, A, A, D]
            x_patch[tt] = attn_gamma[tt] * base.extract_patch(
                acnn_inp, filter_y, filter_x, acnn_inp_depth)

            # CNN [B, A, A, D] => [B, RH2, RW2, RD2]
            h_acnn[tt] = acnn(x_patch[tt])
            h_acnn_last[tt] = h_acnn[tt][-1]

            # Dense segmentation network [B, R] => [B, M]
            amlp_inp = h_acnn_last[tt]
            amlp_inp = tf.reshape(amlp_inp, [-1, amlp_inp_dim])
            h_core = amlp(amlp_inp)[-1]
            h_core_img = tf.reshape(
                h_core, [-1, acnn_h, acnn_w, attn_mlp_depth])

            # DCNN
            skip = [None] + h_acnn[tt][::-1][1:] + [x_patch[tt]]
            h_adcnn[tt] = adcnn(h_core_img, skip=skip)

            # Output
            y_out[tt] = base.extract_patch(
                h_adcnn[tt][-1], filter_y_inv, filter_x_inv, 1)
            y_out[tt] = tf.exp(y_out_lg_gamma[tt]) * y_out[tt] + y_out_beta
            y_out[tt] = tf.sigmoid(y_out[tt])
            y_out[tt] = tf.reshape(y_out[tt], [-1, 1, inp_height, inp_width])

            # Scoring network
            if ctrl_rnn_inp_struct == 'dense':
                smlp_inp = tf.concat(1, [h_crnn[tt], h_core])
            elif ctrl_rnn_inp_struct == 'attn':
                smlp_inp = tf.concat(1, [h_crnn[tt][-1], h_core])
            s_out[tt] = smlp(smlp_inp)[-1]

            # Here is the knob kick in GT segmentations at this timestep.
            # [B, N, 1, 1]
            if use_knob:
                _gt_knob_segm = tf.expand_dims(
                    tf.expand_dims(gt_knob_segm[:, tt, 0: 1], 2), 3)

                if fixed_order:
                    _y_out = tf.expand_dims(y_gt[:, tt, :, :], 3)
                else:
                    grd_match = tf.expand_dims(grd_match, 3)
                    _y_out = tf.expand_dims(tf.reduce_sum(
                        grd_match * y_gt, 1), 3)
                # Add independent uniform noise to groundtruth.
                _noise = tf.random_uniform(
                    tf.pack([num_ex, inp_height, inp_width, 1]), 0,
                    gt_segm_noise)
                _y_out = _y_out - _y_out * _noise
                _y_out = phase_train_f * _gt_knob_segm * _y_out + \
                    (1 - phase_train_f * _gt_knob_segm) * \
                    tf.reshape(y_out[tt], [-1, inp_height, inp_width, 1])
            else:
                _y_out = tf.reshape(y_out[tt],
                                    [-1, inp_height, inp_width, 1])
            canvas = tf.stop_gradient(tf.maximum(_y_out, canvas))

#########################
# Model outputs
#########################
        s_out = tf.concat(1, s_out)
        model['s_out'] = s_out
        y_out = tf.concat(1, y_out)
        model['y_out'] = y_out
        attn_box = tf.concat(1, attn_box)
        model['attn_box'] = attn_box
        x_patch = tf.concat(1, [tf.expand_dims(x_patch[tt], 1)
                                for tt in xrange(timespan)])
        model['x_patch'] = x_patch

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
        model['attn_ctr'] = attn_ctr
        model['attn_size'] = attn_size
        model['attn_top_left'] = attn_top_left
        model['attn_bot_right'] = attn_bot_right
        model['attn_box_gt'] = attn_box_gt
        attn_ctr_norm = tf.concat(1, [tf.expand_dims(tmp, 1)
                                      for tmp in attn_ctr_norm])
        attn_lg_size = tf.concat(1, [tf.expand_dims(tmp, 1)
                                     for tmp in attn_lg_size])
        model['attn_ctr_norm'] = attn_ctr_norm
        model['attn_lg_size'] = attn_lg_size
        attn_params = tf.concat(2, [attn_ctr_norm, attn_lg_size])
        attn_params_gt = tf.concat(2, [attn_ctr_norm_gt, attn_lg_size_gt])

#########################
# Loss function
#########################
        y_gt_shape = tf.shape(y_gt)
        num_ex_f = tf.to_float(y_gt_shape[0])
        max_num_obj = tf.to_float(y_gt_shape[1])

############################
# Box loss
############################
        if fixed_order:
            # [B, T] for fixed order.
            iou_soft_box = base.f_iou(attn_box, attn_box_gt, pairwise=False)
        else:
            if use_knob:
                # [B, T, T] for matching.
                iou_soft_box = tf.concat(
                    1, [tf.expand_dims(iou_soft_box[tt], 1)
                        for tt in xrange(timespan)])
            else:
                iou_soft_box = base.f_iou(attn_box, attn_box_gt,
                                          timespan, pairwise=True)

        identity_match = base.get_identity_match(num_ex, timespan, s_gt)
        if fixed_order:
            match_box = identity_match
        else:
            match_box = base.f_segm_match(iou_soft_box, s_gt)

        model['match_box'] = match_box
        match_sum_box = tf.reduce_sum(match_box, reduction_indices=[2])
        match_count_box = tf.reduce_sum(match_sum_box, reduction_indices=[1])
        match_count_box = tf.maximum(1.0, match_count_box)

        # [B] if fixed order, [B, T] if matching.
        if fixed_order:
            iou_soft_box_mask = iou_soft_box
        else:
            iou_soft_box_mask = tf.reduce_sum(iou_soft_box * match_box, [1])
        iou_soft_box = tf.reduce_sum(iou_soft_box_mask, [1])
        iou_soft_box = tf.reduce_sum(iou_soft_box / match_count_box) / num_ex_f

        if box_loss_fn == 'mse':
            box_loss = base.f_match_loss(
                attn_params, attn_params_gt, match_box, timespan,
                base.f_squared_err, model=model)
        elif box_loss_fn == 'huber':
            box_loss = base.f_match_loss(
                attn_params, attn_params_gt, match_box, timespan, base.f_huber)
        elif box_loss_fn == 'iou':
            box_loss = -iou_soft_box
        elif box_loss_fn == 'wt_cov':
            box_loss = -base.f_weighted_coverage(iou_soft_box, attn_box_gt)
        elif box_loss_fn == 'bce':
            box_loss_fn = base.f_match_loss(
                y_out, y_gt, match_box, timespan, f_bce)
        else:
            raise Exception('Unknown box_loss_fn: {}'.format(box_loss_fn))
        model['box_loss'] = box_loss

        box_loss_coeff = tf.constant(1.0)
        model['box_loss_coeff'] = box_loss_coeff
        tf.add_to_collection('losses', box_loss_coeff * box_loss)

##############################
# Segmentation loss
##############################
        # IoU (soft)
        iou_soft_pairwise = base.f_iou(y_out, y_gt, timespan, pairwise=True)
        real_match = base.f_segm_match(iou_soft_pairwise, s_gt)
        if fixed_order:
            iou_soft = base.f_iou(y_out, y_gt, pairwise=False)
            match = identity_match
        else:
            iou_soft = iou_soft_pairwise
            match = real_match
        model['match'] = match
        match_sum = tf.reduce_sum(match, reduction_indices=[2])
        match_count = tf.reduce_sum(match_sum, reduction_indices=[1])
        match_count = tf.maximum(1.0, match_count)

        # Weighted coverage (soft)
        wt_cov_soft = base.f_weighted_coverage(iou_soft_pairwise, y_gt)
        model['wt_cov_soft'] = wt_cov_soft
        unwt_cov_soft = base.f_unweighted_coverage(
            iou_soft_pairwise, match_count)
        model['unwt_cov_soft'] = unwt_cov_soft

        # [B] if fixed order, [B, T] if matching.
        if fixed_order:
            iou_soft_mask = iou_soft
        else:
            iou_soft_mask = tf.reduce_sum(iou_soft * match, [1])
        iou_soft = tf.reduce_sum(iou_soft_mask, [1])
        iou_soft = tf.reduce_sum(iou_soft / match_count) / num_ex_f
        model['iou_soft'] = iou_soft

        if segm_loss_fn == 'iou':
            segm_loss = -iou_soft
        elif segm_loss_fn == 'wt_cov':
            segm_loss = -wt_cov_soft
        elif segm_loss_fn == 'bce':
            segm_loss = f_match_bce(y_out, y_gt, match, timespan)
        else:
            raise Exception('Unknown segm_loss_fn: {}'.format(segm_loss_fn))
        model['segm_loss'] = segm_loss
        segm_loss_coeff = tf.constant(1.0)
        tf.add_to_collection('losses', segm_loss_coeff * segm_loss)

####################
# Score loss
####################
        conf_loss = base.f_conf_loss(s_out, match, timespan, use_cum_min=True)
        model['conf_loss'] = conf_loss
        tf.add_to_collection('losses', loss_mix_ratio * conf_loss)

####################
# Total loss
####################
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        model['loss'] = total_loss

####################
# Optimizer
####################
        learn_rate = tf.train.exponential_decay(
            base_learn_rate, global_step, steps_per_learn_rate_decay,
            learn_rate_decay, staircase=True)
        model['learn_rate'] = learn_rate
        eps = 1e-7

        train_step = GradientClipOptimizer(
            tf.train.AdamOptimizer(learn_rate, epsilon=eps),
            clip=clip_gradient).minimize(total_loss, global_step=global_step)
        model['train_step'] = train_step

####################
# Statistics
####################
        # Here statistics (hard measures) is always using matching.
        y_out_hard = tf.to_float(y_out > 0.5)
        iou_hard = base.f_iou(y_out_hard, y_gt, timespan, pairwise=True)
        wt_cov_hard = base.f_weighted_coverage(iou_hard, y_gt)
        model['wt_cov_hard'] = wt_cov_hard
        unwt_cov_hard = base.f_unweighted_coverage(iou_hard, match_count)
        model['unwt_cov_hard'] = unwt_cov_hard
        iou_hard_mask = tf.reduce_sum(iou_hard * real_match, [1])
        iou_hard = tf.reduce_sum(tf.reduce_sum(iou_hard_mask, [1]) /
                                 match_count) / num_ex_f
        model['iou_hard'] = iou_hard

        dice = base.f_dice(y_out_hard, y_gt, timespan, pairwise=True)
        dice = tf.reduce_sum(tf.reduce_sum(
            dice * real_match, reduction_indices=[1, 2]) / match_count) / \
            num_ex_f
        model['dice'] = dice

        model['count_acc'] = base.f_count_acc(s_out, s_gt)
        model['dic'] = base.f_dic(s_out, s_gt, abs=False)
        model['dic_abs'] = base.f_dic(s_out, s_gt, abs=True)

################################
# Controller output statistics
################################
        if fixed_gamma:
            attn_lg_gamma_mean = tf.constant([0.0])
            attn_box_lg_gamma_mean = tf.constant([2.0])
            y_out_lg_gamma_mean = tf.constant([2.0])
        else:
            attn_lg_gamma_mean = tf.reduce_sum(
                attn_lg_gamma) / num_ex_f / timespan
            attn_box_lg_gamma_mean = tf.reduce_sum(
                attn_box_lg_gamma) / num_ex_f / timespan
            y_out_lg_gamma_mean = tf.reduce_sum(
                y_out_lg_gamma) / num_ex_f / timespan
        model['attn_lg_gamma_mean'] = attn_lg_gamma_mean
        model['attn_box_lg_gamma_mean'] = attn_box_lg_gamma_mean
        model['y_out_lg_gamma_mean'] = y_out_lg_gamma_mean

####################
# Debug gradients
####################
        ctrl_mlp_b_grad = tf.gradients(total_loss, model['ctrl_mlp_b_0'])
        model['ctrl_mlp_b_grad'] = ctrl_mlp_b_grad[0]

####################
# Glimpse
####################
        # T * T2 * [H', W'] => [T, T2, H', W']
        if ctrl_rnn_inp_struct == 'attn':
            crnn_glimpse_map = tf.concat(
                1, [tf.expand_dims(tf.concat(
                    1, [tf.expand_dims(crnn_glimpse_map[tt][tt2], 1)
                        for tt2 in xrange(num_ctrl_rnn_iter)]), 1)
                    for tt in xrange(timespan)])
            crnn_glimpse_map = tf.reshape(
                crnn_glimpse_map, [-1, timespan, num_ctrl_rnn_iter, crnn_h,
                                   crnn_w])
            model['ctrl_rnn_glimpse_map'] = crnn_glimpse_map

    return model
