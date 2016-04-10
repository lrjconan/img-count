import cslab_environ

import ris_base as base
from utils import logger
from utils.grad_clip_optim import GradientClipOptimizer
import fg_segm_reader
import h5py
import image_ops as img
import nnlib as nn
import numpy as np
import tensorflow as tf

log = logger.get()


def _get_idx_mask(idx, timespan):
    """Computes the binary mask given an index.

    Args:
        idx: [B]

    Returns:
        mask: [B, T]
    """
    eye = tf.constant(np.eye(timespan, dtype='float32'))
    return tf.gather(eye, idx)


def _get_identity_match(num_ex, timespan, s_gt):
    zeros = tf.zeros(tf.pack([num_ex, timespan, timespan]))
    eye = tf.expand_dims(tf.constant(np.eye(timespan), dtype='float32'), 0)
    mask_x = tf.expand_dims(s_gt, 1)
    mask_y = tf.expand_dims(s_gt, 2)
    match = zeros + eye
    match = match * mask_x * mask_y

    return match


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
    loss_mix_ratio = opt['loss_mix_ratio']
    base_learn_rate = opt['base_learn_rate']
    learn_rate_decay = opt['learn_rate_decay']
    steps_per_learn_rate_decay = opt['steps_per_learn_rate_decay']
    gt_box_ctr_noise = opt['gt_box_ctr_noise']
    gt_box_pad_noise = opt['gt_box_pad_noise']
    gt_segm_noise = opt['gt_segm_noise']
    clip_gradient = opt['clip_gradient']
    fixed_order = opt['fixed_order']
    if add_skip_conn not in opt:
        add_skip_conn = True
    else:
        add_skip_conn = opt['add_skip_conn']

    rnd_hflip = opt['rnd_hflip']
    rnd_vflip = opt['rnd_vflip']
    rnd_transpose = opt['rnd_transpose']
    rnd_colour = opt['rnd_colour']

    with tf.device(base.get_device_fn(device)):
        # Input definition
        # Input image, [B, H, W, D]
        x = tf.placeholder('float', [None, inp_height, inp_width, inp_depth])
        x_shape = tf.shape(x)
        num_ex = x_shape[0]
        y_gt = tf.placeholder('float', [None, timespan, inp_height, inp_width])
        # Groundtruth confidence score, [B, T]
        s_gt = tf.placeholder('float', [None, timespan])
        # Order in which we feed in the samples
        order = tf.placeholder('int32', [None, timespan])
        # Whether in training stage.
        phase_train = tf.placeholder('bool')
        phase_train_f = tf.to_float(phase_train)
        model['x'] = x
        model['y_gt'] = y_gt
        model['s_gt'] = s_gt
        model['phase_train'] = phase_train
        model['order'] = order

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
        acnn_inp_depth = inp_depth + 1

        # Groundtruth bounding box, [B, T, 2]
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
                        min_padding=25.0)
        attn_ctr = [None] * timespan
        attn_size = [None] * timespan
        attn_top_left = [None] * timespan
        attn_bot_right = [None] * timespan

        # Attention CNN definition
        acnn_filters = attn_cnn_filter_size
        acnn_nlayers = len(acnn_filters)
        acnn_channels = [acnn_inp_depth] + attn_cnn_depth
        acnn_pool = attn_cnn_pool
        acnn_act = [tf.nn.relu] * acnn_nlayers
        acnn_use_bn = [use_bn] * acnn_nlayers
        acnn = nn.cnn(acnn_filters, acnn_channels, acnn_pool, acnn_act,
                      acnn_use_bn, phase_train=phase_train, wd=wd,
                      scope='attn_cnn', model=model)

        x_patch = [None] * timespan
        h_acnn = [None] * timespan
        h_acnn_last = [None] * timespan

        # Attention RNN definition
        acnn_subsample = np.array(acnn_pool).prod()
        arnn_h = filter_height / acnn_subsample
        arnn_w = filter_width / acnn_subsample
        amlp_inp_dim = arnn_h * arnn_w * acnn_channels[-1]

        # Attention MLP definition
        core_depth = attn_mlp_depth
        core_dim = arnn_h * arnn_w * core_depth
        amlp_dims = [amlp_inp_dim] + [core_dim] * num_attn_mlp_layers
        amlp_act = [tf.nn.relu] * num_attn_mlp_layers
        amlp_dropout = None
        # amlp_dropout = [1.0 - mlp_dropout_ratio] * num_attn_mlp_layers
        amlp = nn.mlp(amlp_dims, amlp_act, dropout_keep=amlp_dropout,
                      phase_train=phase_train, wd=wd, scope='attn_mlp',
                      model=model)

        # DCNN [B, RH, RW, MD] => [B, A, A, 1]
        adcnn_filters = attn_dcnn_filter_size
        adcnn_nlayers = len(adcnn_filters)
        adcnn_unpool = attn_dcnn_pool
        adcnn_act = [tf.nn.relu] * adcnn_nlayers
        adcnn_channels = [attn_mlp_depth] + attn_dcnn_depth
        adcnn_use_bn = [use_bn] * (adcnn_nlayers - 1) + [False]
        if add_skip_conn:
            adcnn_skip_ch = [0] + acnn_channels[::-1][1:]
        else:
            adcnn_skip_ch = None
        adcnn = nn.dcnn(adcnn_filters, adcnn_channels, adcnn_unpool,
                        adcnn_act, use_bn=adcnn_use_bn, skip_ch=adcnn_skip_ch,
                        phase_train=phase_train, wd=wd, model=model,
                        scope='attn_dcnn')
        h_adcnn = [None] * timespan

        # Attention box
        attn_box = [None] * timespan

        # Y out
        y_out = [None] * timespan
        # y_out_lg_gamma = [None] * timespan
        y_out_lg_gamma = tf.constant([2.0])
        y_out_beta = tf.constant([-5.0])

        for tt in xrange(timespan):
            # Get a new greedy match based on order.
            if fixed_order:
                attn_ctr[tt] = attn_ctr_gt_noise[:, tt, :]
                attn_size[tt] = attn_size_gt_noise[:, tt, :]
            else:
                mask = _get_idx_mask(order[:, tt], timespan)
                # [B, T, 1]
                mask = tf.expand_dims(mask, 2)
                attn_ctr[tt] = tf.reduce_sum(mask * attn_ctr_gt_noise, 1)
                attn_size[tt] = tf.reduce_sum(mask * attn_size_gt_noise, 1)
                
            attn_top_left[tt], attn_bot_right[tt] = base.get_box_coord(
                attn_ctr[tt], attn_size[tt])

            # [B, H, H']
            filter_y = base.get_gaussian_filter(
                attn_ctr[tt][:, 0], attn_size[tt][:, 0], 0.0,
                inp_height, filter_height)

            # [B, W, W']
            filter_x = base.get_gaussian_filter(
                attn_ctr[tt][:, 1], attn_size[tt][:, 1], 0.0,
                inp_width, filter_width)

            # [B, H', H]
            filter_y_inv = tf.transpose(filter_y, [0, 2, 1])

            # [B, W', W]
            filter_x_inv = tf.transpose(filter_x, [0, 2, 1])

            # Attended patch [B, A, A, D]
            acnn_inp = tf.concat(3, [x, canvas])
            x_patch[tt] = base.extract_patch(
                acnn_inp, filter_y, filter_x, acnn_inp_depth)

            if tt == 0:
                model['filter_x'] = filter_x
                model['filter_y'] = filter_y
                model['x_patch'] = x_patch[tt]

            # CNN [B, A, A, D] => [B, RH2, RW2, RD2]
            h_acnn[tt] = acnn(x_patch[tt])
            h_acnn_last[tt] = h_acnn[tt][-1]
            amlp_inp = h_acnn_last[tt]
            amlp_inp = tf.reshape(amlp_inp, [-1, amlp_inp_dim])
            h_core = amlp(amlp_inp)[-1]
            h_core = tf.reshape(h_core, [-1, arnn_h, arnn_w, attn_mlp_depth])

            # DCNN
            if add_skip_conn:
                skip = [None] + h_acnn[tt][::-1][1:] + [x_patch[tt]]
            else:
                skip = None
            h_adcnn[tt] = adcnn(h_core, skip=skip)

            # Output
            y_out[tt] = base.extract_patch(
                h_adcnn[tt][-1], filter_y_inv, filter_x_inv, 1)
            y_out[tt] = tf.exp(y_out_lg_gamma) * y_out[tt] + y_out_beta
            y_out[tt] = tf.sigmoid(y_out[tt])
            y_out[tt] = tf.reshape(y_out[tt], [-1, 1, inp_height, inp_width])

            # Canvas
            if fixed_order:
                _y_out = y_gt[:, tt, :, :]
            else:
                mask = tf.expand_dims(mask, 3)
                _y_out = tf.reduce_sum(mask * y_gt, 1)

            _y_out = tf.expand_dims(_y_out, 3)
            # Add independent uniform noise to groundtruth.
            _noise = tf.random_uniform(
                tf.pack([num_ex, inp_height, inp_width, 1]),
                0, gt_segm_noise)
            _y_out = _y_out - _y_out * _noise
            canvas += _y_out

        y_out = tf.concat(1, y_out)
        model['y_out'] = y_out
        x_patch = tf.concat(1, [tf.expand_dims(x_patch[tt], 1)
                                for tt in xrange(timespan)])
        model['x_patch'] = x_patch

        # Loss function
        learn_rate = tf.train.exponential_decay(
            base_learn_rate, global_step, steps_per_learn_rate_decay,
            learn_rate_decay, staircase=True)
        model['learn_rate'] = learn_rate
        eps = 1e-7

        y_gt_shape = tf.shape(y_gt)
        num_ex_f = tf.to_float(y_gt_shape[0])
        max_num_obj = tf.to_float(y_gt_shape[1])

        # Loss for fine segmentation
        identity_match = _get_identity_match(num_ex, timespan, s_gt)
        iou_soft_pairwise = base.f_iou(y_out, y_gt, timespan, pairwise=True)
        real_match = base.f_segm_match(iou_soft_pairwise, s_gt)
        if fixed_order:
            iou_soft = base.f_iou(y_out, y_gt, pairwise=False)
            match = identity_match
        else:
            iou_soft = iou_soft_pairwise
            match = real_match
        # match = base.f_segm_match(iou_soft, s_gt)
        model['match'] = match
        match_sum = tf.reduce_sum(match, reduction_indices=[2])
        match_count = tf.reduce_sum(match_sum, reduction_indices=[1])
        match_count = tf.maximum(1.0, match_count)

        # Weighted coverage (soft)
        wt_cov_soft = base.f_weighted_coverage(iou_soft_pairwise, y_gt)
        model['wt_cov_soft'] = wt_cov_soft
        unwt_cov_soft = base.f_unweighted_coverage(iou_soft_pairwise, match_count)
        model['unwt_cov_soft'] = unwt_cov_soft

        # IOU (soft)
        if fixed_order:
            iou_soft_mask = iou_soft
        else:
            iou_soft_mask = tf.reduce_sum(iou_soft * match, [1])
        iou_soft = tf.reduce_sum(iou_soft_mask, [1])
        iou_soft = tf.reduce_sum(iou_soft / match_count) / num_ex_f
        model['iou_soft'] = iou_soft
        # gt_wt = f_coverage_weight(y_gt)
        # wt_iou_soft = tf.reduce_sum(tf.reduce_sum(iou_soft_mask * gt_wt, [1]) /
        #                             match_count) / num_ex_f
        # model['wt_iou_soft'] = wt_iou_soft

        if segm_loss_fn == 'iou':
            segm_loss = -iou_soft
        elif segm_loss_fn == 'wt_iou':
            segm_loss = -wt_iou_soft
        elif segm_loss_fn == 'wt_cov':
            segm_loss = -wt_cov_soft
        elif segm_loss_fn == 'bce':
            segm_loss = base.f_match_bce(y_out, y_gt, match, timespan)
        else:
            raise Exception('Unknown segm_loss_fn: {}'.format(segm_loss_fn))
        model['segm_loss'] = segm_loss
        segm_loss_coeff = tf.constant(1.0)
        tf.add_to_collection('losses', segm_loss_coeff * segm_loss)

        total_loss = tf.add_n(tf.get_collection(
            'losses'), name='total_loss')
        model['loss'] = total_loss

        train_step = GradientClipOptimizer(
            tf.train.AdamOptimizer(learn_rate, epsilon=eps),
            clip=clip_gradient).minimize(total_loss, global_step=global_step)
        model['train_step'] = train_step

        # Statistics
        # [B, M, N] * [B, M, N] => [B] * [B] => [1]
        y_out_hard = tf.to_float(y_out > 0.5)
        iou_hard = base.f_iou(y_out_hard, y_gt, timespan, pairwise=True)
        wt_cov_hard = base.f_weighted_coverage(iou_hard, y_gt)
        model['wt_cov_hard'] = wt_cov_hard
        unwt_cov_hard = base.f_unweighted_coverage(iou_hard, match_count)
        model['unwt_cov_hard'] = unwt_cov_hard
        # [B, T]
        iou_hard_mask = tf.reduce_sum(iou_hard * match, [1])
        iou_hard = tf.reduce_sum(tf.reduce_sum(iou_hard_mask, [1]) /
                                 match_count) / num_ex_f
        model['iou_hard'] = iou_hard
        # wt_iou_hard = tf.reduce_sum(tf.reduce_sum(iou_hard_mask * gt_wt, [1]) /
        #                             match_count) / num_ex_f
        # model['wt_iou_hard'] = wt_iou_hard

        # Attention coordinate for debugging [B, T, 2]
        attn_top_left = tf.concat(1, [tf.expand_dims(tmp, 1)
                                      for tmp in attn_top_left])
        attn_bot_right = tf.concat(1, [tf.expand_dims(tmp, 1)
                                       for tmp in attn_bot_right])
        attn_ctr = tf.concat(1, [tf.expand_dims(tmp, 1)
                                 for tmp in attn_ctr])
        attn_size = tf.concat(1, [tf.expand_dims(tmp, 1)
                                  for tmp in attn_size])
        model['attn_ctr'] = attn_ctr
        model['attn_size'] = attn_size
        model['attn_top_left'] = attn_top_left
        model['attn_bot_right'] = attn_bot_right

    return model
