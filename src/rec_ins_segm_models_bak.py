
        # h_ccnn = [tf.concat(1, [tf.expand_dims(h_ccnn[tt][ii], 1)
        #                         for tt in xrange(timespan)])
        #           for ii in xrange(len(ccnn_filters))]
        # model['h_ccnn'] = h_ccnn
        # h_acnn = [tf.concat(1, [tf.expand_dims(h_acnn[tt][ii], 1)
        #                         for tt in xrange(timespan)])
        #           for ii in xrange(len(acnn_filters))]
        # model['h_acnn'] = h_acnn
        # h_dcnn = [tf.concat(1, [tf.expand_dims(h_dcnn[tt][ii], 1)
        #                         for tt in xrange(timespan)])
        #           for ii in xrange(len(dcnn_filters))]
        # model['h_dcnn'] = h_dcnn
        
def get_attn_model(opt, device='/cpu:0'):
    """The original model"""
    model = {}

    timespan = opt['timespan']
    inp_height = opt['inp_height']
    inp_width = opt['inp_width']
    inp_depth = opt['inp_depth']
    padding = opt['padding']
    attn_size = opt['attn_size']

    ctrl_cnn_filter_size = opt['ctrl_cnn_filter_size']
    ctrl_cnn_depth = opt['ctrl_cnn_depth']
    ctrl_rnn_hid_dim = opt['ctrl_rnn_hid_dim']

    num_ctrl_mlp_layers = opt['num_ctrl_mlp_layers']
    ctrl_mlp_dim = opt['ctrl_mlp_dim']

    attn_cnn_filter_size = opt['attn_cnn_filter_size']
    attn_cnn_depth = opt['attn_cnn_depth']
    dcnn_filter_size = opt['dcnn_filter_size']
    dcnn_depth = opt['dcnn_depth']
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

    rnd_hflip = opt['rnd_hflip']
    rnd_vflip = opt['rnd_vflip']
    rnd_transpose = opt['rnd_transpose']
    rnd_colour = opt['rnd_colour']

    with tf.device(_get_device_fn(device)):
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
        model['x'] = x
        model['y_gt'] = y_gt
        model['s_gt'] = s_gt
        model['phase_train'] = phase_train

        # Random image transformation
        x, y_gt = _rnd_img_transformation(
            x, y_gt, padding, phase_train,
            rnd_hflip=rnd_hflip, rnd_vflip=rnd_vflip,
            rnd_transpose=rnd_transpose, rnd_colour=rnd_colour)
        model['x_trans'] = x
        model['y_gt_trans'] = y_gt

        # Controller CNN definition
        ccnn_filters = ctrl_cnn_filter_size
        ccnn_nlayers = len(ccnn_filters)
        ccnn_channels = [inp_depth] + ctrl_cnn_depth
        ccnn_pool = [2] * ccnn_nlayers
        ccnn_act = [tf.nn.relu] * ccnn_nlayers
        ccnn_use_bn = [use_bn] * ccnn_nlayers
        ccnn = nn.cnn(ccnn_filters, ccnn_channels, ccnn_pool, ccnn_act,
                      ccnn_use_bn, phase_train=phase_train, wd=wd,
                      scope='ctrl_cnn', model=model)
        h_ccnn = [None] * timespan

        # Controller RNN definition
        ccnn_subsample = np.array(ccnn_pool).prod()
        crnn_h = inp_height / ccnn_subsample
        crnn_w = inp_width / ccnn_subsample
        crnn_dim = ctrl_rnn_hid_dim
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
            (num_ctrl_mlp_layers - 1) + [7]
        cmlp_act = [tf.nn.relu] * (num_ctrl_mlp_layers - 1) + [None]
        # cmlp_dropout = None
        cmlp_dropout = [1.0 - mlp_dropout_ratio] * num_ctrl_mlp_layers
        cmlp = nn.mlp(cmlp_dims, cmlp_act, add_bias=True,
                      dropout_keep=cmlp_dropout,
                      phase_train=phase_train, wd=wd, scope='ctrl_mlp')

        # Score MLP definition
        smlp = nn.mlp([crnn_dim, 1], [tf.sigmoid], wd=wd, scope='smlp')

        # Attention filters
        filters_y = [None] * timespan
        filters_x = [None] * timespan

        # Groundtruth bounding box, [B, T, 2]
        attn_ctr_gt, attn_delta_gt, attn_lg_var_gt, attn_box_gt, \
            attn_top_left_gt, attn_bot_right_gt, idx_map = \
            _get_gt_attn(y_gt, attn_size, padding_ratio=attn_box_padding_ratio)
        if use_gt_attn:
            attn_ctr = attn_ctr_gt
            attn_delta = attn_delta_gt
            attn_lg_var = attn_lg_var_gt
            attn_ctr = [tf.reshape(tmp, [-1, 2])
                        for tmp in tf.split(1, timespan, attn_ctr)]
            attn_delta = [tf.reshape(tmp, [-1, 2])
                          for tmp in tf.split(1, timespan, attn_delta)]
            attn_lg_var = [tf.reshape(tmp, [-1, 2])
                           for tmp in tf.split(1, timespan, attn_lg_var)]
            attn_lg_gamma = [tf.zeros(tf.pack([num_ex, 1, 1, 1]))
                             for tt in xrange(timespan)]
            attn_gamma = [None] * timespan
        else:
            attn_ctr = [None] * timespan
            attn_delta = [None] * timespan
            attn_lg_var = [None] * timespan
            attn_lg_gamma = [None] * timespan
            attn_gamma = [None] * timespan

        gtbox_top_left = [None] * timespan
        gtbox_bot_right = [None] * timespan
        attn_top_left = [None] * timespan
        attn_bot_right = [None] * timespan

        # Attention CNN definition
        acnn_filters = attn_cnn_filter_size
        acnn_nlayers = len(acnn_filters)
        acnn_channels = [inp_depth] + attn_cnn_depth
        acnn_pool = [2] * acnn_nlayers
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
        arnn_h = attn_size / acnn_subsample
        arnn_w = attn_size / acnn_subsample

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
        amlp_dropout = [1.0 - mlp_dropout_ratio] * num_attn_mlp_layers
        amlp = nn.mlp(amlp_dims, amlp_act, dropout_keep=amlp_dropout,
                      phase_train=phase_train, wd=wd, scope='attn_mlp')

        # Controller CNN [B, H, W, D] => [B, RH1, RW1, RD1]
        h_ccnn[tt] = ccnn(x)
        h_ccnn_last = h_ccnn[tt][-1]
        crnn_inp = tf.reshape(h_ccnn_last, [-1, crnn_inp_dim])

        model['gt_knob_prob_box'] = tf.constant(0.0)
        model['gt_knob_prob_segm'] = tf.constant(0.0)

        for tt in xrange(timespan):
            # Controller RNN [B, R1]
            crnn_state[tt], crnn_g_i[tt], crnn_g_f[tt], crnn_g_o[
                tt] = crnn_cell(crnn_inp, crnn_state[tt - 1])
            h_crnn[tt] = tf.slice(
                crnn_state[tt], [0, crnn_dim], [-1, crnn_dim])
            if not use_gt_attn:
                ctrl_out = cmlp(h_crnn[tt])[-1]
                _ctr = tf.slice(ctrl_out, [0, 0], [-1, 2])
                _lg_delta = tf.slice(ctrl_out, [0, 2], [-1, 2])
                attn_ctr[tt], attn_delta[tt] = _unnormalize_attn(
                    _ctr, _lg_delta, inp_height, inp_width, attn_size)
                attn_lg_var[tt] = tf.zeros(tf.pack([num_ex, 2]))
                # attn_lg_var[tt] = tf.slice(ctrl_out, [0, 4], [-1, 2])
                attn_lg_gamma[tt] = tf.slice(ctrl_out, [0, 6], [-1, 1])

            attn_gamma[tt] = tf.reshape(tf.exp(attn_gamma[tt]),
                                        [-1, 1, 1, 1])
            attn_top_left[tt], attn_bot_right[tt] = _get_attn_coord(
                attn_ctr[tt], attn_delta[tt], attn_size)

            # [B, H, A]
            filters_y[tt] = _get_attn_filter(
                attn_ctr[tt][:, 0], attn_delta[tt][:, 0],
                attn_lg_var[tt][:, 0], inp_height, attn_size)
            # [B, W, A]
            filters_x[tt] = _get_attn_filter(
                attn_ctr[tt][:, 1], attn_delta[tt][:, 1],
                attn_lg_var[tt][:, 1], inp_width, attn_size)

            # Attended patch [B, A, A, D]
            x_patch[tt] = attn_gamma[tt] * _extract_patch(
                x, filters_y[tt], filters_x[tt], inp_depth)

            # CNN [B, A, A, D] => [B, RH2, RW2, RD2]
            h_acnn[tt] = acnn(x_patch[tt])
            h_acnn_last[tt] = h_acnn[tt][-1]

            if use_attn_rnn:
                # RNN [B, T, R2]
                arnn_inp = tf.reshape(h_acnn_last[tt], [-1, arnn_inp_dim])
                arnn_state[tt], arnn_g_i[tt], arnn_g_f[tt], arnn_g_o[tt] = \
                    arnn_cell(arnn_inp, arnn_state[tt - 1])

        # Scoring network
        h_crnn_all = tf.concat(1, [tf.expand_dims(h, 1) for h in h_crnn])
        score_inp = tf.reshape(h_crnn_all, [-1, ctrl_rnn_hid_dim])
        s_out = tf.reshape(smlp(score_inp)[-1], [-1, timespan])
        model['s_out'] = s_out

        # Dense segmentation network [B, T, R] => [B, T, M]
        if use_attn_rnn:
            h_arnn = [tf.slice(arnn_state[tt], [0, arnn_dim], [-1, arnn_dim])
                      for tt in xrange(timespan)]
            h_arnn_all = tf.concat(
                1, [tf.expand_dims(h, 1) for h in h_arnn])
            h_arnn_all = tf.reshape(h_arnn_all, [-1, amlp_inp_dim])
            amlp_inp = h_arnn_all
        else:
            h_acnn_all = tf.concat(1, [tf.expand_dims(h, 1)
                                       for h in h_acnn_last])
            h_acnn_all = tf.reshape(h_acnn_all, [-1, amlp_inp_dim])
            amlp_inp = h_acnn_all

        h_core = amlp(amlp_inp)[-1]
        h_core = tf.reshape(h_core, [-1, arnn_h, arnn_w, attn_mlp_depth])

        # DCNN [B * T, RH, RW, MD] => [B * T, A, A, 1]
        dcnn_filters = dcnn_filter_size
        dcnn_nlayers = len(dcnn_filters)
        dcnn_unpool = [2] * (dcnn_nlayers - 1) + [1]
        dcnn_act = [tf.nn.relu] * (dcnn_nlayers - 1) + [None]
        dcnn_channels = [attn_mlp_depth] + dcnn_depth
        dcnn_use_bn = [use_bn] * dcnn_nlayers

        skip, skip_ch = _build_skip_conn_attn(
            acnn_channels, h_acnn, x_patch, timespan)

        dcnn = nn.dcnn(dcnn_filters, dcnn_channels, dcnn_unpool,
                       dcnn_act, use_bn=dcnn_use_bn, skip_ch=skip_ch,
                       phase_train=phase_train, wd=wd, model=model)
        h_dcnn = dcnn(h_core, skip=skip)

        # Concat all attentions.
        attn_top_left = tf.concat(1, [tf.expand_dims(tmp, 1)
                                      for tmp in attn_top_left])
        attn_bot_right = tf.concat(1, [tf.expand_dims(tmp, 1)
                                       for tmp in attn_bot_right])
        attn_ctr = tf.concat(1, [tf.expand_dims(tmp, 1)
                                 for tmp in attn_ctr])
        attn_delta = tf.concat(1, [tf.expand_dims(tmp, 1)
                                   for tmp in attn_delta])
        attn_gamma = tf.concat(1, [tf.expand_dims(tmp, 1)
                                   for tmp in attn_gamma])
        attn_gamma = tf.reshape(attn_gamma, [-1, 1, 1, 1])
        model['attn_ctr'] = attn_ctr
        model['attn_delta'] = attn_delta
        model['attn_top_left'] = attn_top_left
        model['attn_bot_right'] = attn_bot_right

        # Prob
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

        # Inverse attention [B, T, A, A, 1] => [B, T, H, W, 1]
        # Filters T * [B, L, A] => [B * T, L, A]
        filters_y_all = tf.reshape(
            tf.concat(1, [tf.expand_dims(f, 1) for f in filters_y]),
            [-1, inp_height, attn_size])
        filters_x_all = tf.reshape(
            tf.concat(1, [tf.expand_dims(f, 1) for f in filters_x]),
            [-1, inp_width, attn_size])
        filters_y_all_inv = tf.transpose(filters_y_all, [0, 2, 1])
        filters_x_all_inv = tf.transpose(filters_x_all, [0, 2, 1])
        y_out = _extract_patch(
            h_dcnn[-1] + 5.0, filters_y_all_inv, filters_x_all_inv, 1)
        y_out = 1.0 / attn_gamma * y_out
        # y_out_b = nn.weight_variable([1])
        y_out = tf.sigmoid(y_out - 5.0)
        # y_out = tf.sigmoid(y_out - tf.exp(y_out_b))
        y_out = tf.reshape(y_out, [-1, timespan, inp_height, inp_width])

        gamma = 10.0
        const_ones = tf.ones(
            tf.pack([num_ex * timespan, attn_size, attn_size, 1])) * gamma
        attn_box = _extract_patch(
            const_ones, filters_y_all_inv, filters_x_all_inv, 1)
        attn_box_b = 5.0
        attn_box = tf.sigmoid(attn_box - attn_box_b)
        attn_box = tf.reshape(attn_box, [-1, timespan, inp_height, inp_width])

        model['y_out'] = y_out
        model['attn_box'] = attn_box

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

        # Loss for attnention box
        iou_soft_box = _f_iou(attn_box, attn_box_gt, timespan, pairwise=True)
        model['attn_box_gt'] = attn_box_gt
        match_box = _segm_match(iou_soft_box, s_gt)
        model['match_box'] = match_box
        match_sum_box = tf.reduce_sum(match_box, reduction_indices=[2])
        match_count_box = tf.reduce_sum(
            match_sum_box, reduction_indices=[1])
        if box_loss_fn == 'iou':
            iou_soft_box = tf.reduce_sum(tf.reduce_sum(
                iou_soft_box * match_box, reduction_indices=[1, 2])
                / match_count_box) / num_ex
            box_loss = -iou_soft_box
        elif box_loss_fn == 'bce':
            box_loss = _match_bce(attn_box, attn_box_gt, match_box, timespan)
        elif box_loss_fn == 'mse':
            _s_gt = tf.reshape(s_gt, [-1, timespan, 1])
            _s_count = tf.reduce_sum(s_gt, reduction_indices=[1])
            _attn_top_left = tf.to_float(
                attn_top_left) * _s_gt / inp_height * 2 - 1
            _attn_top_left_gt = tf.to_float(
                attn_top_left_gt) * _s_gt / inp_height * 2 - 1
            _attn_bot_right = tf.to_float(
                attn_bot_right) * _s_gt / inp_height * 2 - 1
            _attn_bot_right_gt = tf.to_float(
                attn_bot_right_gt) * _s_gt / inp_height * 2 - 1
            diff1 = (_attn_top_left - _attn_top_left_gt)
            diff2 = (_attn_bot_right - _attn_bot_right_gt)
            box_loss = tf.reduce_sum(tf.reduce_sum(
                diff1 * diff1 + diff2 * diff2, reduction_indices=[1, 2]) /
                _s_count) / num_ex
        else:
            raise Exception('Unknown box_loss_fn: {}'.format(box_loss_fn))
        model['box_loss'] = box_loss

        box_loss_coeff = tf.constant(1.0)
        tf.add_to_collection('losses', box_loss_coeff * box_loss)

        # Loss for fine segmentation
        iou_soft = _f_iou(y_out, y_gt, timespan, pairwise=True)

        match = _segm_match(iou_soft, s_gt)
        model['match'] = match
        match_sum = tf.reduce_sum(match, reduction_indices=[2])
        match_count = tf.reduce_sum(match_sum, reduction_indices=[1])

        # match = match_box
        # model['match'] = match
        # match_sum = match_sum_box
        # match_count = match_count_box

        iou_soft = tf.reduce_sum(tf.reduce_sum(
            iou_soft * match, reduction_indices=[1, 2]) / match_count) / num_ex
        model['iou_soft'] = iou_soft
        if segm_loss_fn == 'iou':
            segm_loss = -iou_soft
        elif segm_loss_fn == 'bce':
            segm_loss = _match_bce(y_out, y_gt, match, timespan)
        else:
            raise Exception('Unknown segm_loss_fn: {}'.format(segm_loss_fn))
        model['segm_loss'] = segm_loss
        segm_loss_coeff = tf.constant(1.0)
        tf.add_to_collection('losses', segm_loss_coeff * segm_loss)

        # Score loss
        conf_loss = _conf_loss(s_out, match, timespan, use_cum_min=True)
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
        iou_hard = _f_iou(y_out_hard, y_gt, timespan, pairwise=True)
        iou_hard = tf.reduce_sum(tf.reduce_sum(
            iou_hard * match, reduction_indices=[1, 2]) / match_count) / num_ex
        model['iou_hard'] = iou_hard

        dice = _f_dice(y_out_hard, y_gt, timespan, pairwise=True)
        dice = tf.reduce_sum(tf.reduce_sum(
            dice * match, reduction_indices=[1, 2]) / match_count) / num_ex
        model['dice'] = dice

        model['count_acc'] = _count_acc(s_out, s_gt)
        model['dic'] = _dic(s_out, s_gt, abs=False)
        model['dic_abs'] = _dic(s_out, s_gt, abs=True)

    return model
