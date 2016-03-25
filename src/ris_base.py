import cslab_environ

from tensorflow.python.framework import ops
import numpy as np
import tensorflow as tf

# Register gradient for Hungarian algorithm.
ops.NoGradient("Hungarian")


def get_device_fn(device):
    """Choose device for different ops."""
    OPS_ON_CPU = set(['ResizeBilinear', 'ResizeBilinearGrad', 'Mod', 'CumMin',
                      'CumMinGrad', 'Hungarian', 'Reverse', 'SparseToDense',
                      'BatchMatMul'])

    def _device_fn(op):
        if op.type in OPS_ON_CPU:
            return "/cpu:0"
        else:
            # Other ops will be placed on GPU if available, otherwise CPU.
            return device

    return _device_fn


def cum_min(s, d):
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


def f_dice(a, b, timespan, pairwise=False):
    """
    Computes DICE score.

    Args:
        a: [B, N, H, W], or [N, H, W], or [H, W]
        b: [B, N, H, W], or [N, H, W], or [H, W]
           in pairwise mode, the second dimension can be different,
           e.g. [B, M, H, W], or [M, H, W], or [H, W]
        pairwise: whether the inputs are already aligned, outputs [B, N] or
                  the inputs are orderless, outputs [B, N, M].
    """

    if pairwise:
        # N * [B, 1, M]
        y_list = [None] * timespan
        # [B, N, H, W] => [B, N, 1, H, W]
        a = tf.expand_dims(a, 2)
        # [B, N, 1, H, W] => N * [B, 1, 1, H, W]
        a_list = tf.split(1, timespan, a)
        # [B, M, H, W] => [B, 1, M, H, W]
        b = tf.expand_dims(b, 1)
        card_b = tf.reduce_sum(b + 1e-5, [3, 4])

        for ii in xrange(timespan):
            # [B, 1, M]
            y_list[ii] = 2 * f_inter(a_list[ii], b) / \
                (tf.reduce_sum(a_list[ii] + 1e-5, [3, 4]) + card_b)

        # N * [B, 1, M] => [B, N, M]
        return tf.concat(1, y_list)
    else:
        card_a = tf.reduce_sum(a + 1e-5, _get_reduction_indices(a))
        card_b = tf.reduce_sum(b + 1e-5, _get_reduction_indices(b))
        return 2 * f_inter(a, b) / (card_a + card_b)


def f_inter(a, b):
    """Computes intersection."""
    reduction_indices = _get_reduction_indices(a)
    return tf.reduce_sum(a * b, reduction_indices=reduction_indices)


def f_union(a, b, eps=1e-5):
    """Computes union."""
    reduction_indices = _get_reduction_indices(a)
    return tf.reduce_sum(a + b - (a * b) + eps,
                         reduction_indices=reduction_indices)


def _get_reduction_indices(a):
    """Gets the list of axes to sum over."""
    dim = len(a.get_shape())

    return [dim - 2, dim - 1]


def f_iou(a, b, timespan=None, pairwise=False):
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

    if pairwise:
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
            y_list[ii] = f_inter(a_list[ii], b) / f_union(a_list[ii], b)

        # N * [B, 1, M] => [B, N, M]
        return tf.concat(1, y_list)
    else:
        return f_inter(a, b) / f_union(a, b)


def f_coverage(iou):
    """
    Coverage function proposed in [1]

    [1] N. Silberman, D. Sontag, R. Fergus. Instance segmentation of indoor
    scenes using a coverage loss. ECCV 2015.

    Args:
        iou: [B, N, N]. Pairwise IoU.
    """
    return tf.reduce_max(iou, [1])


def coverage_weight(y_gt):
    """
    Compute the normalized weight for each groundtruth instance.
    """
    # [B, T]
    y_gt_sum = tf.reduce_sum(y_gt, [2, 3])
    # Plus one to avoid dividing by zero.
    # The resulting weight will be zero for any zero cardinality instance.
    # [B, 1]
    y_gt_sum_sum = tf.reduce_sum(
        y_gt_sum, [1], keep_dims=True) + tf.to_float(tf.equal(y_gt_sum, 0))

    # [B, T]
    return y_gt_sum / y_gt_sum_sum


def f_weighted_coverage(iou, y_gt):
    """
    Weighted coverage score.

    Args:
        iou: [B, N, N]. Pairwise IoU.
        y_gt: [B, N, H, W]. Groundtruth segmentations.
    """
    cov = f_coverage(iou)
    wt = coverage_weight(y_gt)
    num_ex = tf.to_float(tf.shape(y_gt)[0])

    return tf.reduce_sum(cov * wt) / num_ex


def f_unweighted_coverage(iou, count):
    """
    Unweighted coverage score.

    Args:
        iou: [B, N, N]. Pairwise IoU.
    """
    # [B, N]
    cov = f_coverage(iou)
    num_ex = tf.to_float(tf.shape(iou)[0])
    return tf.reduce_sum(tf.reduce_sum(cov, [1]) / count) / num_ex


def f_conf_loss(s_out, match, timespan, use_cum_min=True):
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
        s_out_min = cum_min(s_out, timespan)
        # [B, N]
        s_bce = f_bce(s_out_min, match_sum)
    else:
        s_bce = f_bce(s_out, match_sum)
    loss = tf.reduce_sum(s_bce) / num_ex / max_num_obj

    return loss


def f_greedy_match(score, matched):
    """Compute greedy matching given the IOU, and matched.

    Args:
        score: [B, N] relatedness score, positive.
        matched: [B, N] binary mask

    Returns:
        match: [B, N] binary mask
    """
    score = score * (1.0 - matched)
    max_score = tf.reshape(tf.reduce_max(
        score, reduction_indices=[1]), [-1, 1])
    match = tf.to_float(tf.equal(score, max_score))
    match_sum = tf.reshape(tf.reduce_sum(
        match, reduction_indices=[1]), [-1, 1])

    return match / match_sum


def f_segm_match(iou, s_gt):
    """Matching between segmentation output and groundtruth.

    Args:
        y_out: [B, T, H, W], output segmentations
        y_gt: [B, T, H, W], groundtruth segmentations
        s_gt: [B, T], groudtruth score sequence
    """
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


def f_bce(y_out, y_gt):
    """Binary cross entropy."""
    eps = 1e-5
    return -y_gt * tf.log(y_out + eps) - (1 - y_gt) * tf.log(1 - y_out + eps)


def f_match_bce(y_out, y_gt, match, timespan):
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
            f_bce(y_out_list[ii], y_gt), reduction_indices=[2, 3]) *
            tf.reshape(match_list[ii], [-1, timespan]),
            reduction_indices=[1]), 1)

    # N * [B, 1] => [B, N] => [B]
    bce_total = tf.reduce_sum(tf.concat(1, bce_list), reduction_indices=[1])

    return tf.reduce_sum(bce_total / match_count) / num_ex / height / width


def f_count_acc(s_out, s_gt):
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


def f_dic(s_out, s_gt, abs=False):
    """Difference in count.

    Args:
        s_out:
        s_gt:
    """
    num_ex = tf.to_float(tf.shape(s_out)[0])
    count_out = tf.reduce_sum(tf.to_float(s_out > 0.5), reduction_indices=[1])
    count_gt = tf.reduce_sum(s_gt, reduction_indices=[1])
    count_diff = count_out - count_gt
    if abs:
        count_diff = tf.abs(count_diff)
    count_diff = tf.reduce_sum(tf.to_float(count_diff)) / num_ex

    return count_diff


def build_skip_conn_inner(cnn_channels, h_cnn, x):
    """Build skip connection."""
    skip = [None]
    skip_ch = [0]
    for jj, layer in enumerate(h_cnn[-2::-1] + [x]):
        skip.append(layer_reshape)
        ch_idx = len(cnn_channels) - jj - 2
        skip_ch.append(cnn_channels[ch_idx])

    return skip, skip_ch


def build_skip_conn(cnn_channels, h_cnn, x, timespan):
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


def build_skip_conn_attn(cnn_channels, h_cnn_time, x_time, timespan):
    """Build skip connection for attention based model."""
    skip = [None]
    skip_ch = [0]

    nlayers = len(h_cnn_time[0])
    timespan = len(h_cnn_time)

    for jj in xrange(nlayers):
        lidx = nlayers - jj - 2
        if lidx >= 0:
            ll = [h_cnn_time[tt][lidx] for tt in xrange(timespan)]
        else:
            ll = x_time
        layer = tf.concat(1, [tf.expand_dims(l, 1) for l in ll])
        ss = tf.shape(layer)
        layer = tf.reshape(layer, tf.pack([-1, ss[2], ss[3], ss[4]]))
        skip.append(layer)
        ch_idx = lidx + 1
        skip_ch.append(cnn_channels[ch_idx])

    return skip, skip_ch


def get_attn_filter(center, delta, lg_var, image_size, filter_size):
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


def extract_patch(x, f_y, f_x, nchannels):
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


def get_gt_attn(y_gt, attn_size, padding_ratio=0.0, center_shift_ratio=0.0):
    """Get groundtruth attention box given segmentation."""
    s = tf.shape(y_gt)
    # [B, T, H, W, 2]
    idx = get_idx_map(s)
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
        box = get_filled_box_idx(idx, top_left, bot_right)
    else:
        log.warning('Not padding groundtruth box')

    ctr = (bot_right + top_left) / 2.0
    delta = (bot_right - top_left + 1.0) / attn_size
    lg_var = tf.zeros(tf.shape(ctr)) + 1.0

    return ctr, delta, lg_var, box, top_left, bot_right, idx


def get_idx_map(shape):
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


def get_filled_box_idx(idx, top_left, bot_right):
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


def get_unnormalize_attn(ctr, lg_delta, inp_height, inp_width, attn_size):
    """Unnormalize the attention parameters to image size."""
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


def get_attn_coord(ctr, delta, attn_size):
    """Get attention coordinates given parameters."""
    a = ctr * 2.0
    b = delta * attn_size - 1.0
    top_left = (a - b) / 2.0
    bot_right = (a + b) / 2.0 + 1.0

    return top_left, bot_right
