import tensorflow as tf
import numpy as np


def weight_variable(shape, wd=None, name=None):
    """Initialize weights."""
    initial = tf.truncated_normal(shape, stddev=0.01)
    var = tf.Variable(initial, name=name)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def get_train_model(opt):
    timespan = opt['timespan']
    inp_height = opt['inp_height']
    inp_width = opt['inp_width']
    num_hid_dec = opt['num_hid_dec']
    wd = opt['weight_decay']
    filter_size = opt['filter_size']

    x = tf.placeholder('float', [None, timespan, inp_width, inp_height])
    x_l = tf.split(1, timespan, x)
    x_err = [None] * timespan
    read_out = [None] * timespan
    h_enc = [None] * timespan
    z = [None] * timespan
    h_dec = [None] * timespan
    canvas = [None] * (timespan + 1)
    x_rec = [None] * timespan
    ctl = [None] * timespan
    ctr_x = [None] * timespan
    ctr_y = [None] * timespan
    delta = [None] * timespan
    mu_x = [None] * timespan
    mu_y = [None] * timespan
    lg_var = [None] * timespan
    lg_gamma = [None] * timespan
    filter_x = [None] * timespan
    filter_y = [None] * timespan
    w = [None] * timespan
    write_out = [None] * timespan

    # Weights for read/write head.
    # From decoder hidden state to controller variables.
    w_hdec_ctl = weight_variables(
        [num_hid_dec, 5], wd=wd, name='w_hdec_ctl')
    b_hdec_ctl = weight_variables(
        [5], wd=wd, name='b_hdec_ctl')

    # Weights from decoder to write
    w_hdec_w = weight_variable(
        [num_hid_dec, filter_size * filter_size], wd=wd, name='w_hdec_w')
    b_hdec_w = weight_variable(
        [filter_size * filter_size], wd=wd, name='b_hdec_w')

    span_x = np.tile(np.arange(inp_width), [1, filter_size])
    span_y = np.tile(np.arange(inp_height), [1, filter_size])

    for t in xrange(timespan):
        # Read head
        # (g_x, g_y, log_var, log_delta, log_gamma)
        ctl[t] = tf.matmul(h_dec[t], w_hdec_ctl)
        ctr_x[t] = (inp_width + 1) / 2 * (ctl[t][0] + 1)
        ctr_y[t] = (inp_height + 1) / 2 * (ctl[t][1] + 1)
        delta[t] = ((np.max(inp_width, inp_height) - 1) /
                    (filter_read_size - 1) * tf.exp(ctl[3]))
        lg_var[t] = ctl[t][2]
        lg_gamma[t] = ctl[t][4]

        mu_x = tf.tile(ctr_x + delta[t] *
                       (np.arange(filter_read_size) -
                        filter_read_size / 2 - 0.5), [1, inp_width])
        mu_y = tf.tile(ctr_y + delta[t] *
                       (np.arange(filter_read_size) -
                        filter_read_size / 2 - 0.5), [1, inp_height])

        filter_x = (1 / tf.sqrt(tf.exp(lg_var[t])) / tf.sqrt(2 * np.pi) *
                    tf.exp(-0.5 * (span_x - mu_x) * (span_x - mu_x) /
                           tf.exp(lg_var[t])))
        filter_y = (1 / tf.sqrt(tf.exp(lg_var[t])) / tf.sqrt(2 * np.pi) *
                    tf.exp(-0.5 * (span_y - mu_y) * (span_y - mu_y) /
                           tf.exp(lg_var[t])))
        read_out[t] = tf.mul(tf.exp(lg_gamma), tf.matmul(
            tf.matmul(filter_x, x_l[t]), tf.transpose(filter_y)))

        ######
        # Batch dimension mismatch with filter!!!
        ######
        w[t] = tf.reshape(tf.matmul(h_dec[t], w_hdec_w) +
                          b_hdec_w, [-1, filter_size, filter_size])
        write_out[t] = tf.mul(1 / tf.exp(lg_gamma), tf.matmul(
            tf.matmul(tf.transpose(filter_y), w[t]), filter_x))

        pass

    m = {}
    return m

if __name__ == '__main__':
    pass
