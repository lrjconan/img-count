import cslab_environ

from utils import logger
from utils.batch_iter import BatchIterator

import argparse
import numpy as np
import tensorflow as tf

log = logger.get()


def get_data(num_ex, timespan):
    """Get parity toy data."""
    inp = np.zeros((num_ex, timespan, 1), dtype='float32')
    label = np.zeros((num_ex, timespan, 1), dtype='float32')
    random = np.random.RandomState(2)
    for ii in xrange(num_ex):
        parity = 0
        for t in xrange(timespan):
            inp[ii, t, 0] = np.round(random.uniform(0, 1))
            parity += inp[ii, t, 0]
            label[ii, t, 0] = parity % 2
        if ii == 0:
            log.info(inp[ii])
            log.info(label[ii])

    return inp, label


def weight_variable(shape):
    """Initialize weights."""
    initial = tf.truncated_normal(shape, stddev=0.001)

    return tf.Variable(initial)


def get_model(opt, device='/cpu:0'):
    """Simple gated recurrent unit."""
    timespan = opt['timespan']
    inp_dim = opt['inp_dim']
    mem_dim = opt['mem_dim']
    out_dim = opt['out_dim']
    inp = tf.placeholder('float', [None, timespan, 1])
    inp_list = tf.split(1, timespan, inp)

    with tf.device(device):
        # Input gate weights
        # Input to input gate
        w_xi = weight_variable([inp_dim, mem_dim])
        # State to input gate
        w_si = weight_variable([mem_dim, mem_dim])
        b_i = tf.Variable(tf.zeros([mem_dim]))  # Initialized to 0

        # Recurrent gate weights
        # Input to recurrent gate
        w_xr = weight_variable([inp_dim, mem_dim])
        # State to recurrent gate
        w_sr = weight_variable([mem_dim, mem_dim])
        b_r = tf.Variable(tf.ones([mem_dim]))  # Initialized to 1

        # Transformation weights
        # Input transformation
        w_xz = weight_variable([inp_dim, mem_dim])
        # State transformation
        w_sz = weight_variable([mem_dim, mem_dim])
        b_z = tf.Variable(tf.zeros([mem_dim]))  # Initialized to 0

        # Output transformation
        w_o = weight_variable([mem_dim, out_dim])
        b_o = tf.Variable(tf.zeros([out_dim]))

        # State index shifted by 1.
        # s[0] is the initial state
        # The t th state is state[t + 1]
        # init_state = tf.constant(np.zeros(mem_dim, dtype='float32'))
        init_state_w = tf.constant(
            np.zeros((inp_dim, mem_dim), dtype='float32'))
        init_state = tf.matmul(tf.reshape(
            inp_list[0], [-1, inp_dim]), init_state_w)

        # Activation (hidden state)
        s = [None] * (timespan + 1)
        s[0] = init_state
        x = [None] * timespan
        # Input gate
        g_i = [None] * timespan
        # Recurrent gate
        g_r = [None] * timespan
        # Candidate activation
        z = [None] * timespan
        # Output
        y_list = [None] * timespan
        label = tf.placeholder('float', [None, timespan, 1])
        label_flat = tf.reshape(label, [-1, timespan])
        for t in xrange(timespan):
            x[t] = tf.reshape(inp_list[t], [-1, inp_dim])
            g_i[t] = tf.sigmoid(tf.matmul(x[t], w_xi) +
                                tf.matmul(s[t], w_si) + b_i)
            g_r[t] = tf.sigmoid(tf.matmul(x[t], w_xr) +
                                tf.matmul(s[t], w_sr) + b_r)
            z[t] = tf.tanh(tf.matmul(x[t], w_xz) + g_r[t]
                           * tf.matmul(s[t], w_sz) + b_z)
            s[t + 1] = s[t] * (1 - g_i[t]) + z[t] * g_i[t]
            y_list[t] = tf.sigmoid(tf.matmul(s[t + 1], w_o) + b_o)

        y = tf.concat(1, y_list)
        eps = 1e-7
        num_ex = tf.to_float(tf.shape(x[0])[0])
        ce = -tf.reduce_sum(label_flat * tf.log(y + eps) +
                            (1 - label_flat) * tf.log(1 - y + eps)) / num_ex
        pred = tf.round(y)
        correct = tf.equal(pred, label_flat)
        num_out = tf.to_float(tf.size(y))
        acc = tf.reduce_sum(tf.to_float(correct)) / num_out

        lr = 1e-2
        train_step = tf.train.AdamOptimizer(lr, epsilon=eps).minimize(ce)

    model = {
        'inp': inp,
        'inp_list': inp_list,
        'w_xi': w_xi,
        'w_si': w_si,
        'b_si': b_i,
        'w_xr': w_xr,
        'w_sr': w_sr,
        'b_sr': b_r,
        'g_i': g_i,
        'g_r': g_r,
        'w_xz': w_xz,
        'w_sz': w_sz,
        'b_z': b_z,
        'z': z,
        's': s,
        'w_o': w_o,
        'b_o': b_o,
        'y_list': y_list,
        'y': y,
        'pred': pred,
        'label': label,
        'correct': correct,
        'acc': acc,
        'ce': ce,
        'train_step': train_step
    }

    return model


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Train DRAW')
    parser.add_argument('-gpu', default=-1, type=int,
                        help='GPU ID, default CPU')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # Command-line arguments
    args = parse_args()
    log.log_args()

    # Set device
    if args.gpu >= 0:
        device = '/gpu:{}'.format(args.gpu)
    else:
        device = '/cpu:0'
    # Simple parity example
    opt = {
        'timespan': 10,
        'inp_dim': 1,
        'mem_dim': 10,
        'out_dim': 1
    }

    num_ex = 1000
    inp, label = get_data(num_ex, opt['timespan'])
    log.info(inp.shape)
    log.info(label.shape)

    # Initialize session
    m = get_model(opt)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    step = 0
    for epoch in xrange(1000):
        if epoch % 10 == 0:
            ib = inp[0: 1]
            lb = label[0: 1]
            log.info('Input')
            log.info(''.join(['{:.4f} '.format(ib[0, t, 0])
                              for t in xrange(opt['timespan'])]))
            log.info('Label')
            log.info(''.join(['{:.4f} '.format(lb[0, t, 0])
                              for t in xrange(opt['timespan'])]))
            y = sess.run(m['y'], feed_dict={
                m['inp']: ib, m['label']: lb})
            log.info('Output')
            log.info(''.join(['{:.4f} '.format(y[0, t])
                              for t in xrange(opt['timespan'])]))

        for start, end in BatchIterator(num_ex, batch_size=32):
            ib = inp[start: end]
            lb = label[start: end]
            r = sess.run([m['train_step'], m['ce'], m['acc']],
                         feed_dict={m['inp']: ib, m['label']: lb})
            ce = r[1]
            acc = r[2]
            log.info('Step: {}, CE: {:.4f}, Acc: {:.4f}'.format(step, ce, acc))
            step += 1
