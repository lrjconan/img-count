import cslab_environ

import tensorflow as tf

from utils import logger
from utils.saver import Saver
import ris_box_model as box_model
import h5py
import sys

log = logger.get()


def read(folder):
    log.info('Reading pretrained box network from {}'.format(folder))
    saver = Saver(folder)
    ckpt_info = saver.get_ckpt_info()
    model_opt = ckpt_info['model_opt']
    if 'filter_height' not in model_opt:
        model_opt['filter_height'] = model_opt['filter_size']
        model_opt['filter_width'] = model_opt['filter_size']

    ckpt_fname = ckpt_info['ckpt_fname']
    model_id = ckpt_info['model_id']
    model = patch_model.get_model(model_opt)
    ctrl_cnn_nlayers = len(model_opt['attn_cnn_filter_size'])
    ctrl_mlp_nlayers = model_opt['num_attn_mlp_layers']
    weights = {}
    sess = tf.Session()
    saver.restore(sess, ckpt_fname)

    output_list = []
    for net, nlayers in zip(['ctrl_cnn', 'ctrl_mlp'],
                            [attn_cnn_nlayers, attn_mlp_nlayers,
                             attn_dcnn_nlayers]):
        for ii in xrange(nlayers):
            for w in ['w', 'b']:
                key = '{}_{}_{}'.format(net, w, ii)
                log.info(key)
                output_list.append(key)

    for net in ['ctrl_lstm']:
        for w in ['w_xi', 'w_hi', 'b_i', 'w_xf', 'w_hf', 'b_f', 'w_xu',
                  'w_hu', 'b_u', 'w_xo', 'w_ho', 'w_bo']:
            key = '{}_{}'.format(net, w)
            log.info(key)
            output_list.append(key)

    output_var = []
    for key in output_list:
        output_var.append(model[key])

    output_var_value = sess.run(output_var)

    for key, value in zip(output_list, output_var_value):
        weights[key] = value
        log.info(key)
        log.info(value.shape)

    return weights


def save(fname, folder):
    weights = read(folder)
    h5f = h5py.File(fname, 'w')
    for key in weights:
        h5f[key] = weights[key]
    h5f.close()
    log.info('Saved weights to {}'.format(fname))

    pass

if __name__ == '__main__':
    model_id = sys.argv[1]
    save('/ais/gobi3/u/mren/results/img-count/{}/weights.h5'.format(model_id),
         '/ais/gobi3/u/mren/results/img-count/{}'.format(model_id))
