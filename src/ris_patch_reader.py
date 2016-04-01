import cslab_environ

import tensorflow as tf

from utils import logger
from utils.saver import Saver
import ris_patch_model as patch_model
import h5py

log = logger.get()


def read(folder):
    log.info('Reading pretrained patch segmentation network from {}'.format(folder))
    saver = Saver(folder)
    ckpt_info = saver.get_ckpt_info()
    model_opt = ckpt_info['model_opt']
    ckpt_fname = ckpt_info['ckpt_fname']
    model_id = ckpt_info['model_id']
    model = patch_model.get_model(model_opt)
    attn_cnn_nlayers = len(model_opt['attn_cnn_filter_size'])
    attn_mlp_nlayers = model_opt['attn_mlp_nlayers']
    attn_dcnn_nlayers = len(model_opt['attn_dcnn_filter_size'])
    weights = {}
    sess = tf.Session()
    saver.restore(sess, ckpt_fname)

    output_list = []
    for net, nlayers in zip(['attn_cnn', 'attn_mlp', 'attn_dcnn'],
                            [attn_cnn_nlayers, attn_mlp_nlayers,
                             attn_dcnn_nlayers]):
        for ii in xrange(attn_cnn_nlayers):
            w_name = '{}_w_{}'.format(net, ii)
            b_name = '{}_b_{}'.format(net, ii)
            output_list.append(w_name)
            output_list.append(b_name)

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

    pass

if __name__ == '__main__':
    save('/ais/gobi3/u/mren/results/img-count/fg_segm-20160318131644/weights.h5',
         '/ais/gobi3/u/mren/results/img-count/fg_segm-20160318131644')
