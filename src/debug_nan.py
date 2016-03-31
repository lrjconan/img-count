import cslab_environ

from data_api import cvppp
from utils.saver import Saver
from utils import logger
import numpy as np
import ris_double_attn_model as double_attention
import sys
import tensorflow as tf


log = logger.get()

if __name__ == '__main__':
    # model_folder = sys.argv[1]
    # model_folder = '/ais/gobi3/u/mren/results/img-count/rec_ins_segm-20160329234848'
    model_folder = '../results/rec_ins_segm-20160330212049'
    # model_folder = '../results/rec_ins_segm-20160329195154'

    saver = Saver(model_folder)
    ckpt_info = saver.get_ckpt_info()

    # ckpt_fname = ckpt_info['ckpt_fname']
    ckpt_id = 8000
    ckpt_fname = '{}/model.ckpt-{}'.format(model_folder, ckpt_id)
    model_opt = ckpt_info['model_opt']
    if 'squash_ctrl_params' not in model_opt:
        model_opt['squash_ctrl_params'] = False

    model = double_attention.get_model(model_opt)

    sess = tf.Session()
    saver.restore(sess, ckpt_fname)

    model_val = {}
    output_list = []

    for cnn in ['ctrl_cnn', 'attn_cnn', 'attn_dcnn']:
        # for cnn in ['ctrl_cnn']:
        for ii in xrange(len(model_opt['{}_filter_size'.format(cnn)])):
            for w in ['w', 'b']:
                key = '{}_{}_{}'.format(cnn, w, ii)
                log.info(key)
                output_list.append(key)

    for lstm in ['ctrl_lstm']:
        for comp in ['i', 'f', 'i', 'o']:
            for w in ['w_x', 'w_h', 'b_']:
                key = '{}_{}{}'.format(lstm, w, comp)
                log.info(key)
                output_list.append(key)

    for mlp, nlayers in zip(['ctrl_mlp', 'glimpse_mlp', 'score_mlp', 'attn_mlp'],
                            [model_opt['num_ctrl_mlp_layers'],
                             model_opt['num_glimpse_mlp_layers'],
                             1,
                             model_opt['num_attn_mlp_layers']]):
        for ii in xrange(nlayers):
            for w in ['w', 'b']:
                key = '{}_{}_{}'.format(mlp, w, ii)
                log.info(key)
                output_list.append(key)

    output_var = []
    for key in output_list:
        output_var.append(model[key])

    output_val = sess.run(output_var)
    for kk, key in enumerate(output_list):
        model_val[key] = output_val[kk]

    for key in model_val.iterkeys():
        val = model_val[key]
        log.info('Is NaN   {:15s} {}'.format(
            key, val.size == np.isnan(val).sum()))

    for key in model_val.iterkeys():
        val = model_val[key]
        log.info('{:6s} {:15s} {:10.4f} {:10.4f}'.format(
            'Mean', key, val.mean(), np.abs(val).mean()))
        log.info('{:6s} {:15s} {:10.4f} {:10.4f}'.format(
            'Max', key, val.max(), np.abs(val).max()))
        log.info('{:6s} {:15s} {:10.4f} {:10.4f}'.format(
            'Min', key, val.min(), np.abs(val).min()))
