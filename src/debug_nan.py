import cslab_environ

from data_api import cvppp
from utils.saver import Saver
import ris_double_attn_model as double_attention
import sys
import tensorflow as tf


if name == '__main__':
    model_folder = sys.argv[1]
    # model_folder = '../results/rec_ins_segm-20160329195154'

    saver = Saver(model_folder)
    ckpt_info = saver.get_ckpt_info()
    ckpt_fname = ckpt_info['ckpt_fname']
    model_opt = ckpt_info['model_opt']
    model_opt['squash_ctrl_params'] = False

    model = double_attention.get_model(model_opt)

    sess = tf.Session()
    saver.restore(sess, ckpt_fname)

    model_val = {}
    output_list = []

    # for cnn in ['ctrl_cnn', 'attn_cnn', 'attn_dcnn']:
    for cnn in ['ctrl_cnn']:
        for ii in xrange(len(model_opt['ctrl_cnn_filter_size'])):
            print cnn, ii
            for w in ['w', 'b']:
                key = '{}_{}_{}'.format(cnn, w, ii)
                output_list.append(model[key])

    output_val = sess.run(output_list)
    for kk, key in enumerate(output_list):
        model_val[key] = output_val[kk]
        

