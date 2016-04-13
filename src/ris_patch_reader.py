import cslab_environ

import argparse
import tensorflow as tf

from utils import logger
from utils.saver import Saver
import ris_patch_model as patch_model
import h5py
import sys

log = logger.get()


def read(folder):
    log.info('Reading pretrained patch segmentation network from {}'.format(folder))
    saver = Saver(folder)
    ckpt_info = saver.get_ckpt_info()
    model_opt = ckpt_info['model_opt']
    if 'filter_height' not in model_opt:
        model_opt['filter_height'] = model_opt['filter_size']
        model_opt['filter_width'] = model_opt['filter_size']

    ckpt_fname = ckpt_info['ckpt_fname']
    model_id = ckpt_info['model_id']
    model = patch_model.get_model(model_opt)
    attn_cnn_nlayers = len(model_opt['attn_cnn_filter_size'])
    attn_mlp_nlayers = model_opt['num_attn_mlp_layers']
    attn_dcnn_nlayers = len(model_opt['attn_dcnn_filter_size'])
    timespan = model_opt['timespan']
    weights = {}
    sess = tf.Session()
    saver.restore(sess, ckpt_fname)

    output_list = []
    for net, nlayers in zip(['attn_cnn', 'attn_mlp', 'attn_dcnn'],
                            [attn_cnn_nlayers, attn_mlp_nlayers,
                             attn_dcnn_nlayers]):
        for ii in xrange(nlayers):
            for w in ['w', 'b']:
                key = '{}_{}_{}'.format(net, w, ii)
                log.info(key)
                output_list.append(key)
            if net == 'attn_cnn' or net == 'attn_dcnn':
                for tt in xrange(timespan):
                    for w in ['beta', 'gamma']:
                        key = '{}_{}_{}_{}'.format(net, ii, tt, w)
                        if key in model:
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


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Recurrent Instance Segmentation + Attention')

    parser.add_argument('--model_id', default=None)
    parser.add_argument(
        '--results', default='/ais/gobi3/u/mren/results/img-count')
    parser.add_argument(
        '--output', default=None)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    exp_folder = os.path.join(args.results, args.model_id)
    if args.output is None:
        output = os.path.join(exp_folder, 'weights.h5')
    else:
        output = args.output
    save(output, exp_folder)
