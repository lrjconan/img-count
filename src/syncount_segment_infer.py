from utils import logger
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import syncount_gen_data as data
import syncount_segment as model
import tensorflow as tf

log = logger.get()


def idx2img(idx, scale, patch_size, img_size):
    """
    From patch index to image patch.

    Args:
        idx: tuple of size 2, (y, x)
        scale:
        patch_size: tuple of size 2, (height, width)
    Returns:
        y_range:
        x_range:
        yidx_range:
        xidx_range:
    """
    if idx[0] * scale < patch_size[0] / 2:
        ystart = 0
        yidxstart = int(patch_size[0] / 2 - idx[0] * scale)
    else:
        ystart = int(idx[0] * scale - patch_size[0] / 2)
        yidxstart = 0
    if idx[0] * scale + patch_size[0] / 2 - img_size[0] > 0:
        yend = img_size[0]
        yidxend = int(img_size[0] - idx[0] * scale + patch_size[0] / 2)
    else:
        yend = int(idx[0] * scale + patch_size[0] / 2)
        yidxend = patch_size[0]
    if idx[1] * scale < patch_size[1] / 2:
        xstart = 0
        xidxstart = int(patch_size[1] / 2 - idx[1] * scale)
    else:
        xstart = int(idx[1] * scale - patch_size[1] / 2)
        xidxstart = 0
    if idx[1] * scale + patch_size[1] / 2 - img_size[1] > 0:
        xend = img_size[1]
        xidxend = int(img_size[1] - idx[1] * scale + patch_size[1] / 2)
    else:
        xend = int(idx[1] * scale + patch_size[1] / 2)
        xidxend = patch_size[1]

    return (ystart, yend), (xstart, xend), \
           (yidxstart, yidxend), (xidxstart, xidxend)

if __name__ == '__main__':
    ckpt_fname = '../results/syncount_segment-20160103222523/model.ckpt-19'
    opt_fname = '../results/syncount_segment-20160103222523/opt.pkl'

    with open(opt_fname, 'rb') as f_opt:
        opt = pkl.load(f_opt)
    train_model = model.get_train_model(opt)
    sess = tf.Session()
    saver = tf.train.Saver(tf.all_variables())
    saver.restore(sess, ckpt_fname)
    m = model.get_inference_model(opt, sess, train_model)

    # Try multiple scale of the original image here.
    opt['num_ex'] = 1
    opt['radius_lower'] = 5
    opt['radius_upper'] = 45
    raw_data = data.get_raw_data(opt, seed=10)
    image_data = data.get_image_data(opt, raw_data)
    orig_img = image_data['images'][0: 1]
    log.info('Original image: {}'.format(orig_img.shape))

    scale_list = [2.0]
    # scale_list = [0.5, 1.0, 1.5, 2.0]

    for scale in scale_list:
        log.info('Scale: {}'.format(scale))

        # Resample image.
        inp_height = int(np.round(orig_img.shape[1] * scale))
        inp_width = int(np.round(orig_img.shape[2] * scale))
        img = cv2.resize(orig_img[0], (inp_height, inp_width))
        img = img.reshape([-1, inp_height, inp_width, 3])
        log.info('Resampled image: {}'.format(img.shape))

        segm = sess.run(m['segm'], feed_dict={m['inp']: img})
        obj = sess.run(m['obj'], feed_dict={m['inp']: img})

        # Reshape the output
        out_size = opt['output_window_size']
        log.info('Segm shape: {}'.format(segm.shape))
        conv_out_h = int(np.ceil(inp_height / 8.0))
        conv_out_w = int(np.ceil(inp_width / 8.0))
        log.info('Reshape: {}'.format([-1, conv_out_h, conv_out_w, out_size, out_size]))
        segm = segm.reshape([-1, conv_out_h, conv_out_w, out_size, out_size])
        obj = obj.reshape([-1, conv_out_h, conv_out_w])

        out_height = segm.shape[1]
        out_width = segm.shape[2]
        log.info(segm.shape, verbose=2)

        # Plot original image.
        f1 = plt.figure()
        plt.imshow(img[0])

        # Plot segmentation output of every sliding window.
        # Very slow!
        # f2, axarr = plt.subplots(out_height, out_width)
        # for ii in xrange(out_height):
        #     for jj in xrange(out_width):
        #         axarr[ii, jj].imshow(segm[0, ii, jj, :, :])
        #         axarr[ii, jj].set_axis_off()
        # f2.suptitle('Segmentation output')

        # Plot objectness map.
        f3 = plt.figure()
        plt.imshow(obj[0])

        # Sort objectness
        obj_reshape = obj[0].reshape([-1])
        obj_srt_idx = np.argsort(obj_reshape)
        obj_srt_idx = obj_srt_idx[::-1]
        obj_srt_idx2 = np.array(
            [np.floor(obj_srt_idx / out_width), np.mod(obj_srt_idx, out_width)])
        obj_srt_idx2 = obj_srt_idx2.transpose()
        for ii in xrange(obj_reshape.size):
            log.info('1D: {:.2f}'.format(obj_reshape[obj_srt_idx[ii]]), 
                     verbose=2)

        log.info(obj_srt_idx.shape, verbose=2)
        log.info(obj_srt_idx, verbose=2)
        log.info(obj_srt_idx2.shape, verbose=2)
        log.info(obj_srt_idx2, verbose=2)

        # plt.imshow(img[0])
        conv_downsample = 8
        patch_height = segm.shape[3]
        patch_width = segm.shape[4]
        num_proposals = 300
        idx_start = 0
        num_col = num_proposals / 15
        num_row = num_proposals / num_col
        f4, axarr = plt.subplots(num_row, num_col)
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, 
                            wspace=0.0, hspace=0.0)

        for ii in xrange(idx_start, idx_start + num_proposals):
            # Plot top confident segmentation proposals
            # f4 = plt.figure()
            row = (ii - idx_start) / num_col
            col = (ii - idx_start) % num_col
            final_img = np.copy(img[0])
            # Deal with SAME/VALID convolution patching.
            # Need to convert to VALID here.
            mask = np.zeros(img[0].shape)
            yy = obj_srt_idx2[ii, 0]
            xx = obj_srt_idx2[ii, 1]
            log.info((yy, xx), verbose=2)
            segm_i = segm[0][yy, xx]

            y_range, x_range, yidx_range, xidx_range = idx2img(
                (yy, xx), conv_downsample, (patch_height, patch_width),
                (img[0].shape[0], img[0].shape[1]))
            log.info('Image y range: {}'.format(y_range), verbose=2)
            log.info('Image x range: {}'.format(x_range), verbose=2)
            log.info('Patch y range: {}'.format(yidx_range), verbose=2)
            log.info('Patch x range: {}'.format(xidx_range), verbose=2)
            s = ((segm_i > 0.9) * 255).astype('uint8')
            mask[y_range[0]: y_range[1], x_range[0]: x_range[1], 0] = s[
                yidx_range[0]: yidx_range[1], xidx_range[0]: xidx_range[1]]
            final_img += mask
            final_img[final_img > 255] = 255

            # plt.title('Objectness: {:.2f}'.format(obj[0, yy, xx]))
            # plt.imshow(final_img)
            axarr[row, col].set_title('{:.2f}'.format(obj[0, yy, xx]))
            axarr[row, col].imshow(final_img)
            axarr[row, col].set_axis_off()

        f4.suptitle('Segmentation output at scale {}'.format(scale))

        plt.show()
