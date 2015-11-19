"""
Interface to evaluate Fast R-CNN network.

Command-line usage:
    python fast_rcnn_api.py -image {image file}
                            -image_list {image list file}
                            -dataset {dataset name}
                            -datadir {dataset directory}
                            -net {network definition path}
                            -weights {weights path}
                            -gpu {gpu ID}
                            -cpu {cpu mode}
                            -conf {confidence threshold}
                            -nms {NMS threshold}
                            -proposal {proposal file}
                            -output {output file}
"""
from data_api import Mscoco
from fast_rcnn.test import im_detect
from fast_rcnn_model import FastRCNNModel
from fast_rcnn_utils.nms import nms
from fast_rcnn_utils.timer import Timer
from utils import logger
from utils import list_reader
import argparse
import caffe
import cv2
import fast_rcnn_init
import gc
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sio
import scipy.sparse
import selective_search
import sys

log = logger.get('../log/fast_rcnn_api')

NETS = {'vgg16': ('VGG16.prototxt',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024/test.prototxt',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet/test.prototxt',
                     'caffenet_fast_rcnn_iter_40000.caffemodel'),
        'coco_vgg16': ('VGG16/coco/test.prototxt',
                       'coco_vgg16_fast_rcnn_iter_240000.caffemodel')}


class FastRcnnApi():
    """
    Interface to run inference on trained Fast-RCNN network.
    Properties:
        model:
        dataset:
    """

    def __init__(self, model_def_fname, weights_fname,
                 dataset_name, dataset_dir, gpu=True, gpu_id=0):
        if args.dataset == 'mscoco':
            self.dataset = Mscoco(base_dir=dataset_dir, set_name='valid')
        else:
            raise Exception('Unknown dataset: {}.'.format(dataset))

        if not os.path.isfile(model_def_fname):
            raise IOError('Model definition file not found: {}'.format(
                weights_fname))
        if not os.path.isfile(weights_fname):
            raise IOError('Weights file not found: {}'.format(weights_fname))

        self.model = FastRCNNModel(name=model_def_fname,
                                   model_def_fname=model_def_fname,
                                   weights_fname=weights_fname,
                                   gpu=gpu,
                                   gpu_id=gpu_id)
        pass

    def run_detector_with_proposal(self, image_fname, get_local_feat=False,
                                   feat_layer='fc7'):
        """Run Fast-RCNN on a single image, without pre-computed proposals.
        """
        obj_proposals = self.compute_obj_proposals(image_fname)
        obj_proposals = self.shift_obj_proposals(obj_proposals)
        im = cv2.imread(image_fname)
        return self.run_detector(im, obj_proposals,
                                 get_local_feat=get_local_feat,
                                 feat_layer=feat_layer)

    def run_detector(self, im, obj_proposals, get_local_feat=False,
                     feat_layer='fc7'):
        """Run Fast-RCNN on a single image, with pre-computed proposals.
        
        Args:
            im: numpy.ndarray, image pixels.
            obj_proposals: numpy.ndarray, 
        Returns:
            results: numpy.ndarray, N x 5, N is the number of proposals.
            5 dimensions are (x1, y1, x2, y2, confidence).
            local_feat: (optional) numpy.ndarray, N x D, N is the number of i
            proposals, D is the dimension of the feature layer.
        """
        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()

        # K is the number of classes
        # N is the number of proposals
        # scores = (N, K)
        # boxes = (N, K x 4)
        scores, boxes, inv_index = \
            im_detect(self.model.net, im, obj_proposals)
        timer.toc()
        log.info(('Detection took {:.3f}s for '
                  '{:d} object proposals').format(
                      timer.total_time, boxes.shape[0]))
        boxes = boxes.reshape(boxes.shape[0], boxes.shape[1] / 4, 4)
        scores = scores.reshape(scores.shape[0], scores.shape[1], 1)
        results = np.concatenate((boxes, scores), axis=-1)

        if get_local_feat:
            local_feat = self.model.net.blobs[feat_layer].data
            # Average-pooling on convolutional layers.
            if feat_layer == 'pool5':
                local_feat = local_feat.mean(axis=-1).mean(axis=-1)
            local_feat = local_feat[inv_index]
            return results, local_feat
        else:
            return results

    def threshold(self, det_boxes, conf_thresh=0.8, nms_thresh=0.3):
        """Run thresholding on the boxes
        
        Args:
            det_boxes: numpy.ndarray, N x K x 5, N is the number of proposals. 
            K is the number of categories. 5 dimensions are 
            (x1, y1, x2, y2, confidence).
            conf_thresh: number, threshold for confidence value, <0.8 gives more
            boxes.
            nms_thresh: number, threshold for NMS, >0.3 gives more boxes.
        Returns:
            thresh: numpy.ndarray, N x K boolean mask.
        """
        N = det_boxes.shape[0]
        K = det_boxes.shape[1]
        thresh = np.zeros((N, K), dtype='bool')

        for cls_ind in xrange(K):
            cls_boxes = det_boxes[:, cls_ind, :4]
            cls_scores = det_boxes[:, cls_ind, 4]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = np.array(nms(dets, nms_thresh), dtype='int64')
            thresh[keep, cls_ind] = True
            throw = dets[:, 4] <= conf_thresh
            thresh[throw, cls_ind] = False

        return thresh

    def detect_single(self, im, obj_proposals,
                      conf_thresh=0.8, nms_thresh=0.3):
        """Run Fast-RCNN on a single image, with pre-computed proposals.
        
        Args:
            image: numpy.ndarray, image pixels.
            obj_proposals: numpy.ndarray, 
        """
        det_results = self.run_detector(im, obj_proposals)
        det_dict = threshold(det_Results, conf_thresh=conf_thresh,
                             nms_thresh=nms_thresh)
        return det_dict

    def shift_obj_proposals(self, boxes):
        """Shift the boxes and reorder.
        """
        return np.concatenate(
            (boxes[:, 1: 2],
             boxes[:, 0: 1],
             boxes[:, 3: 4] - 1,
             boxes[:, 2: 3] - 1), axis=1)

    def compute_obj_proposals(self, image_fname):
        """Compute object proposals using selective search.
        """
        boxes = selective_search.get_windows([image_fname])
        log.info('Selective search boxes shape: {:s}'.format(boxes[0].shape))
        return boxes[0]

    def vis_all(self, im, results):
        """Visualize detected bounding boxes of all classes"""
        classes = self.dataset.get_cat_list()
        for cls_id in xrange(len(classes)):
            cls = classes[cls_id]
            dets = results[results[:, 4] == cls_id]
            if dets.size > 0:
                self.vis_detections(im, cls, dets)
        plt.show()
        return

    def vis_detections(self, im, class_name, dets):
        """Draw detected bounding boxes."""
        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        for i in xrange(dets.shape[0]):
            bbox = dets[i, :4]
            score = dets[i, 5]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

        ax.set_title(('{} detections').format(class_name), fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.draw()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('-gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('-cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('-net', help='Path to network definition',
                        default=('../lib/fast-rcnn/'
                                 'models/VGG16/coco/test.prototxt'))
    parser.add_argument('-weights', help='Path to weights file',
                        default=('../lib/fast-rcnn/data/fast_rcnn_models/'
                                 'coco_vgg16_fast_rcnn_iter_240000.caffemodel'))
    parser.add_argument('-dataset', default='mscoco', help='Dataset to use')
    parser.add_argument('-datadir', default='../data/mscoco',
                        help='Dataset directory')
    parser.add_argument('-image', help='Image')
    parser.add_argument('-conf', default=0.8, type=float,
                        help='Confidence threshold')
    parser.add_argument('-nms', default=0.3, type=float,
                        help='NMS threshold')
    parser.add_argument('-image_list', help='Image list to run in batch')
    parser.add_argument('-output', help='Output of the extracted boxes')
    parser.add_argument('-proposal', help='Proposals file')
    parser.add_argument('-plot', action='store_true', help='Plot the result')
    parser.add_argument('-local_feat', action='store_true',
                        help='Whether to extract local feature')
    parser.add_argument('-feat_layer',
                        default='fc7',
                        help='Layer name for extracting local feature')
    parser.add_argument('-sparse', action='store_true',
                        help='Whether to store sparse format')

    args = parser.parse_args()

    return args


def assemble_boxes(boxes, thresh, local_feat=None):
    """Assemble the boxes into int16 array
    
    Returns:
        results: numpy.ndarray, (N, 6 + D) dimension. N is the number of boxes.
        6 dimensions are (x1, y1, x2, y2, class_id, score),
        D is the number of local feature dimension. 
        co-ordinates are un-normalized. class_id is 1-based.
    """
    N = boxes.shape[0]
    K = boxes.shape[1]
    if local_feat is None:
        D = 0
    else:
        D = local_feat.shape[-1]
    results = []
    for cls_id in xrange(K):
        cls_boxes = boxes[thresh[:, cls_id], cls_id, :]
        results_i = np.zeros((cls_boxes.shape[0], D + 6), dtype='float32')
        results_i[:, :4] = cls_boxes[:, :4]
        results_i[:, 4] = cls_id
        results_i[:, 5] = cls_boxes[:, 4]
        if local_feat is not None:
            results_i[:, 6:] = local_feat[thresh[:, cls_id]]
        results.append(results_i)
    results = np.concatenate(results, axis=0)
    return results


def save_boxes(results, output, sparse=False):
    """Save results to disk
    
    Args:
        results: list of numpy.ndarray.
        output: string, path to the output file.
        sparse: bool, whether the format is sparse.
    """
    output_dir = os.path.dirname(output)
    if output_dir != '':
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    log.info('Saving to {}'.format(os.path.abspath(output)))
    if sparse:
        log.info('Saving sparse matrix')

        # Compute the separators
        idx = 0
        sep_indices = []
        for i in xrange(len(results)):
            idx += results[i].shape[0]
            sep_indices.append(idx)
        f = h5py.File(output, 'w')
        results = np.concatenate(results, axis=0)
        log.info('Final shape: {}'.format(results.shape))
        results_sparse = scipy.sparse.csr_matrix(results)

        f['sep'] = np.array(sep_indices, dtype='int64')
        f['shape'] = results_sparse._shape
        f['data'] = results_sparse.data
        f['indices'] = results_sparse.indices
        f['indptr'] = results_sparse.indptr
        f.close()
    else:
        array = np.array(results, dtype=object)
        np.save(output, array)
    return


def run_image(fname, args, obj_proposals=None):
    """Run a single image.
    
    Args:
        fname: string, image file name.
        args: command line arguments object.
        obj_proposals: numpy.ndarray, pre-computed object proposals.
    Returns:
        results: numpy.ndarray, Nx(D+6). N is the number of filtered proposals.
        D is the dimension of local features.
    """
    if args.proposal:
        im = cv2.imread(fname)
        det_results = api.run_detector(
            im, obj_proposals,
            get_local_feat=args.local_feat,
            feat_layer=args.feat_layer)
    else:
        det_results = api.run_detector_with_proposal(
            fname,
            get_local_feat=args.local_feat,
            feat_layer=args.feat_layer)

    if args.local_feat:
        det_boxes = det_results[0]
        local_feat = det_results[1]
    else:
        det_boxes = det_results

    thresh = api.threshold(det_boxes, args.conf, args.nms)
    # disable background
    thresh[:, 0] = False

    if args.local_feat:
        results = assemble_boxes(det_boxes, thresh, local_feat)
    else:
        results = assemble_boxes(det_boxes, thresh)

    return results

if __name__ == '__main__':
    args = parse_args()
    log.log_args()
    if args.image:
        log.info('Single file mode')
        log.info('Input image: {}'.format(args.image))
        if not os.path.exists(args.image):
            log.fatal('File not found {}'.format(args.image))
    elif args.image_list:
        log.info('Batch file mode')
        log.info('Input image list: {}'.format(args.image_list))
        if not args.output:
            log.warning('Batch files: not saving results to file')
        image_list = list_reader.read_file_list(args.image_list, check=True)
    else:
        log.fatal('You need to specify input through -image or -image_list.')
    if args.output:
        log.info('Writing output to {}'.format(args.output))

    log.info('Confidence threshold: {:.2f}'.format(args.conf))
    log.info('NMS threshold: {:.2f}'.format(args.nms))

    log.info('Caffe network definition: {}'.format(args.net))
    log.info('Caffe network weights: {}'.format(args.weights))

    api = FastRcnnApi(args.net, args.weights, args.dataset, args.datadir,
                      not args.cpu_mode, args.gpu_id)

    if args.proposal:
        log.info('Loading pre-computed proposals')
        proposal_list = np.load(args.proposal)

    if args.image:
        results = run_image(args.image, args)
        if args.plot:
            im = cv2.imread(args.image)
            api.vis_all(im, results)
    elif args.image_list:
        results = []
        for i, img_fname in enumerate(image_list):
            log.info('Running {:d}/{:d}: {}'.format(
                i + 1, len(image_list), img_fname))
            det_success = False
            max_failure = 20
            failure_counter = 0
            while not det_success:
                try:
                    det_success = True
                    if args.proposal:
                        results_i = run_image(img_fname, args,
                                              obj_proposals=proposal_list[i])
                    else:
                        results_i = run_image(img_fname, args)
                except Exception as e:
                    log.error(e)
                    failure_counter += 1
                    if failure_counter >= max_failure:
                        det_success = True
                        log.error(
                            '{:d} fails, retrying'.format(failure_counter))
                    else:
                        det_success = False
                        log.error(
                            '{:d} fails, aborting'.format(failure_counter))
                    # Re-initialize API
                    log.error('Error occurred, re-initializing network')
                    api = None
                    gc.collect()
                    api = FastRcnnApi(args.net, args.weights, args.dataset,
                                      args.datadir, not args.cpu_mode,
                                      args.gpu_id)
            results.append(results_i)
            log.info('After threshold: {:d} boxes'.format(
                results[-1].shape[0]))

    # Save the results.
    if args.output:
        save_boxes(results, args.output, sparse=args.sparse)
