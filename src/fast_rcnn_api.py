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
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sio
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

    def run_detector_with_proposal(self, image_fname):
        """Run Fast-RCNN on a single image, without pre-computed proposals.
        """
        obj_proposals = self.compute_obj_proposals(image_fname)
        obj_proposals = self.shift_obj_proposals(obj_proposals)
        im = cv2.imread(image_fname)
        return self.run_detector(im, obj_proposals)

    def run_detector(self, im, obj_proposals):
        """Run Fast-RCNN on a single image, with pre-computed proposals.
        
        Args:
            im: numpy.ndarray, image pixels.
            obj_proposals: numpy.ndarray, 
        
        Returns:
            results: numpy.ndarray, N x 5, N is the number of proposals.
            5 dimensions are (x1, y1, x2, y2, confidence).
        """
        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()

        # K is the number of classes
        # N is the number of proposals
        # scores = (N, K)
        # boxes = (N, K x 4)
        scores, boxes = im_detect(self.model.net, im, obj_proposals)
        timer.toc()
        log.info(('Detection took {:.3f}s for '
                  '{:d} object proposals').format(
                      timer.total_time, boxes.shape[0]))
        boxes = boxes.reshape(boxes.shape[0], boxes.shape[1] / 4, 4)
        scores = scores.reshape(scores.shape[0], scores.shape[1], 1)
        results = np.concatenate((boxes, scores), axis=-1)
        return results

    def threshold(self, det_results, conf_thresh=0.8, nms_thresh=0.3):
        """Run thresholding on the boxes
        
        Args:
            det_results: numpy.ndarray, N x 5, N is the number of proposals. 
            5 dimensions are (x1, y1, x2, y2, confidence).
            conf_thresh: number, threshold for confidence value, <0.8 gives more
            boxes.
            nms_thresh: number, threshold for NMS, >0.3 gives more boxes.
        Returns:
            det_dict: dict, keys are class index, values are M x 5 
            numpy.ndarray, M is the number of proposals for the class
        """
        classes = self.dataset.get_cat_list()
        det_dict = {}

        for cls_ind, cls in enumerate(classes):
            cls_boxes = det_results[:, cls_ind, :4]
            cls_scores = det_results[:, cls_ind, -1]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, nms_thresh)
            dets = dets[keep, :]
            dets = dets[dets[:, -1] > conf_thresh]
            if dets.shape[0] > 0:
                det_dict[cls_ind] = dets

        return det_dict

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

    def vis_all(self, im, det_dict):
        """Visualize detected bounding boxes of all classes"""
        classes = self.dataset.get_cat_list()
        for cls_id in det_dict.iterkeys():
            cls = classes[cls_id]
            dets = det_dict[cls_id]
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
            score = dets[i, -1]

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
                        default=('../lib/fast-rcnn/data/fast_rcnn_models/',
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

    args = parser.parse_args()

    return args


def num_boxes(det_dict):
    """Count the number of boxes in the thresholded dictionary."""
    N = 0
    for k in det_dict.iterkeys():
        N += det_dict[k].shape[0]
    return N


def assemble_boxes(det_dict):
    """Assemble the boxes into int16 array
    
    Returns:
        boxes: numpy.ndarray, (N, 6) dimension. N is the number of boxes.
        6 dimensions are (x1, y1, x2, y2, class_id, score). 
        co-ordinates are un-normalized. class_id is 1-based. score is quantized 
        in 0 - 100.
    """
    N = num_boxes(det_dict)
    results = np.zeros((N, 6), dtype='int16')
    idx = 0
    for cls_id in det_dict.iterkeys():
        cls_boxes = det_dict[cls_id]
        num_cb = cls_boxes.shape[0]
        results[idx: idx + num_cb, :4] = np.floor(cls_boxes[:, :4])
        results[idx: idx + num_cb, 4] = cls_id
        results[idx: idx + num_cb, 5] = np.floor(cls_boxes[:, 4] * 100)
        idx += num_cb
    return results


def save_boxes(results, output):
    output_dir = os.path.dirname(output)
    if output_dir != '':
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    array = np.array(results, dtype=object)
    log.info(array)
    np.save(output, array)
    return


if __name__ == '__main__':
    args = parse_args()
    log.log_args()
    if args.image:
        log.info('Single file mode')
        log.info('Input image: {}'.format(args.image))
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
        if args.proposal:
            raise Exception('Not implemented')
        else:
            det_boxes = api.run_detector_with_proposal(args.image)
            det_dict = api.threshold(det_boxes, args.conf, args.nms)
            results = assemble_boxes(det_dict)
    elif args.image_list:
        results = []
        for i, img_fname in enumerate(image_list):
            log.info('Running {:d}/{:d}: {}'.format(
                i + 1, len(image_list), img_fname))
            im = cv2.imread(img_fname)
            if args.proposal:
                obj_proposals = proposal_list[i]
                det_boxes = api.run_detector(im, obj_proposals)
            else:
                det_boxes = api.run_detector_with_proposal(img_fname)
            # conf_list = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
            # nms_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            # for conf_i, nms_i in zip(conf_list, nms_list):
            #     det_dict = api.threshold(det_boxes, conf_i, nms_i)
            #     del det_dict[0]
            #     log.info('conf {:.2f} nms {:.2f} num boxes {:d}'.format(
            #         conf_i, nms_i,
            #         num_boxes(det_dict)))
            det_dict = api.threshold(det_boxes, args.conf, args.nms)
            del det_dict[0]
            final_boxes = assemble_boxes(det_dict)
            results.append(assemble_boxes(det_dict))
            log.info('After threshold: {:d} boxes'.format(results[-1].shape[0]))

    # Save the results.
    if args.output:
        save_boxes(results, args.output)
