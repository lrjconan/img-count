"""
Interface to evaluate Fast R-CNN network

==Command-line Usage
    >> python fast_rcnn_api.py -gpu {gpu ID}
                               -image {image file}
                               -image_list {image list file}
                               -dataset {dataset name}
                               -datadir {dataset directory}
                               -net {network definition path}
                               -weights {weights path}
                               -cpu {cpu mode}
                               -conf {confidence threshold}
                               -nms {NMS threshold}
                               -proposal {proposal file}
                               -output {output file}
                               -plot {whether to plot in single image mode}
                               -local_feat {local feature layer names}
                               -restore {whether to restore from previous run}
                               -num_images_per_shard {size of each shard}
                               -groundtruth {whether proposals contain gt cat}

==Command-line Example
    >> python fast_rcnn_api.py -gpu 2 \
                               -image_list "image_list_train.txt" \
                               -dataset "mscoco" \
                               -datadir "data/mscoco" \
                               -net "fast_rcnn.prototxt" \
                               -weights "fast_rcnn.caffemodel" \
                               -conf 0.3 \
                               -nms 0.8 \
                               -local_feat "fc7,pool5" \
                               -proposal "select_search_train.npy" \
                               -output "fast_rcnn_c3n8_train" \
                               -num_images_per_shard 10000

==API Example
    >> # Run selective search.
    >> import selective_search
    >> ssbox = selective_search.get_windows(['test.png'])
    >>
    >> # Initialize API.
    >> from fast_rcnn_api import FastRCNN_API
    >> api = FastRCNN_API(model_def_fname='fast_rcnn.prototxt',
                          weights_fname='fast_rcnn.caffemodel',
                          dataset_name='mscoco',
                          dataset_dir='data/mscoco',
                          gpu=True,
                          gpu_id=2)
    >>
    >> # Run detector.
    >> im = cv2.imread('test.png')
    >> dets = api.run_detector(im, ssbox)
    >>
    >> # Threhshold the final predictions.
    >> dets = api.threshold(dets, conf_thresh=0.8, nms_thresh=0.3)

"""
from data_api import MSCOCO
from fast_rcnn.test import im_detect
from fast_rcnn_model import FastRCNN_Model
from fast_rcnn_utils.nms import nms
from fast_rcnn_utils.timer import Timer
from utils import logger
from utils import list_reader
from utils.sharded_hdf5 import ShardedFile, ShardedFileWriter, KEY_NUM_ITEM
import argparse
import caffe
import cv2
import fast_rcnn_init
import gc
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import selective_search
import sys

log = logger.get()
FEATURE_DIM = {}


class FastRCNN_API(object):
    """
    Interface to run inference on trained Fast-RCNN network.
    Properties:
        model:
        dataset:
    """

    def __init__(self, model_def_fname, weights_fname,
                 dataset_name, dataset_dir, gpu=True, gpu_id=0):
        if args.dataset == 'mscoco':
            self.dataset = MSCOCO(base_dir=dataset_dir, set_name='valid')
        else:
            raise Exception('Unknown dataset: {}.'.format(dataset))

        if not os.path.isfile(model_def_fname):
            raise IOError('Model definition file not found: {}'.format(
                weights_fname))
        if not os.path.isfile(weights_fname):
            raise IOError('Weights file not found: {}'.format(weights_fname))

        self.model = FastRCNN_Model(name=model_def_fname,
                                    model_def_fname=model_def_fname,
                                    weights_fname=weights_fname,
                                    gpu=gpu,
                                    gpu_id=gpu_id)
        pass

    def run_detector_with_proposal(self, image_fname, feat_layers=None):
        """Run Fast-RCNN on a single image, without pre-computed proposals.
        """
        obj_proposals = self.compute_obj_proposals(image_fname)
        obj_proposals = self.shift_obj_proposals(obj_proposals)
        im = cv2.imread(image_fname)
        return self.run_detector(im, obj_proposals, feat_layers=feat_layers)

    def run_detector(self, im, obj_proposals, feat_layers=None):
        """Run Fast-RCNN on a single image, with pre-computed proposals.

        Args:
            im: numpy.ndarray, image pixels.
            obj_proposals: numpy.ndarray, 
        Returns:
            results: numpy.ndarray, (N, K, 5), N is the number of proposals,
            K is the number of categories,
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

        if feat_layers is not None:
            local_feat = {}
            for layer in feat_layers:
                data = self.model.net.blobs[layer].data
                # Average-pooling on convolutional layers.
                if len(data.shape) == 4:
                    data = data.mean(axis=-1).mean(axis=-1)
                if len(data.shape) != 2:
                    raise Exception('Number of feature dimension is not 2.')
                data = data[inv_index]
                local_feat[layer] = data
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

    def vis_all(self, im, results_dict):
        """Visualize detected bounding boxes of all classes"""
        classes = self.dataset.get_cat_list()
        for cls_id in xrange(len(classes)):
            clas = classes[cls_id]
            dets = results['boxes'][results['categories'] == cls_id]
            if dets.size > 0:
                self.vis_detections(im, clas, dets)
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
    parser.add_argument('-local_feat', default=None,
                        help=('Comma delimited list of layer names for '
                              'extracting local feature'))
    parser.add_argument('-num_images_per_shard', default=10000, type=int,
                        help='Number of images per shard')
    parser.add_argument('-restore', action='store_true',
                        help='Whether to continue from previous checkpoint.')
    parser.add_argument('-groundtruth', action='store_true',
                        help=('Whether object proposals contain groundtruth '
                              'category in the 5th dimension'))
    parser.add_argument('-shuffle', action='store_true',
                        help='Whether to shuffle the image sequence.')
    parser.add_argument('-max_num_images', default=-1,
                        type=int, help='Maximum number of images to run')

    args = parser.parse_args()

    return args


def assemble_boxes(boxes, thresh, local_feat=None):
    """Assemble the boxes into int16 array

    Args:
        boxes: numpy.ndarray, (N, K, 4) dimension
        thresh: numpy.ndarray, (N, K) dimension
        local_feat: dict, keys are layer names, values are (N, D) numpy ndarray
    Returns:
        results_dict:
    """
    N = boxes.shape[0]
    K = boxes.shape[1]
    boxes_remain = []
    categories = []
    scores = []
    feats = {}

    # Copy feature layer names.
    for key in local_feat.iterkeys():
        feats[key] = []

    # Extract for each class.
    for cls_id in xrange(K):
        cls_boxes = boxes[thresh[:, cls_id], cls_id, :]
        boxes_remain.append(cls_boxes[:, :4].astype('float32'))
        num_boxes = cls_boxes.shape[0]
        categories.append(np.array([cls_id] * num_boxes, dtype='int64'))
        scores.append(cls_boxes[:, 4].astype('float32'))
        for key in local_feat.iterkeys():
            feats[key].append(local_feat[key][thresh[:, cls_id]])

    # Concatenate matrices
    boxes_remain = np.concatenate(boxes_remain, axis=0)
    categories = np.concatenate(categories, axis=0)
    scores = np.concatenate(scores, axis=0)

    for key in feats.iterkeys():
        feats[key] = np.concatenate(feats[key], axis=0)

    # Compile resulting dictionary.
    results_dict = {
        'boxes': boxes_remain,
        'categories': categories,
        'scores': scores
    }

    for key in feats.iterkeys():
        results_dict[key] = feats[key]

    return results_dict


def run_image(fname, args, obj_proposals=None, gt_cat=None):
    """Run a single image.

    Args:
        fname: string, image file name.
        args: command line arguments object.
        obj_proposals: numpy.ndarray, pre-computed object proposals.
        gt_cat: numpy.ndarray, groundtruth mode only. Groundtruth categories.
    Returns:
        results_dict: dict, keys are 'boxes', 'categories', 'scores', and 
        local feature layer names.
    """
    if args.groundtruth:
        if gt_cat is None:
            raise Exception('Need to provide groundtruth category in gt mode')

    # Local feature layers.
    feat_layers = None
    if args.local_feat:
        feat_layers = args.local_feat.split(',')

    # Run detectors.
    if args.proposal:
        if len(obj_proposals) > 0:
            im = cv2.imread(fname)
            det_results = api.run_detector(
                im, obj_proposals,
                feat_layers=feat_layers)
        else:
            log.error('Empty proposals for image {}'.format(fname))
            return None
    else:
        det_results = api.run_detector_with_proposal(
            fname,
            feat_layers=feat_layers)

    if args.local_feat:
        det_boxes = det_results[0]
        local_feat = det_results[1]
    else:
        det_boxes = det_results

    # Apply threshold.
    if args.groundtruth:
        # Do not throw out any boxes if in groundtruth mode.
        thresh = np.zeros((det_boxes.shape[0], det_boxes.shape[1]),
                          dtype='bool')

        # +1 for adding background category.
        thresh[np.arange(gt_cat.shape[0]), gt_cat + 1] = True
        log.info('Goundtruth category')
        log.info(gt_cat + 1)
        log.info('Prediction max')
        log.info(np.argmax(det_boxes[:, :, 4], axis=1))

        # Assign delta-distribution on groundtruth category.
        det_boxes[:, :, 4] = 0.0
        det_boxes[np.arange(gt_cat.shape[0]), gt_cat + 1, 4] = 1.0
    else:
        thresh = api.threshold(det_boxes, args.conf, args.nms)

    # Disable background.
    thresh[:, 0] = False

    # Pack the results.
    if args.local_feat:
        results_dict = assemble_boxes(det_boxes, thresh, local_feat)
        if len(FEATURE_DIM) < len(feat_layers):
            for name in feat_layers:
                FEATURE_DIM[name] = results_dict[name].shape[-1]
    else:
        results_dict = assemble_boxes(det_boxes, thresh)

    # Add image information.
    results_dict['image'] = fname

    return results_dict

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
        if args.shuffle:
            rand = np.random.RandomState(2)
            rand.shuffle(image_list)
        if args.max_num_images != -1:
            if args.max_num_images <= 0:
                log.fatal('max_num_images must be greater than 0.')
            else:
                image_list = image_list[:args.max_num_images]
    else:
        log.fatal('You need to specify input through -image or -image_list.')

    if args.output:
        log.info('Writing output to {}'.format(args.output))
        log.info('Num images per shard: {:d}'.format(
            args.num_images_per_shard))

    log.info('Confidence threshold: {:.2f}'.format(args.conf))
    log.info('NMS threshold: {:.2f}'.format(args.nms))

    log.info('Caffe network definition: {}'.format(args.net))
    log.info('Caffe network weights: {}'.format(args.weights))

    api = FastRCNN_API(args.net, args.weights, args.dataset, args.datadir,
                       not args.cpu_mode, args.gpu_id)

    if args.proposal:
        log.info('Loading pre-computed proposals')
        proposal_list = np.load(args.proposal)
        if args.groundtruth:
            log.info('Groundtruth mode')
            gt_cat_list = [p[:, 4] for p in proposal_list]
            proposal_list = [p[:, :4] for p in proposal_list]

    if args.image:
        results = run_image(args.image, args)
        if args.plot:
            im = cv2.imread(args.image)
            api.vis_all(im, results_dict)
    elif args.image_list:
        num_images = len(image_list)

        if args.output:
            dirname = os.path.dirname(args.output)
            if not os.path.exists(dirname):
                log.info('Making directory {}'.format(dirname))
                os.makedirs(dirname)

            num_shards = int(
                np.ceil(num_images / float(args.num_images_per_shard)))
            output_file = ShardedFile(args.output, num_shards)
            writer = ShardedFileWriter(output_file, num_objects=num_images)
            if args.restore:
                log.info('Restore previous run')
                checkpoint = 0
                checkfile = 0
                for i in xrange(num_shards):
                    fname_i = output_file.get_fname(i)
                    if os.path.exists(fname_i):
                        h5file = h5py.File(fname_i)
                        if KEY_NUM_ITEM in h5file:
                            num = h5file[KEY_NUM_ITEM][0]
                            checkpoint += num
                            log.info('Found {:d} items in {}'.format(
                                num, fname_i))
                            checkfile += 1
                        else:
                            break
                    else:
                        break
                log.info('Restarting from checkpoint {:d}'.format(checkpoint))
                writer.seek(pos=checkpoint, shard=checkfile)
            it = writer
        else:
            it = xrange(num_images)

        for i in it:
            img_fname = image_list[i]
            log.info('Running {:d}/{:d}: {}'.format(
                i + 1, len(image_list), img_fname))
            det_success = False
            max_failure = 20
            failure_counter = 0
            while not det_success:
                try:
                    det_success = True
                    if args.proposal:
                        if args.groundtruth:
                            results = run_image(img_fname, args,
                                                obj_proposals=proposal_list[i],
                                                gt_cat=gt_cat_list[i])
                        else:
                            results = run_image(img_fname, args,
                                                obj_proposals=proposal_list[i])
                    else:
                        results = run_image(img_fname, args)
                except Exception as e:
                    log.error(e)
                    log.log_exception(e)
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
                    api = FastRCNN_API(args.net, args.weights, args.dataset,
                                       args.datadir, not args.cpu_mode,
                                       args.gpu_id)

            if results is not None:
                log.info('After threshold: {:d} boxes'.format(
                    results['boxes'].shape[0]))

                # Write to output.
                try:
                    if args.output:
                        log.info('Writing to output buffer')
                        writer.write(results, key=img_fname)
                except Exception as e:
                    log.error(e)
                    log.log_exception(e)
                    raise e

        # Close writer and flush to file.
        writer.close()
