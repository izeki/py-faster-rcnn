#!/usr/bin/env python

"""Test a LSTM-RCNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, get_output_dir, cfg_from_file, cfg_from_list
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from fast_rcnn.nms_wrapper import nms
from datasets.factory import get_imdb
from utils.timer import Timer
from utils.blob import im_list_to_blob
import numpy as np
import cv2
import caffe
import six.moves.cPickle as pickle
from six.moves import range
import argparse
import pprint
import time
import os
import sys

class TestWraper(object):
    '''
    This wraper class is used to help the generation of sequential data
    in the testing process which is required in the RNN context.
    '''
    def __init__(self, net, imdb, max_per_image=100, thresh=0.05, vis=false):
        assert cfg.TEST.HAS_RPN is True, 'RPN must present!'
        self._net = net
        self._imdb = imdb
        self._max_per_image = max_per_image
        self._thresh = thresh
        self._vis = vis
        self._num_images = self._imdb.image_index
        self._num_classes = self._imdb.num_classes
        self._perm = np.arange(len(self.num_images))
        self._cur = 0
        
    def _get_next_sequence_inds(self):
        """Retun the roidb indices for the next minibatch in sequential order."""
        db_inds = self._perm[self._cur:self._cur+cfg.TEST.VIDEO_CLIP_LENGTH * cfg.TEST.FRAME_PER_BATCH]
        self.cur += cfg.TEST.FRAME_PER_BATCH
        
        return db_inds        
    
    def _get_next_sequence_data(self):
        """Return the blobs and image indices to used for the next minibatch in sequential order."""
        db_inds = self._get_next_sequence_inds()
        return self._get_next_sequence(db_inds), db_inds
    
    def _get_next_sequence(self, db_inds):
        """construct a sequential minibatch."""        

        # Get the input image blob, formatted for caffe
        im_blob, im_infos = _get_seq_image_blob(db_inds)

        blobs = {'data': im_blob, 'im_info': im_infos}        

        # clip_markers: TxN flags to control whether the previous state should be kelpt
        # T is the time length and N is the data length per unit time
        # Set the bias to the forget gate to 5.0 as explained in the clockwork RNN paper
        clip_markers_blob = np.ones((blobs['data'].shape[0], 1,1,1), dtype=np.float32)
        clip_markers_blob[0:cfg.TRAIN.FRAME_PER_BATCH,:,:,:] = 0

        blobs['clip_markers'] = clip_markers_blob

        return blobs
    
    def _get_seq_image_blob(self, db_inds):
        """
        Builds a sequential input blob from the images in the imdb
        Only one test scale is implemented.
        """
        assert len(cfg.TEST.SCALES) == 1, 'Only one test scale is supported!'
        processed_ims = []
        #im_scales = []
        im_info_blob = np.zeros((0, 3), dtype=np.float32)
        for i in db_inds:
            im = cv2.imread(self._imdb.image_path_at(i))    
            im_orig = im.astype(np.float32, copy=True)
            im_orig -= cfg.PIXEL_MEANS
            im_shape = im_orig.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            
            for target_size in cfg.TEST.SCALES:
                im_scale = float(target_size) / float(im_size_min)
                # Prevent the biggest axis from being more than MAX_SIZE
                if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                    im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)

                im = cv2.resize(
                    im_orig,
                    None, None,
                    fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR
                )

            #im_scales.append(im_scale)
            processed_ims.append(im)            
            im_info = np.array([[im.shape[0], im.shape[1], im_scale]],
                                dtype=np.float32)
            im_info_blob = np.vstack((im_info_blob, im_info))
            
        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, im_info_blob
    
    def im_detect(self, blobs):
        # reshape network inputs
        self.net.blobs['data'].reshape(*(blobs['data'].shape))
        self.net.blobs['im_info'].reshape(*(blobs['im_info'].shape))

        # do forward
        forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
        blobs_out = net.forward(**forward_kwargs)
        
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        # (batch_len * batch_size) x num_rois x 5
        num_images = blobs['data'].shape[0]
        for i in range(num_images):
            im_scale = blobs['im_info'][i][2]
            boxes = rois[i, :, 1:5] / im_scales
        
        # (batch_len * batch_size) x num_rois x num_classes
        scores = blobs_out['cls_prob']
        # (batch_len * batch_size) x num_rois x (5 * num_classes)
        pred_boxes = np.tile(boxes, (1, 1, scores.shape[1]))
        
        return scores, pred_boxes
    
    def vis_detections(self, im, class_name, dets, thresh=0.3):
        """Visual debugging of detections."""
        import matplotlib.pyplot as plt

        im = im[:, :, (2, 1, 0)]
        for i in range(np.minimum(10, dets.shape[0])):
            bbox = dets[i, :4]
            score = dets[i, -1]

            if score > thresh:
                plt.cla()
                plt.imshow(im)
                plt.gca().add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor='g', linewidth=3)
                    )
                plt.title('{}  {:.3f}'.format(class_name, score))
                plt.show()
        
    def test_model(self):
        # all detections are collected into:
        # all_boxes[cls][image] = N x 5 array of detections in
        # (x1, y1, x2, y2, score)
        all_boxes = [[[] for _ in range(self._num_images)]
                    for _ in range(self._num_classes)]
        
        output_dir = get_output_dir(self._imdb, self._net)
        
        # timers
        _t = {'im_detect' : Timer(), 'misc' : Timer()}
        
        for i in range(cfg.TEST.NUM_VIDEO_BATCHES):
            blobs, db_inds = self._get_next_sequence_data()
            
            _t['im_detect'].tic()
            scores, boxes = self.im_detect(blobs)
            _t['im_detect'].toc()
            
            _t['misc'].tic()
            for j in db_inds:
                for k in range(self._num_classes):
                    inds = np.where(scores[j, :, k] > self._thresh)[0]
                    cls_scores = scores[j, inds, k]
                    cls_boxes = boxes[j, inds, k*4:(k+1)*4]
                    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                        .astype(np.float32, copy=False)
                    keep = nms(cls_dets, cfg.TEST.NMS)
                    cls_dets = cls_dets[keep, :]
                    if vis:
                        im = cv2.imread(self._imdb.image_path_at(j))
                        vis_detections(im, self._imdb.classes[j], cls_dets)
                    all_boxes[k][j] = cls_dets          
                # Limit to max_per_image detections *over all classes*
                if self._max_per_image > 0:
                    image_scores = np.hstack([all_boxes[k][j][:, -1]
                                              for k in range(1, self._imdb.num_classes)])
                    if len(image_scores) > self._max_per_image:
                        image_thresh = np.sort(image_scores)[-self._max_per_image]
                        for k in range(1, self._imdb.num_classes):
                            keep = np.where(all_boxes[k][j][:, -1] >= image_thresh)[0]
                            all_boxes[k][j] = all_boxes[k][j][keep, :]
            _t['misc'].toc()
            
            print(
                'im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(
                    i + 1,
                    cfg.TEST.NUM_VIDEO_BATCHES,
                    _t['im_detect'].average_time,
                    _t['misc'].average_time
                )
            )
        
        det_file = os.path.join(output_dir, 'detections.pkl')

        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        self._imdb.evaluate_detections(all_boxes, output_dir)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU id to use',
        default=0, type=int
    )
    parser.add_argument(
        '--def', dest='prototxt',
        help='prototxt file defining the network',
        default=None, type=str
    )
    parser.add_argument(
        '--net', dest='caffemodel',
        help='model to test',
        default=None, type=str
    )
    parser.add_argument(
        '--cfg', dest='cfg_file',
        help='optional config file', default=None, type=str
    )
    parser.add_argument(
        '--wait', dest='wait',
        help='wait until net file exists',
        default=True, type=bool
    )
    parser.add_argument(
        '--imdb', dest='imdb_name',
        help='dataset to test',
        default='voc_2007_test', type=str
    )
    parser.add_argument(
        '--comp', dest='comp_mode', help='competition mode',
        action='store_true')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys', default=None,
        nargs=argparse.REMAINDER
    )
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections',
        action='store_true'
    )
    parser.add_argument(
        '--num_dets', dest='max_per_image',
        help='max number of detections per image',
        default=100, type=int
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)

    tw = TestWraper(net, imdb, max_per_image=args.max_per_image, vis=args.vis)

    print('Testng...')
    tw.test_model()
    print('Done testng.')

