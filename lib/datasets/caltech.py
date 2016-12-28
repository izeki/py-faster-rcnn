import os, glob
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import six.moves.cPickle as pickle
import subprocess
import uuid
import json
from datasets.caltech_eval import caltech_eval
from fast_rcnn.config import cfg
from six.moves import range

class caltech(imdb):
    def __init__(self, image_set, caltech_path=None):
        imdb.__init__(self, 'caltech_' + image_set)
        self._image_set = image_set  # data set for processing, the value can be "triain" or "test"
        if caltech_path is None:
            self._data_path = self._get_default_path()
        else:
            self._data_path = caltech_path
        self._classes = ('__background__',  # always index 0
                         'pedestrian')
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        # caltech dataset specific config options
        self.config = {'cleanup': True,
                       'rpn_file': None,
                       'net_model': 'Faster-RCNN', # model used for detection. the value could be "Faster-RCNN" or "LSTM-RCNN"
                       'min_width': 10,  # Minimum width for objects to be included      
                       'use_occul': False, # If set to true, the occluded objects within min_visible_ratio will be included 
                       'min_visible_ratio': 2, # Minimum ratio of area visible for occluded objects to be included
                       'skip_frame': False, # If set to true, the frames_to_skip frames will be skipped.
                       'frames_to_skip': 30 # The number of frames to skip
                      }
        
        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)
        
    def _get_default_path(self):
        """
        Return the default path where caltech data set is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'caltech-dataset/dataset')
    
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(*self._image_index[i])

    def image_path_from_index(self, set_number, seq_number, frame_number):
        """
        Construct an image path from the image's "index" identifier.
        """
        
        image_path = os.path.join(self._data_path, '/images/set{:02d}/V{:03d}.seq/{}'.format(set_number, seq_number, frame_number)
                                  + self._image_ext)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)

        return image_path
    
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        if self._image_set == 'train':
            return self.discover_training()
        elif self._image_set == 'test':
            return self.discover_testing()
        else:
            raise ValueError('image_set can only be either "train" or "test".')
    
    def discover_seq(self, set_number, seq_number):
        """
        Return the list of the frame numbers for the video sequence and video set.
        """
        num_frames = len(glob.glob(os.path.join(self._data_path, '/images/set{:02d}/V{:03d}.seq/*'.format(set_number, seq_number) + self._image_ext)))
        
        skip_frames = self.cofig['skip_frame']        
        if skip_frames:
            frame_modulo = self.config['frames_to_skip']
            num_frames = int(floor(num_frames / frame_modulo))

            return [(set_number, seq_number, frame_modulo * i - 1) for i in range(1, num_frames + 1)]
        else:
            return [(set_number, seq_number, i) for i in range(num_frames)]

    def discover_set(self, set_number):
        """
        Return the list of the frame numbers by giving the video set.
        """
        num_sequences = len(glob.glob(os.path.join(self._data_path, '/images/set{:02d}/V*.seq'.format(set_number))))

        tuples = []
        for seq_number in range(num_sequences):
            tuples += self.discover_seq(set_number, seq_number)

        return tuples

    def discover_training(self):
        """
        Construct the list of frame numbers for training.
        """
        training = []
        for set_number in range(5 + 1):
            training += self.discover_set(set_number)

        return training        

    def discover_testing(self):
        """
        Construct the list of frame numbers for testing.
        """
        testing = []
        for set_number in range(6, 10 + 1):
            testing += self.discover_set(set_number, skip_frames = True)

        return testing
    
    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)

            print('{} gt roidb loaded from {}'.format(self.name, cache_file))

            return roidb

        gt_roidb = [self._load_caltech_annotation(*index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)

        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb
    
    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb
    
    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']

        print('loading {}'.format(filename))
        assert os.path.exists(filename), 'rpn data not found at: {}'.format(filename)

        with open(filename, 'rb') as f:
            box_list = pickle.load(f)

        return self.create_roidb_from_box_list(box_list, gt_roidb)
    
    def load_annotations(self):
        if self.annotations:
            return
        
        annotation_path = os.path.join(self._data_path + '/annotations.json')
        
        assert os.path.exists(annotation_path), 'Path does not exist: {}'.format(annotation_path)        

        with open(annotation_path) as json_file:
            self.annotations = json.load(json_file)
    
    def _load_caltech_annotation(self, set_number, seq_number, frame_number):
        """
        Load image and bounding boxes info from json file in the caltech data
        format.
        """
        self.load_annotations()        
        
        # Retrieve objects for that frame in annotations
        try:
            objects = self.annotations['set{:02d}'.format(set_number)]['V{:03d}'.format(seq_number)]['frames']['{}'.format(frame_number)]
        except KeyError as e:
            objects = None # Simply no objects for that frame
        
        # find all persons/people in the frame
        pedestrians = []
        min_width = self.condif['min_width']
        use_occul = self.config['use_occul']
        min_visible_ratio = self.config['min_visible_ratio']        
        
        if objects:
            for o in objects:
                good = False
                pos = (o['pos'][1], o['pos'][0], o['pos'][3], o['pos'][2]) # Convert to (y, x, h, w)
                
                if o['lbl'] in ['person', 'people']:
                    good = True

                    # Remove objects with very small width (are they errors in labeling?!)
                    if pos[3] < min_width:
                        good = False
                    
                    if o['occl'] == 1 && use_occul:
                        if type(o['posv']) == int:
                            good = False
                        else:
                            visible_pos = (o['posv'][1], o['posv'][0], o['posv'][3], o['posv'][2]) # Convert to (y, x, h, w)
                            if visible_pos[2] * visible_pos[3] < min_visible_ratio * pos[2] * pos[3]:
                                good = False
                                pos = visible_pos
                    elif o['occl'] == 1:
                        good = False
                    
                    if good:
                        pedestrians.append(pos)
        
        num_objs = len(pedestrians)
        
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for caltech data is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        
        # Load object bounding boxes into a data frame.
        for ped in pedestrians:
            # Make pixel indexes 0-based
            x1 = ped[1] - 1
            y1 = ped[0] - 1
            x2 = ped[1] + ped[3] - 1
            y2 = ped[0] + ped[2] - 1
            cls = self._class_to_ind['pedestrian']
            
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}
    
    def _get_caltech_results_filename_template(self):
        # caltech-dataset/dataset/results/<net_model>_det_test_pedestrian.txt
        net_model = self.config['net_mode']
        filename = net_model + 'det_' + self._image_set + '_{:s}.txt'
        
        path = os.path.join(
            self._data_path,
            'results',
            filename)

        return path
    
    def _write_caltech_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue

            print('Writing {} caltech results file'.format(cls))
            
            filename = self._get_caltech_results_filename_template().format(cls)
            
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]

                    if dets == []:
                        continue

                    # the caltech expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
                        
    def _do_eval(self, output_dir='output'):
        self.load_annotations()
        annofile = self.annotations
        imagesetfile = os.path.join(self._data_path, '/images/') 
        image_set_index = self._image_index
        
        cachedir = os.path.join(self._data_path, 'annotations_cache')
        aps = []
        
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue

            filename = self._get_caltech_results_filename_template().format(cls)
            rec, prec, ap = caltech_eval(
                filename, annofile, imagesetfile, image_set_index, cls, cachedir, ovthresh=0.5)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))

            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')

        for ap in aps:
            print('{:.3f}'.format(ap))

        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
    
    def evaluate_detections(self, all_boxes, output_dir):
        self._write_caltech_results_file(all_boxes)
        self._do_eval(output_dir)

        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue

                filename = self._get_caltech_results_filename_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['cleanup'] = False
        else:
            self.config['cleanup'] = True
            
if __name__ == '__main__':
    from datasets.caltech import caltech
    d = caltech('train')
    res = d.roidb
    from IPython import embed
    embed()