import os
import six.moves.cPickle as pickle
import numpy as np

def parse_rec(annofile, set_number, seq_number, frame_number, min_width=10, use_occul=False, min_visible_ratio=2):
    # Retrieve objects for that frame in annotations
    try:
        objects = annofile['set{:02d}'.format(set_number)]['V{:03d}'.format(seq_number)]['frames']['{}'.format(frame_number)]
    except KeyError as e:
        objects = None # Simply no objects for that frame    
        
    """ Parse a caltech annotation json file """
    # find all persons/people in the frame
    pedestrians = []
    
    if objects:
        for o in objects:
            ped_struct = {}
            occul = False
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
                        occul = True
                        visible_pos = (o['posv'][1], o['posv'][0], o['posv'][3], o['posv'][2]) # Convert to (y, x, h, w)
                        if visible_pos[2] * visible_pos[3] < min_visible_ratio * pos[2] * pos[3]:
                            good = False
                            pos = visible_pos
                elif o['occl'] == 1:
                    good = False

                if good:
                    ped_struct['name'] = 'pedestrian'
                    ped_struct['occul'] = occul
                    ped_struct['bbox'] = [pos[1] - 1,
                                          pos[0] - 1,
                                          pos[1] + pos[3] - 1, 
                                          pos[0] + pos[2] - 1 ]
                    pedestrians.append(ped_struct)

    return pedestrians


def caltech_ap(rec, prec):
    """ ap = caltech_ap(rec, prec)
    Compute caltech dataset AP given precision and recall.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def caltech_eval(detpath,
                 annofile,
                 imagesetfile,
                 image_set_index,
                 classname,
                 cachedir,
                 ovthresh=0.5):
    """rec, prec, ap = caltech_eval(detpath,
                                    annofile,
                                    imagesetfile,
                                    image_set_index,
                                    classname,
                                    cachedir,
                                    [ovthresh])

    Top level function that does the caltech pedestrian dataset evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annofile: annotation file 
        annofile should be the json annotations file.
    imagesetfile: the path to the image dataset.
    image_set_index : Index for the image dataset.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annofile
    # assumes imagesetfile is a path to the image dataset
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, im_index in enumerate(image_set_index):
            recs[i] = parse_rec(annofile, *im_index)
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(image_set_index)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'w') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for i, im_index in enumerate(image_set_index):
        R = [obj for obj in recs[i] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        occul = np.array([x['occul'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~occul)
        class_recs[imagename] = {'bbox': bbox,
                                 'occul': occul,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['occul'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a occuluded
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = caltech_ap(rec, prec)

    return rec, prec, ap