from __future__ import absolute_import
import numpy as np
import argparse
from skimage.measure import label, regionprops
import cv2
import os
import logging
from datetime import datetime

from data.utils.file_utils import get_annotations_from_xml, get_list_from_file
from data.cfg import pascal_classes
from utils.box_utils import get_iou_matrix


def parse_args():
  parser = argparse.ArgumentParser(description='Segmentation evaluation')
  parser.add_argument('--image_list', dest='image_list',
                      default=None, type=str,
                      help='image list')
  parser.add_argument('--annotation_dir', dest='annotation_dir',
                      default=None, type=str,
                      help='Annotation dir')
  parser.add_argument('--seg_dir', dest='seg_dir',
                      default=None, type=str,
                      help='Segmentation results dir')
  args = parser.parse_args()
  return args


def main():
  logdir = os.path.dirname(args.seg_dir)
  log_filename = os.path.join(logdir, 'corloc_eval_%s.log' % datetime.now())
  logging.basicConfig(filename=log_filename, level=logging.DEBUG)
  tp = np.zeros(len(pascal_classes))
  gt = np.zeros(len(pascal_classes))

  image_list = get_list_from_file(args.image_list)
  for image in image_list:
    _, img_labels, bboxes = get_annotations_from_xml(os.path.join(args.annotation_dir, '%s.xml'%image), pascal_classes)
    gt += np.maximum(img_labels, 0)
    seg_mask = cv2.imread(os.path.join(args.seg_dir, '%s.png'%image))
    seg_mask = seg_mask[:, :, 0]
    seg_mask =  seg_mask.astype(np.int32)
    for i in range(1, len(pascal_classes)):
      if img_labels[i] > 0:
        mask = (seg_mask == i)
        if np.sum(mask) > 0:
          regionmap = label(mask)
          region_prop = regionprops(regionmap)
          region_area = np.array([r.area for r in region_prop])
          best_region_idx = np.argmax(region_area)
          bbox = np.array(region_prop[best_region_idx].bbox)
          bbox = np.expand_dims(bbox, 0)
          gt_bbox = bboxes[bboxes[:, 0] == i][:, 1:]
          iou = get_iou_matrix(gt_bbox, bbox)
          if np.amax(iou > 0.5):
            tp[i] += 1
  for i in range(1, len(pascal_classes)):
    logging.info('%s: %.3f'%(pascal_classes[i], tp[i]/gt[i]))
  logging.info('CorLoc: %.3f'%(np.sum(tp[1:]/gt[1:])/(len(pascal_classes) - 1)))
  logging.info('Instance correct rate: %d/%d'%(np.sum(tp), np.sum(gt)))



if __name__ == "__main__":
  args = parse_args()
  assert args.seg_dir, 'Must provide segmentation result directory'
  main()