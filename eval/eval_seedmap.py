from __future__ import absolute_import
import numpy as np
import argparse
import cv2
import os
import logging
from datetime import datetime

from data.utils.file_utils import get_annotations_from_xml, get_list_from_file
from data.cfg import pascal_classes
from utils.box_utils import boxes_to_seg


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
                      help='Location cue map dir')
  parser.add_argument('--seedmask_dir', dest='seedmask_dir',
                      default=None, type=str,
                      help='Location cue map dir')
  args = parser.parse_args()
  return args


def main():
  logdir = os.path.dirname(args.seg_dir)
  log_filename = os.path.join(logdir, 'seed_eval_%s.log' % datetime.now())
  logging.basicConfig(filename=log_filename, level=logging.DEBUG)
  logging.info('Folder to be evaluated: %s, %s'%(args.seg_dir, args.seedmask_dir))
  tp = np.zeros(len(pascal_classes))
  gt = np.zeros(len(pascal_classes))
  pd = np.zeros(len(pascal_classes))

  image_list = get_list_from_file(args.image_list)
  for image in image_list:
    img_size, img_labels, bboxes = get_annotations_from_xml(os.path.join(args.annotation_dir, '%s.xml'%image),
                                                            pascal_classes)
    fg_map = np.zeros(img_size, dtype=np.bool)
    for i in range(1, len(pascal_classes)):
      if img_labels[i] > 0:
        seg = cv2.imread(os.path.join(args.seg_dir, '%s_%s.png'%(image, pascal_classes[i])))
        seedmask = cv2.imread(os.path.join(args.seedmask_dir, '%s_%s.png'%(image, pascal_classes[i])))
        cue = np.logical_and((seg == i), (seedmask==1))
        gt_bbox = bboxes[np.absolute(bboxes[:, 0]) == i][:, 1:]
        box_map = boxes_to_seg(gt_bbox, img_size)
        gt[i] += np.sum(box_map)
        tp[i] += np.sum(np.logical_and(box_map, cue))
        pd[i] += np.sum(cue)
        fg_map = np.logical_or(fg_map, box_map)

    bg_cue = np.logical_and((seg == 0), (seedmask==1))
    bg_map = np.logical_not(fg_map)
    gt[0] += np.sum(bg_map)
    tp[0] += np.sum(np.logical_and(bg_map, bg_cue))
    pd[0] += np.sum(bg_cue)

  for i in range(len(pascal_classes)):
    logging.info('%s: recall: %.3f, precision: %.3f'%(pascal_classes[i], tp[i]/gt[i], tp[i]/pd[i]))
  logging.info('Average recall: %.3f'%(np.sum(tp[1:]/gt[1:])/(len(pascal_classes) - 1)))
  logging.info('Average precision: %.3f'%(np.sum(tp[1:]/pd[1:])/(len(pascal_classes) - 1)))


if __name__ == "__main__":
  args = parse_args()
  assert args.seg_dir, 'Must provide seg directory'
  assert args.seedmask_dir, 'Must provide seed mask directory'
  main()