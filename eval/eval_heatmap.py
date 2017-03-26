import numpy as np
import os
import argparse
import logging
from datetime import datetime
import cv2

from utils.box_utils import point_in_box
from data.utils.file_utils import get_annotations_from_xml, get_list_from_file
from data.cfg import pascal_classes

def parse_args():
  parser = argparse.ArgumentParser(description='Segmentation evaluation')
  parser.add_argument('--image_list', dest='image_list',
                      default=None, type=str,
                      help='image list')
  parser.add_argument('--annotation_dir', dest='annotation_dir',
                      default=None, type=str,
                      help='Annotation dir')
  parser.add_argument('--heatmap_dir', dest='heatmap_dir',
                      default=None, type=str,
                      help='Heatmap are saved here')
  args = parser.parse_args()
  return args


def main():
  logdir = os.path.dirname(args.heatmap_dir)
  log_filename = os.path.join(logdir, 'pointing_eval_%s.log' % datetime.now())
  logging.basicConfig(filename=log_filename, level=logging.DEBUG)
  logging.info('Folder to be evaluated: %s'%args.heatmap_dir)
  tp = np.zeros(len(pascal_classes))
  gt = np.zeros(len(pascal_classes))

  image_list = get_list_from_file(args.image_list)
  for image in image_list:
    img_size, img_labels, bboxes = get_annotations_from_xml(os.path.join(args.annotation_dir, '%s.xml'%image),
                                                     pascal_classes)
    gt += np.maximum(img_labels, 0)
    for i in range(1, len(pascal_classes)):
      if img_labels[i] > 0:
        heatmap = np.load(os.path.join(args.heatmap_dir, '%s_%s.npy'%(image, pascal_classes[i])))
        heatmap = cv2.resize(heatmap, (img_size[1], img_size[0]))
        global_argmax = np.where(heatmap==np.amax(heatmap))
        hmax = global_argmax[0]
        wmax = global_argmax[1]
        point = np.concatenate([hmax, wmax]) + 1
        gt_bbox = bboxes[bboxes[:, 0] == i][:, 1:]
        is_in = point_in_box(point, gt_bbox)
        if np.amax(is_in):
          tp[i] += 1

  for i in range(1, len(pascal_classes)):
    logging.info('%s: %.3f'%(pascal_classes[i], tp[i]/gt[i]))
  logging.info('CorLoc: %.3f'%(np.sum(tp[1:]/gt[1:])/(len(pascal_classes) - 1)))
  logging.info('Instance correct rate: %d/%d'%(np.sum(tp), np.sum(gt)))


if __name__ == "__main__":
  args = parse_args()
  assert args.heatmap_dir, 'Must provide heatmap directory'
  main()