from __future__ import absolute_import
import numpy as np
import cv2
from scipy.ndimage import zoom
import pylab
import os
import argparse

import caffe

from utils.visualizer import visualize_heatmap
from data.utils.file_utils import get_annotations_from_xml, get_list_from_file
import data.cfg as cfg

caffe.set_device(0)
caffe.set_mode_gpu()

mean_pixel = np.array([104.008, 116.669, 122.675])


def parse_args():
  parser = argparse.ArgumentParser(description='Get FG heatmap')
  parser.add_argument('--dataset', dest='dataset',
                      default='pascal', type=str,
                      help='imagenet or pascal')
  parser.add_argument('--image_set', dest='image_set',
                      default='trainval', type=str,
                      help='image set')
  parser.add_argument('--result_dir', dest='result_dir',
                      default=None, type=str,
                      help='Heatmap will be saved here')
  parser.add_argument('--vis_dir', dest='vis_dir',
                      default=None, type=str,
                      help='Heatmap visualization will be saved here')


  args = parser.parse_args()
  return args


def preprocess(image, size):
  image = np.array(image)
  H, W, _ = image.shape
  image = zoom(image.astype('float32'), (size / H, size / W, 1.0), order=1)

  image = image[:, :, [2, 1, 0]]
  image = image - mean_pixel

  image = image.transpose([2, 0, 1])
  return image


def main(args):
  net_CAM = caffe.Net(os.path.join(cfg.root_dir, 'SEC', 'deploy.prototxt'),
                      os.path.join(cfg.root_dir, 'SEC', 'weights.caffemodel'),
                      caffe.TEST)
  data_dir = getattr(cfg, '%s_dir' % args.dataset)
  filelist = get_list_from_file(os.path.join(data_dir, 'ImageSets', 'Main', '%s.txt' % args.image_set))
  print 'Num of images', len(filelist)

  for file in filelist:
    image = pylab.imread(os.path.join(data_dir, 'JPEGImages', '%s.jpg' % file))
    H = image.shape[0]
    W = image.shape[1]
    _, labels, _ = get_annotations_from_xml(os.path.join(data_dir,
                                                         'Annotations', '%s.xml' % file),
                                            cfg.pascal_classes)
    if np.sum(np.absolute(labels[1:])) > 0:
      print file

      net_CAM.blobs['images'].data[...][0] = preprocess(image, 321.0)
      net_CAM.forward()

      CAM_scores = net_CAM.blobs['fc7_CAM'].data[0]
      params = net_CAM.params['scores'][0].data[...]

      for i in range(1, 21):
        if labels[i] != 0:
          w = params[i - 1]
          heat_maps = np.sum(CAM_scores * w[:, None, None], axis=0)
          if args.result_dir:
            np.save('%s/%s_%s' % (args.result_dir, file, cfg.pascal_classes[i]), heat_maps)
          if args.vis_dir:
            vis = visualize_heatmap(np.maximum(heat_maps, 0))
            cv2.imwrite('%s/%s_%s.png' % (args.vis_dir, file, cfg.pascal_classes[i]), vis)


if __name__ == "__main__":
  args = parse_args()
  if args.result_dir and not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)
  if args.vis_dir and not os.path.exists(args.vis_dir):
    os.mkdir(args.vis_dir)
  main(args)
