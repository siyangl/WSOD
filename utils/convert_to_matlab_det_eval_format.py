import cPickle as pickle
import scipy.io as sio
import argparse
import numpy as np
import os

from data.cfg import pascal_classes, pascal_dir
from data.utils.file_utils import get_list_from_file


def parse_args():
  parser = argparse.ArgumentParser(description='Format convert')
  parser.add_argument('--image_list', dest='image_list',
                      default=None, type=str,
                      help='image list')
  parser.add_argument('--annotation_dir', dest='annotation_dir',
                      default=None, type=str,
                      help='Annotation dir')
  parser.add_argument('--detection_file', dest='detection_file',
                      default=None, type=str,
                      help='The pickle file from fast RCNN')
  parser.add_argument('--thres', dest='thres',
                      default=0.8, type=float,
                      help='Only visualize result with confidence larger than this threshold')
  parser.add_argument('--result_dir', dest='result_dir',
                      default=None, type=str,
                      help='Save mat files to here')
  args = parser.parse_args()
  return args


def main():
  with open(args.detection_file, 'rb') as f:
    all_dets = pickle.load(f)
  result_dir = os.path.join(os.path.dirname(args.detection_file), 'mat_det')
  if not os.path.exists(result_dir):
    os.mkdir(result_dir)

  filelist = get_list_from_file(os.path.join(pascal_dir, 'ImageSets', 'Main', 'test.txt'))
  for i in range(len(filelist)):
    det_list = []
    file = filelist[i]
    # xml_path = os.path.join(pascal_dir, 'Annotations_bak_true_anno', '%s.xml' % file)
    # _, label, obj_labels = get_annotations_from_xml(xml_path, pascal_classes)
    # obj_labels = np.array(get_object_labels_from_xml(xml_path))
    for j in range(1, len(pascal_classes)):
      # if label[j] > 0:
      #   all_gt[j] += 1
      dets = all_dets[j][i]
      # print dets.shape
      valid_dets = dets[dets[:, 4] > args.thres, :]
      valid_dets_and_category = np.zeros((valid_dets.shape[0], 6))
      if valid_dets.shape[0] > 0:
        # valid_dets[:, 4] = j
        valid_dets_and_category[:, 5] = j
        valid_dets_and_category[:, 0:5] = valid_dets
        det_list.append(valid_dets_and_category)
    if len(det_list) > 0:
      det_list = np.concatenate(det_list, axis=0)
      sio.savemat(os.path.join(result_dir, '%s.mat'%file), mdict={'dets': det_list})
    # print det_list
    # break


if __name__ == "__main__":
  args = parse_args()
  main()