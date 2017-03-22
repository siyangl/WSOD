import numpy as np

import data.utils.file_utils as file_utils
from cfg import ROOT_DIR

def test_get_label_and_objects_from_xml():
  xmlpath='%s/data/utils/test_xml.xml'%ROOT_DIR
  pascal_classes = ['__background__',
                    'aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor']
  size, labels, objects = file_utils.get_annotations_from_xml(xmlpath, pascal_classes)
  assert np.array_equal(size, np.array([375, 500, 3]))
  assert np.array_equal(labels, np.array([0, 0, 0, 0, 0, 0, 0,
                                          1, 0, 1, 0, 0, 0, 0,
                                          0, -1, 0, 0, 0, 0, 0]))
  assert np.array_equal(objects, np.array([[9, 211, 263, 339, 324],
                                           [7, 264, 165, 372, 253],
                                           [-9, 244, 5, 374, 67],
                                           [9, 194, 241, 299, 295],
                                           [-15, 186, 277, 220, 312]]))


def test_get_list_from_file():
  file = '%s/data/utils/sample_list.txt'%ROOT_DIR
  content = file_utils.get_list_from_file(file)
  assert len(content) == 2


if __name__ == "__main__":
  test_get_label_and_objects_from_xml()
  test_get_list_from_file()