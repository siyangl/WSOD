import numpy as np
import os

import tensorflow as tf

import data.utils.file_utils as file_utils

tf.app.flags.DEFINE_string('image_list', None,
                           'List of input images')
tf.app.flags.DEFINE_string('annotation_dir', None,
                           'Annotation data directory')
tf.app.flags.DEFINE_string('save_dir', None,
                           'Save the statistics to the directory')
tf.app.flags.DEFINE_string('category_list_file', None,
                           'List of categories')
FLAGS = tf.app.flags.FLAGS


def main(unused):
  category_list = file_utils.get_list_from_file(FLAGS.category_list_file)
  num_categories = len(category_list)
  co_occur_freq = np.zeros((num_categories, num_categories))
  category_freq = np.zeros((num_categories))
  image_list = file_utils.get_list_from_file(FLAGS.image_list)
  for image in image_list:
    xml_path = os.path.join(FLAGS.annotation_dir, '%s.xml'%image)
    _, labels, objs = file_utils.get_annotations_from_xml(xml_path, category_list)
    obj_labels = np.unique(np.absolute(objs[:, 0]))
    labels = np.absolute(labels)
    if obj_labels.shape[0] == 1:
      co_occur_freq[obj_labels[0], obj_labels[0]] += 1
    for l in range(0, obj_labels.shape[0]):
      for k in range(l+1, obj_labels.shape[0]):
        co_occur_freq[obj_labels[l], obj_labels[k]] += 1
        co_occur_freq[obj_labels[k], obj_labels[l]] += 1
    category_freq += labels

  co_occur_prob = co_occur_freq/np.expand_dims(category_freq, 1)
  if FLAGS.save_dir:
    np.save(os.path.join(FLAGS.save_dir, 'pascal_co_occur'), co_occur_prob)


if __name__ == "__main__":
  tf.app.run()
