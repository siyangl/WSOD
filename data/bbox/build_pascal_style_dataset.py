# Modified from https://github.com/tensorflow/models/tree/master/inception/inception/data
# For proposals, the input format is .mat. If 'boxes' present in mat content, treat boxes
# category-independent proposals. Category-specific proposals use the category name as key.
# Each image file has a proposal mat with identical filename.
# boxes should be in [#boxes, 4] format, where coordinates are in order of
# [hmin, wmin, hmax, wmax] in absolute coordinate and 1-index

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading
import scipy.io as sio
import numpy as np
import tensorflow as tf

import data.utils.file_utils as file_utils
import utils.box_utils as box_utils


tf.app.flags.DEFINE_string('image_dir', None,
                           'Image data directory')
tf.app.flags.DEFINE_string('annotation_dir', None,
                           'Annotation data directory')
tf.app.flags.DEFINE_string('image_set', None,
                           'train, val, test, trainval, etc.')
tf.app.flags.DEFINE_string('image_list_file', None,
                           'List of images')
tf.app.flags.DEFINE_string('category_list_file', None,
                           'List of categories')
tf.app.flags.DEFINE_string('roi_dir', None,
                           'Proposal data directory')
tf.app.flags.DEFINE_string('output_dir', None,
                           'Output data directory')
tf.app.flags.DEFINE_bool('flip_offline', False,
                         'Flip images offline for data augmentation')
tf.app.flags.DEFINE_bool('normalize_bbox', True,
                         'normalize bbox to range [0, 1]')
tf.app.flags.DEFINE_integer('data_shards', 128,
                            'Number of shards in TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')

# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   dog
#   cat
#   flower
# where each line corresponds to a label. We map each label contained in
# the file to an integer corresponding to the line number starting from 0.
tf.app.flags.DEFINE_string('label_list_file', None, 'A list of possible labels')
FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _convert_to_example(filename, image_buffer, label, roi_label, proposal, obj_label, obj_bbox, height, width):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    text: string, unique human-readable, e.g. 'dog'
    height: integer, image height in pixels
    width: integer, image width in pixels
    bbox: list of bounding boxes; each box is a list of integers
      specifying [ymin, xmin, ymax, xmax]. All boxes are assumed to belong to
      the same label as the image label
    obj_label: list of object labels. e.g. [4, 2, 19], length may vary
  Returns:
    Example proto
  """

  # convert to list
  label = label.tolist()
  roi_label = roi_label.tolist()
  assert obj_label.shape[0] == obj_bbox.shape[0]
  obj_label = obj_label.tolist()

  # object boxes
  obj_ymin = (obj_bbox[:, 0]).tolist()
  obj_xmin = (obj_bbox[:, 1]).tolist()
  obj_ymax = (obj_bbox[:, 2]).tolist()
  obj_xmax = (obj_bbox[:, 3]).tolist()

  # proposals
  assert proposal.shape[1] == 4
  ymin = (proposal[:, 0]).tolist()
  xmin = (proposal[:, 1]).tolist()
  ymax = (proposal[:, 2]).tolist()
  xmax = (proposal[:, 3]).tolist()

  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/roi_label': _int64_feature(roi_label),
      'image/proposal/bbox/xmin': _float_feature(xmin),
      'image/proposal/bbox/xmax': _float_feature(xmax),
      'image/proposal/bbox/ymin': _float_feature(ymin),
      'image/proposal/bbox/ymax': _float_feature(ymax),
      'image/object/bbox/label':_int64_feature(obj_label),
      'image/object/bbox/xmin': _float_feature(obj_xmin),
      'image/object/bbox/xmax': _float_feature(obj_xmax),
      'image/object/bbox/ymin': _float_feature(obj_ymin),
      'image/object/bbox/ymax': _float_feature(obj_ymax),
      'image/format': _bytes_feature(image_format),
      'image/filename': _bytes_feature(os.path.basename(filename)),
      'image/encoded': _bytes_feature(image_buffer)}))
  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _is_png(filename):
  """Determine if a file contains a PNG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a PNG.
  """
  return '.png' in filename


def _process_image(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  image_data = tf.gfile.FastGFile(filename, 'r').read()

  # Convert any PNG to JPEG's for consistency.
  if _is_png(filename):
    print('Converting PNG to JPEG for %s' % filename)
    image_data = coder.png_to_jpeg(image_data)

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               roi_labels, labels, proposals, obj_labels, obj_bboxes, num_shards):
  """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in xrange(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_dir, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i]
      roi_label = roi_labels[i]
      proposal = proposals[i]
      obj_label = obj_labels[i]
      obj_bbox = obj_bboxes[i]
      assert len(obj_bbox) == len(obj_label)

      image_buffer, height, width = _process_image(filename, coder)

      example = _convert_to_example(filename, image_buffer, label,
                                    roi_label, proposal, obj_label, obj_bbox, height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(name, filenames, roi_labels, labels, proposals, obj_labels, obj_bboxes, num_shards):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  assert len(filenames) == len(roi_labels)
  assert len(filenames) == len(labels)
  assert len(filenames) == len(proposals)
  assert len(filenames) == len(obj_labels)
  assert len(filenames) == len(obj_bboxes)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  # threads = []
  for i in xrange(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i+1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in xrange(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames,
            roi_labels, labels, proposals, obj_labels, obj_bboxes, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _find_image_files(data_dir, annotation_dir, image_list_file, category_list_file):
  """Build a list of all images files and labels in the data set.

  Args:
    data_dir: string, path to the root directory of images.
    annotation_dir: path to annotation files
    image_list_file: a text file listing all image file names
    category_list_file: a text file listing all possible classes for this dataset

  Returns:
    filenames: list of strings; each string is a path to an image file.
    labels: list of vector; each vector identifies the ground truth.
    roi_labels: list of vector, identifying one class of interest
    obj_labels: list of array, instance-level annotation
    obj_bboxes: list of array, instance-level box annotation
    proposals: regions of interest (for the class of interest if roi_labels are valid,
      else for all exsiting categories
  """

  labels = []
  filenames = []
  proposals = []
  roi_labels = []
  obj_labels = []
  obj_bboxes = []
  filename_list = file_utils.get_list_from_file(image_list_file)
  class_list = file_utils.get_list_from_file(category_list_file)
  print(len(filename_list))

  # Construct the list of JPEG files and labels.
  for filename in filename_list:
    jpeg_file_path = '%s/%s.jpg' % (data_dir, filename)
    # read .xml label file
    xml_path = '%s/%s.xml' % (annotation_dir, filename)
    img_size, label, objects = file_utils.get_annotations_from_xml(xml_path, class_list)
    bbox = objects[:, 1:]
    obj_label = objects[:, 0]

    if FLAGS.normalize_bbox:
      bbox = box_utils.bbox_normalize(bbox, img_size[0:2], one_index=True)

    # get bbox
    if FLAGS.roi_dir:
      roi_dir = os.path.join(FLAGS.roi_dir, '%s.mat' % filename)
      roi_contents = sio.loadmat(roi_dir)
      if not 'boxes' in roi_contents:  # category-dependent proposals
        file_duplicates = 0
        for key in roi_contents.keys():
          if not key.startswith('__'):
            label_index = class_list.index(key)
            if label[label_index] == -1:
              continue
            roi_label = np.zeros(len(class_list), dtype=np.int64)
            roi_label[label_index] = 1
            proposal = roi_contents[key]
            assert proposal.shape[1] == 4
            if FLAGS.normalize_bbox:
              proposal = box_utils.bbox_normalize(proposal, img_size[0:2], one_index=True)
            file_duplicates += 1
            proposals.append(proposal)
            roi_labels.append(roi_label)
      else:  # category-independent proposals
        file_duplicates = 1
        proposal = roi_contents['boxes']
        assert proposal.shape[1] == 4
        if FLAGS.normalize_bbox:
          proposal = box_utils.bbox_normalize(proposal, img_size[0:2], one_index=True)
        proposals.append(proposal)
        roi_labels.append(label)
    else:
      file_duplicates = 1
      proposals.append(np.array([[-1, -1, -1, -1]]))
      roi_labels.append(label)

    labels.extend([label] * file_duplicates)
    filenames.extend([jpeg_file_path]*file_duplicates)
    obj_labels.extend([obj_label]*file_duplicates)
    obj_bboxes.extend([bbox]*file_duplicates)

  print(len(labels))
  print(len(filenames))
  print(len(obj_labels))
  shuffled_index = range(len(filenames))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]
  roi_labels = [roi_labels[i] for i in shuffled_index]
  proposals = [proposals[i] for i in shuffled_index]
  obj_labels = [obj_labels[i] for i in shuffled_index]
  obj_bboxes = [obj_bboxes[i] for i in shuffled_index]

  # print(filenames[0])
  # print(labels[0])
  # print(roi_labels[0])
  # print(obj_bboxes[0])
  # print(obj_labels[0])
  return filenames, labels, roi_labels, proposals, obj_labels, obj_bboxes


def _process_dataset(name, data_dir, annotation_dir, image_list_file, category_list_file, num_shards):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    labels_file: string, path to the labels file.
  """
  filenames, labels, roi_labels, proposals, obj_labels, obj_bboxes =\
    _find_image_files(data_dir, annotation_dir, image_list_file, category_list_file)
  _process_image_files(name, filenames, roi_labels, labels, proposals, obj_labels, obj_bboxes, num_shards)


def main(unused_argv):
  assert not FLAGS.data_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.data_shards')
  assert FLAGS.image_dir
  assert FLAGS.annotation_dir
  assert FLAGS.image_set
  assert FLAGS.image_list_file
  assert FLAGS.category_list_file
  assert FLAGS.output_dir, 'Output_dir should be provided'
  if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)
  print('Saving results to %s' % FLAGS.output_dir)

  # Run it!
  _process_dataset(FLAGS.image_set, FLAGS.image_dir, FLAGS.annotation_dir,
                   FLAGS.image_list_file, FLAGS.category_list_file, FLAGS.data_shards)


if __name__ == '__main__':
  tf.app.run()
