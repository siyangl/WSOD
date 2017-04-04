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

import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf

import data.utils.file_utils as file_utils

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
tf.app.flags.DEFINE_string('seg_dir', None,
                           'seg mask data directory')
tf.app.flags.DEFINE_string('seg_mask_dir', None,
                           'seg weights data directory')
tf.app.flags.DEFINE_string('output_dir', None,
                           'Output data directory')
tf.app.flags.DEFINE_bool('flip_offline', False,
                         'Flip images offline for data augmentation')
tf.app.flags.DEFINE_bool('one_hot', True,
                         'Duplicate images with more than one positive labels')
tf.app.flags.DEFINE_integer('data_shards', 128,
                            'Number of shards in TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')
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


def _convert_to_example(filename, image_buffer, label, roi_label, seg, seg_mask, height, width):
  """Build an Example proto for an example."""

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
    'image/seg': _bytes_feature(seg),
    'image/seg_mask': _bytes_feature(seg_mask),
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

    # Initializes function that decodes gray-scale PNG data.
    self._decode_png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._decode_png_data, channels=1)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

  def decode_png(self, image_data):
    image = self._sess.run(self._decode_png,
                           feed_dict={self._decode_png_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 1
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


def _process_mask(filename, coder):
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

  assert _is_png(filename)
  # Decode the Gray scale png.
  image = coder.decode_png(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 1

  return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               labels, roi_labels, seg_files, seg_mask_files, num_shards):
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
      label = labels[i].tolist()
      roi_label = roi_labels[i].tolist()
      seg_file = seg_files[i]
      seg_mask_file = seg_mask_files[i]

      image_buffer, height, width = _process_image(filename, coder)
      seg_buffer, _, _ = _process_mask(seg_file, coder)
      seg_mask_buffer, _, _ = _process_mask(seg_mask_file, coder)
      example = _convert_to_example(filename, image_buffer, label,
                                    roi_label, seg_buffer, seg_mask_buffer, height, width)
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


def _process_image_files(name, filenames, labels, roi_labels, segs, seg_masks, num_shards):
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
  assert len(filenames) == len(segs)
  assert len(filenames) == len(seg_masks)

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
            labels, roi_labels, segs, seg_masks, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _find_image_files(data_dir, annotation_dir, image_list_file, category_list_file,
                      seg_dir, seg_mask_dir):
  """Build a list of all images files and labels in the data set."""

  labels = []
  filenames = []
  seg_files = []
  roi_labels = []
  seg_mask_files = []
  filename_list = file_utils.get_list_from_file(image_list_file)
  class_list = file_utils.get_list_from_file(category_list_file)
  print(len(filename_list))

  # Construct the list of JPEG files and labels.
  for filename in filename_list:
    jpeg_file_path = '%s/%s.jpg' % (data_dir, filename)
    # read .xml label file
    xml_path = '%s/%s.xml' % (annotation_dir, filename)
    img_size, label, _ = file_utils.get_annotations_from_xml(xml_path, class_list)

    # get segmentation
    if FLAGS.one_hot:
      file_duplicates = 0
      for i in range(1, len(class_list)):
        cls = class_list[i]
        if label[i] > 0:
          file_duplicates += 1
          roi_label = np.zeros(len(class_list), dtype=np.int64)
          roi_label[i] = 1
          seg_file = os.path.join(seg_dir, '%s_%s.png'%(filename, cls))
          seg_mask_file = os.path.join(seg_mask_dir, '%s_%s.png'%(filename, cls))
          seg_files.append(seg_file)
          seg_mask_files.append(seg_mask_file)
          roi_labels.append(roi_label)
    else:
      file_duplicates = 1
      roi_label = label
      seg_file = os.path.join(seg_dir, '%s.png' % (filename))
      seg_mask_file = os.path.join(seg_mask_dir, '%s.png' % (filename))
      seg_files.append(seg_file)
      seg_mask_files.append(seg_mask_file)
      roi_labels.append(roi_label)

    labels.extend([label] * file_duplicates)
    filenames.extend([jpeg_file_path]*file_duplicates)

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  print(len(labels))
  print(len(filenames))
  print(len(seg_files))
  print(len(seg_mask_files))


  shuffled_index = range(len(filenames))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]
  roi_labels = [roi_labels[i] for i in shuffled_index]
  seg_files = [seg_files[i] for i in shuffled_index]
  seg_mask_files = [seg_mask_files[i] for i in shuffled_index]

  # print(filenames[0])
  # print(labels[0])
  # print(roi_labels[0])
  # print(obj_bboxes[0])
  # print(obj_labels[0])
  # return
  return filenames, labels, roi_labels, seg_files, seg_mask_files


def _process_dataset(image_set, image_dir, annotation_dir,
                     image_list_file, category_list_file, seg_dir, seg_mask_dir, num_shards):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    labels_file: string, path to the labels file.
  """
  filenames, labels, roi_labels, segs, seg_mask = _find_image_files(image_dir,
                                                                    annotation_dir,
                                                                    image_list_file,
                                                                    category_list_file,
                                                                    seg_dir,
                                                                    seg_mask_dir)
  _process_image_files(image_set, filenames, labels, roi_labels, segs, seg_mask, num_shards)


def main(unused_argv):
  assert not FLAGS.data_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.data_shards')
  assert FLAGS.output_dir, 'Must give a output dir'
  assert FLAGS.seg_dir, 'Must give a seg dir'
  assert FLAGS.seg_mask_dir, 'Must give a seg weights dir'

  if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)
  print('Saving results to %s' % FLAGS.output_dir)

  # Run it!
  _process_dataset(FLAGS.image_set, FLAGS.image_dir, FLAGS.annotation_dir,
                   FLAGS.image_list_file, FLAGS.category_list_file,
                   FLAGS.seg_dir, FLAGS.seg_mask_dir, FLAGS.data_shards)


if __name__ == '__main__':
  tf.app.run()
