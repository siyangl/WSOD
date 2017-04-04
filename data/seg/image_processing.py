import tensorflow as tf


def parse_example_proto(example_serialized, num_classes):
  """Parses an Example proto containing a training example of an image."""
  # Dense features in Example proto.
  feature_map = {
    'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                        default_value=''),
    'image/class/label': tf.FixedLenFeature([num_classes],
                                            dtype=tf.int64),  # need a default value to handle errors
    'image/class/roi_label': tf.FixedLenFeature([num_classes],
                                                dtype=tf.int64),  # need a default value to handle errors
    'image/height': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
    'image/width': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
    'image/filename': tf.FixedLenFeature([], dtype=tf.string,
                                         default_value=''),
    'image/seg': tf.FixedLenFeature([], dtype=tf.string,
                                    default_value=''),
    'image/seg_mask': tf.FixedLenFeature([], dtype=tf.string,
                                         default_value='')
  }

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)
  size = [features['image/height'], features['image/width']]
  roi_label = features['image/class/roi_label']
  img_data = features['image/encoded']
  filenames = features['image/filename']
  seg = features['image/seg']
  seg_weight = features['image/seg_mask']
  return img_data, filenames, size, label, roi_label, seg, seg_weight


def decode_jpeg(image_buffer, image_size, scope=None):
  """Decode a JPEG string into one 3-D float image Tensor.
  Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor
  """
  with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3)

    # resize images
    image = tf.image.resize_images(image, image_size, image_size)
    image = tf.cast(image, tf.uint8)
    return image


def decode_png(image_buffer, image_size, scope=None):
  """Decode a JPEG string into one 3-D float image Tensor.
  Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor
  """
  with tf.op_scope([image_buffer], scope, 'decode_png'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_png(image_buffer, channels=1)

    # resize images
    image = tf.image.resize_images(image, image_size, image_size, method=1)
    return image


def batch_inputs(dataset, train=False, flip_image=False, crop_image=False, batch_size=1,
                 num_preprocess_threads=4, image_size=321):
  """Contruct batches of training or evaluation examples from the image dataset.
  Args:
    dataset: instance of Dataset class specifying the dataset.
      See dataset.py for details.
    batch_size: integer
    train: boolean
    num_preprocess_threads: integer, total number of preprocessing threads
    num_readers: integer, number of parallel readers
  Returns:
    images: 4-D float Tensor of a batch of images
    labels: 1-D integer Tensor of [batch_size].
  Raises:
    ValueError: if data is not found
  """
  with tf.name_scope('batch_processing'):
    data_files = dataset.data_files()
    if train:
      filename_queue = tf.train.string_input_producer(data_files,
                                                      shuffle=True,
                                                      capacity=16)
    else:
      filename_queue = tf.train.string_input_producer(data_files,
                                                      shuffle=False,
                                                      capacity=1)

    reader = dataset.reader()
    _, example_serialized = reader.read(filename_queue)
    images_and_labels = []

    # Parse a serialized Example proto to extract the image and metadata.
    image_buffer, filenames, sizes, label_index, roi_labels, segs_buffer, seg_weights_buffer = \
      parse_example_proto(example_serialized, dataset.num_classes())

    image = decode_jpeg(image_buffer, image_size=image_size)
    segs = decode_png(segs_buffer, image_size=41)
    seg_weights = decode_png(seg_weights_buffer, image_size=41)

    if train and flip_image:
      image_seg = tf.concat(2, [image, segs, seg_weights])
      image_seg = tf.image.random_flip_left_right(image_seg)
      image = image_seg[:, :, 0:3]
      segs = tf.to_int32(image_seg[:, :, 3:4])
      seg_weights = tf.to_int32(image_seg[:, :, 4:5])
    images_and_labels.append([image, filenames, sizes, label_index, roi_labels, segs, seg_weights])

    images, filename_batch, size_batch, label_index_batch, roi_label_batch, segs_batch, seg_weights_batch = \
      tf.train.batch_join(
              images_and_labels,
              batch_size=batch_size,
              capacity=2 * num_preprocess_threads * batch_size)

    images = tf.cast(images, tf.float32)
    height = image_size
    width = image_size
    depth = 3
    images = tf.reshape(images, shape=[batch_size, height, width, depth])
    label_batch = tf.reshape(label_index_batch, [batch_size, dataset.num_classes()])

    return images, filename_batch, size_batch, label_batch, roi_label_batch, segs_batch, seg_weights_batch
