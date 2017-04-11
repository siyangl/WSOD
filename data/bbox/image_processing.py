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
    'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64)
  }

  # # get bbox annotation
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  feature_map.update(
          {k: sparse_float32 for k in ['image/proposal/bbox/xmin',
                                       'image/proposal/bbox/ymin',
                                       'image/proposal/bbox/xmax',
                                       'image/proposal/bbox/ymax']})

  feature_map.update(
          {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                       'image/object/bbox/ymin',
                                       'image/object/bbox/xmax',
                                       'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)
  proposal = [features['image/proposal/bbox/ymin'],
              features['image/proposal/bbox/xmin'],
              features['image/proposal/bbox/ymax'],
              features['image/proposal/bbox/xmax']
              ]
  obj_bboxes = [features['image/object/bbox/ymin'],
                features['image/object/bbox/xmin'],
                features['image/object/bbox/ymax'],
                features['image/object/bbox/xmax']
                ]
  size = [features['image/height'], features['image/width']]

  roi_label = tf.cast(features['image/class/roi_label'], dtype=tf.int32)
  img_data = features['image/encoded']
  obj_labels = tf.cast(features['image/object/bbox/label'], tf.int32)
  filenames = features['image/filename']
  return img_data, label, filenames, size, proposal, obj_labels, obj_bboxes, roi_label


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
    return image


def _flip_bbox_left_right(bboxes, scope=None):
  """Box coordinate in order [hmin, wmin, hmax, wmax"""
  assert bboxes.get_shape().as_list()[1] == 4
  top, left, bottom, right = tf.unpack(bboxes, axis=1)
  new_top = top
  new_left = 1 - right
  new_bottom = bottom
  new_right = 1 - left
  flipped_bbox = tf.pack([new_top, new_left, new_bottom, new_right], axis=1)
  return flipped_bbox


def batch_inputs(dataset, train=False, flip_image=False, crop_image=False, batch_size=1,
                 max_num_proposals=2000,
                 num_preprocess_threads=4,
                 image_size=224):
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
    filenames_bath: string Tensor [batch_size]
    size_batch: int32 Tensor [batch_size, 2], [h, w]
    label_batch: 1-D integer Tensor of [batch_size, num_classes].
    proposal_batch: 3-D tensor [batch_size, num_proposals_per_image, 4]
    roi_label_batch: 2-D tensor [batch_size, num_classes]
    obj_labels_batch: 2-D tensor [batch_size, 50]
    obj_bbox_batch: 3-D tensor [batch_size, 50, 4]
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
    num_classes = dataset.num_classes()
    image_buffer, label_index, filenames, sizes, proposals, obj_labels, obj_bboxes, roi_labels = \
      parse_example_proto(example_serialized, num_classes)

    obj_labels = tf.sparse_to_dense(obj_labels.indices, [50], obj_labels.values, default_value=0)
    coord_list = []
    for coord in obj_bboxes:
      dense_coord = tf.sparse_to_dense(coord.indices, [50], coord.values)
      coord_list.append(tf.expand_dims(dense_coord, [1]))
    obj_bbox = tf.concat(1, coord_list)

    coord_list = []
    for coord in proposals:
      dense_coord = tf.sparse_to_dense(coord.indices, [max_num_proposals], coord.values)
      coord_list.append(tf.expand_dims(dense_coord, [1]))
    proposal = tf.concat(1, coord_list)

    if train:
      if crop_image:
        image = decode_jpeg(image_buffer, image_size=int(image_size*1.1))
        image = tf.random_crop(image, [image_size, image_size, 3])
      else:
        image = decode_jpeg(image_buffer, image_size=image_size)

      if flip_image:
        flip_flag = tf.to_float(tf.greater(tf.random_uniform([]), 0.5))
        image_flipped = tf.image.flip_left_right(image)
        obj_bbox_flipped = _flip_bbox_left_right(obj_bbox)
        proposal_flipped = _flip_bbox_left_right(proposal)
        image = (1 - flip_flag)*image + flip_flag*image_flipped
        obj_bbox = (1 - flip_flag)*obj_bbox + flip_flag*obj_bbox_flipped
        proposal = (1 - flip_flag)*proposal + flip_flag*proposal_flipped
        # image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_brightness(image, max_delta=32./255.)
        # image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
        # image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    else:
      image = decode_jpeg(image_buffer, image_size=image_size)

    images_and_labels.append([image, filenames, sizes, label_index, proposal, roi_labels, obj_labels, obj_bbox])
    (images, filename_batch, size_batch, label_index_batch, proposal_batch,
     roi_label_batch, obj_labels_batch, obj_bbox_batch) = tf.train.batch_join(
            images_and_labels,
            batch_size=batch_size,
            capacity=2 * num_preprocess_threads * batch_size)

    images = tf.cast(images, tf.float32)
    height = image_size
    width = image_size
    depth = 3
    images = tf.reshape(images, shape=[batch_size, height, width, depth])
    label_batch = tf.reshape(label_index_batch, [batch_size, num_classes])

    return (images, filename_batch, size_batch, label_batch, proposal_batch,
            roi_label_batch, obj_labels_batch, obj_bbox_batch)
