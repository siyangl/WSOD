from __future__ import absolute_import
import tensorflow as tf

import network.vgg as vgg


def rcnn(inputs, rois, num_classes, train=False, dropout=False, weight_decay=0.0005):
  """
  :param inputs: [b, h, w, c]
  :param rois: [b, #boxes, 4] same #roi for each image
  :param num_classes: scalar
  :param train:
  :param dropout:
  :param weight_decay:
  :return:
  """
  img_mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
  inputs = inputs - img_mean
  endpoints = {}
  features, endpoints = vgg.vgg_conv(inputs, endpoints)

  # roi pool
  batch_size = inputs.get_shape().as_list()[0]
  num_rois = rois.get_shape().as_list()[1]
  batch_rois = tf.reshape(rois, [batch_size*num_rois, 4])
  batch_roi_index = tf.reshape(
          tf.tile(tf.expand_dims(tf.range(batch_size), 0), [num_rois, 1]),
          [batch_size*num_rois])
  roi_features = tf.image.crop_and_resize(features, batch_rois, batch_roi_index, [7, 7])
  endpoints['roi_pool'] = roi_features
  logits, endpoints = vgg.vgg_cls(roi_features,
                                  num_classes,
                                  endpoints,
                                  train=train,
                                  dropout=dropout,
                                  weight_decay=weight_decay)
  logits = tf.reshape(logits, [batch_size, num_rois, num_classes])
  return logits, endpoints