from __future__ import absolute_import
import tensorflow as tf

import network.deeplab as deeplab

layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope


# conv layers
def vgg_conv(inputs, endpoints, weight_decay=0.0005):
  with arg_scope([layers.convolution2d], rate=1, padding='SAME',
                 weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                 weights_regularizer=layers.l2_regularizer(weight_decay)):
    net = layers.convolution2d(inputs, 64, [3, 3], scope='vgg_16/conv1/conv1_1')
    net = layers.convolution2d(net, 64, [3, 3], scope='vgg_16/conv1/conv1_2')
    endpoints['conv1'] = net
    net = layers.max_pool2d(net, [2, 2], scope='vgg_16/pool1')
    endpoints['pool1'] = net
    net = layers.convolution2d(net, 128, [3, 3], scope='vgg_16/conv2/conv2_1')
    net = layers.convolution2d(net, 128, [3, 3], scope='vgg_16/conv2/conv2_2')
    endpoints['conv2'] = net
    net = layers.max_pool2d(net, [2, 2], scope='vgg_16/pool2')
    endpoints['pool2'] = net
    net = layers.convolution2d(net, 256, [3, 3], scope='vgg_16/conv3/conv3_1')
    net = layers.convolution2d(net, 256, [3, 3], scope='vgg_16/conv3/conv3_2')
    net = layers.convolution2d(net, 256, [3, 3], scope='vgg_16/conv3/conv3_3')
    endpoints['conv3'] = net
    net = layers.max_pool2d(net, [2, 2], scope='vgg_16/pool3')
    endpoints['pool3'] = net
    net = layers.convolution2d(net, 512, [3, 3], scope='vgg_16/conv4/conv4_1')
    net = layers.convolution2d(net, 512, [3, 3], scope='vgg_16/conv4/conv4_2')
    net = layers.convolution2d(net, 512, [3, 3], scope='vgg_16/conv4/conv4_3')
    endpoints['conv4'] = net
    net = layers.max_pool2d(net, [2, 2], scope='vgg_16/pool4')
    endpoints['pool4'] = net
    net = layers.convolution2d(net, 512, [3, 3], scope='vgg_16/conv5/conv5_1')
    net = layers.convolution2d(net, 512, [3, 3], scope='vgg_16/conv5/conv5_2')
    net = layers.convolution2d(net, 512, [3, 3], scope='vgg_16/conv5/conv5_3')
    endpoints['conv5'] = net
  return net, endpoints


def vgg_feature(inputs, endpoints, train=False, dropout=False, weight_decay=0.0005):
  if len(inputs.get_shape().as_list()) > 2:
    inputs = layers.flatten(inputs)
  with arg_scope([layers.fully_connected],
                 weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                 weights_regularizer=layers.l2_regularizer(weight_decay)):
    net = layers.fully_connected(inputs, 4096, scope='vgg_16/fc6')
    endpoints['fc6'] = net
    if train and dropout:
      net = tf.nn.dropout(net, 0.5, name='dropout6')
    net = layers.fully_connected(net, 4096, scope='vgg_16/fc7')
    endpoints['fc7'] = net
    if train and dropout:
      net = tf.nn.dropout(net, 0.5, name='dropout7')
  return net, endpoints


# input size: 7*7*512
def vgg_cls(inputs, num_classes, endpoints, train=False, dropout=False, weight_decay=0.0005):
  net, endpoints = vgg_feature(inputs, endpoints, train=train, dropout=dropout, weight_decay=weight_decay)
  with arg_scope([layers.fully_connected],
                 weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                 weights_regularizer=layers.l2_regularizer(weight_decay)):
    net = layers.fully_connected(net, num_classes, activation_fn=None, scope='vgg_16/fc8')
    endpoints['fc8'] = net
  return net, endpoints


# standard vgg architecture
def vgg_std(inputs, num_classes, pool_stride=2, train=False, dropout=False, weight_decay=0.0005):
  img_mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
  inputs = inputs - img_mean
  endpoints = {}
  features, endpoints = vgg_conv(inputs, endpoints)
  # features = layers.avg_pool2d(features, [pool_stride, pool_stride], stride=pool_stride, scope='vgg_16/pool5')
  features = layers.max_pool2d(features, [pool_stride, pool_stride], stride=pool_stride, scope='vgg_16/pool5')
  endpoints['pool5'] = features
  logits, endpoints = vgg_cls(features, num_classes, endpoints, train=train, dropout=dropout, weight_decay=weight_decay)
  return logits, endpoints


def vgg_fcn(inputs, num_classes, pool_stride=41, train=False, dropout=False, weight_decay=0.0005):
  img_mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
  inputs = inputs - img_mean
  endpoints = {}
  features, endpoints = deeplab.vgg_conv_dilation(inputs, endpoints, weight_decay=weight_decay)

  with arg_scope([layers.convolution2d, layers.max_pool2d, layers.avg_pool2d], padding='SAME'):
    with arg_scope([layers.convolution2d], rate=1,
                   weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                   weights_regularizer=layers.l2_regularizer(weight_decay)):
      net = layers.max_pool2d(features, [3, 3], 1, scope='vgg_16/pool5')
      net = layers.avg_pool2d(net, [3, 3], 1, scope='vgg_16/pool5a')
      endpoints['pool5'] = net
      net = layers.convolution2d(net, 1024, [3, 3], rate=12, scope='vgg_16/fc6')
      endpoints['fc6'] = net
      if dropout and train:
        net = tf.nn.dropout(net, 0.5)
      net = layers.convolution2d(net, 1024, [1, 1], scope='vgg_16/fc7')
      endpoints['fc7'] = net
      if dropout and train:
        net = tf.nn.dropout(net, 0.5)
      net = layers.avg_pool2d(net, [pool_stride, pool_stride], padding='VALID', scope='vgg_16/fc7/avg_pool')
      endpoints['fc7_pool'] = net
      net = layers.convolution2d(net, num_classes, [1, 1], activation_fn=None, scope='vgg_16/fc8')
      net = layers.flatten(net)
      endpoints['fc8'] = net
  return net, endpoints

# # conv layers
# def vgg_conv_multiout(inputs):
#   with slim.scopes.arg_scope([slim.ops.conv2d, slim.ops.fc], stddev=0.01, weight_decay=0.0005):
#     net = slim.ops.conv2d(inputs, 64, [3, 3], scope='vgg_16/conv1/conv1_1')
#     net = slim.ops.conv2d(net, 64, [3, 3], scope='vgg_16/conv1/conv1_2')
#     pool1 = slim.ops.max_pool(net, [2, 2], scope='vgg_16/pool1')
#     net = slim.ops.conv2d(pool1, 128, [3, 3], scope='vgg_16/conv2/conv2_1')
#     net = slim.ops.conv2d(net, 128, [3, 3], scope='vgg_16/conv2/conv2_2')
#     pool2 = slim.ops.max_pool(net, [2, 2], scope='vgg_16/pool2')
#     net = slim.ops.conv2d(pool2, 256, [3, 3], scope='vgg_16/conv3/conv3_1')
#     net = slim.ops.conv2d(net, 256, [3, 3], scope='vgg_16/conv3/conv3_2')
#     net = slim.ops.conv2d(net, 256, [3, 3], scope='vgg_16/conv3/conv3_3')
#     pool3 = slim.ops.max_pool(net, [2, 2], scope='vgg_16/pool3')
#     net = slim.ops.conv2d(pool3, 512, [3, 3], scope='vgg_16/conv4/conv4_1')
#     net = slim.ops.conv2d(net, 512, [3, 3], scope='vgg_16/conv4/conv4_2')
#     net = slim.ops.conv2d(net, 512, [3, 3], scope='vgg_16/conv4/conv4_3')
#     pool4 = slim.ops.max_pool(net, [2, 2], scope='vgg_16/pool4')
#     net = slim.ops.conv2d(pool4, 512, [3, 3], scope='vgg_16/conv5/conv5_1')
#     net = slim.ops.conv2d(net, 512, [3, 3], scope='vgg_16/conv5/conv5_2')
#     pool5 = slim.ops.conv2d(net, 512, [3, 3], scope='vgg_16/conv5/conv5_3')
#   return pool5, pool4, pool3, pool2, pool1
