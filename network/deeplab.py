import tensorflow as tf

layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope


# conv layers
def vgg_conv_dilation(inputs, endpoints, weight_decay=0.0005):
  with arg_scope([layers.convolution2d, layers.max_pool2d], padding='SAME'):
    with arg_scope([layers.convolution2d], rate=1,
                   weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                   weights_regularizer=layers.l2_regularizer(weight_decay)):
      net = layers.convolution2d(inputs, 64, [3, 3], scope='vgg_16/conv1/conv1_1')
      net = layers.convolution2d(net, 64, [3, 3], scope='vgg_16/conv1/conv1_2')
      endpoints['conv1'] = net
      net = layers.max_pool2d(net, [3, 3], scope='vgg_16/pool1')
      endpoints['pool1'] = net
      net = layers.convolution2d(net, 128, [3, 3], scope='vgg_16/conv2/conv2_1')
      net = layers.convolution2d(net, 128, [3, 3], scope='vgg_16/conv2/conv2_2')
      endpoints['conv2'] = net
      net = layers.max_pool2d(net, [3, 3], scope='vgg_16/pool2')
      endpoints['pool2'] = net
      net = layers.convolution2d(net, 256, [3, 3], scope='vgg_16/conv3/conv3_1')
      net = layers.convolution2d(net, 256, [3, 3], scope='vgg_16/conv3/conv3_2')
      net = layers.convolution2d(net, 256, [3, 3], scope='vgg_16/conv3/conv3_3')
      endpoints['conv3'] = net
      net = layers.max_pool2d(net, [3, 3], scope='vgg_16/pool3')
      endpoints['pool3'] = net
      net = layers.convolution2d(net, 512, [3, 3], scope='vgg_16/conv4/conv4_1' )
      net = layers.convolution2d(net, 512, [3, 3], scope='vgg_16/conv4/conv4_2' )
      net = layers.convolution2d(net, 512, [3, 3], scope='vgg_16/conv4/conv4_3' )
      endpoints['conv4'] = net
      net = layers.max_pool2d(net, [3, 3], 1, scope='vgg_16/pool4')
      endpoints['pool4'] = net
      net = layers.convolution2d(net, 512, [3, 3], rate=2, scope='vgg_16/conv5/conv5_1' )
      net = layers.convolution2d(net, 512, [3, 3], rate=2, scope='vgg_16/conv5/conv5_2' )
      net = layers.convolution2d(net, 512, [3, 3], rate=2, scope='vgg_16/conv5/conv5_3' )
      endpoints['conv5'] = net
  return net, endpoints


def deeplab_top(inputs, endpoints, num_classes, train=False, dropout=False, weight_decay=0.0005):
  with arg_scope([layers.convolution2d, layers.max_pool2d, layers.avg_pool2d], padding='SAME'):
    with arg_scope([layers.convolution2d], rate=1,
                   weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                   weights_regularizer=layers.l2_regularizer(weight_decay)):
      net = layers.max_pool2d(inputs, [3, 3], 1, scope='vgg_16/pool5')
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
      net = layers.convolution2d(net, num_classes, [1, 1], activation_fn=None, scope='vgg_16/fc8')
      endpoints['fc8'] = net
    return net, endpoints


def deeplab(inputs, num_classes, train=False, dropout=False, weight_decay=0.0005):
  img_mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
  inputs = inputs - img_mean
  endpoints = {}
  feature, endpoints = vgg_conv_dilation(inputs, endpoints, weight_decay=weight_decay)
  seg, endpoints = deeplab_top(feature, endpoints,
                               num_classes=num_classes,
                               train=train, dropout=dropout,
                               weight_decay=weight_decay)
  return seg, endpoints
