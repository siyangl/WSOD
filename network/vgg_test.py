from __future__ import absolute_import
import cv2
import os
import numpy as np
import tensorflow as tf

from network.vgg import vgg_std
from cfg import ROOT_DIR


def main(unused):
  model_path = os.path.join(ROOT_DIR, 'data', 'imagenet_models', 'VGG16.ckpt')
  im = cv2.imread(os.path.join(ROOT_DIR, 'network', 'laska.png'))
  im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC)
  im = np.reshape(im, (1, 224, 224, 3))
  im = im[:, :, :, (2, 1, 0)]

  g = tf.Graph()
  with g.as_default():
    x = tf.placeholder(tf.float32, [1, 224, 224, 3])
    logits, endpoints = vgg_std(x, 1000)
    prob = tf.nn.softmax(logits)
    for v in tf.trainable_variables():
      print v.name
    assert len(endpoints.keys()) == 13
    saver = tf.train.Saver(tf.trainable_variables())
    with tf.Session() as sess:
      saver.restore(sess, model_path)
      [scores_out] = sess.run([prob], feed_dict={x: im})
      scores = scores_out[0]
      assert scores.shape[0] == 1000
      preds = (np.argsort(scores)[::-1])[0:5]
      assert preds[0] == 356
      # for p in preds:
      #   print scores[p]
      # print preds[0:5]
      print "Passed"


if __name__ == '__main__':
  tf.app.run()