import numpy as np
import os
from datetime import datetime
import logging
import tensorflow as tf
import math

import data.bbox.image_processing as image_processing
import data.datasets as datasets
import network.vgg as vgg
import metrics

tf.app.flags.DEFINE_string('dataset_name', None, 'Name of dataset')
tf.app.flags.DEFINE_string('data_dir', None, 'Dataset directory')
tf.app.flags.DEFINE_integer('image_size', 224, 'Input image size')
tf.app.flags.DEFINE_integer('batch_size', 1, 'Batch size.')
tf.app.flags.DEFINE_string('checkpoint', None, 'Continue training from previous checkpoint')
tf.app.flags.DEFINE_bool('multilabel', False, 'Multi label dataset')
tf.app.flags.DEFINE_string('network', 'vgg_std', 'vgg_std or vgg_fcn')
FLAGS = tf.app.flags.FLAGS


def main(unused):
  logdir = os.path.dirname(FLAGS.checkpoint)
  log_filename = os.path.join(logdir, 'classification_eval_%s.log' % datetime.now())
  logging.basicConfig(filename=log_filename, level=logging.DEBUG)
  logging.info('Ckpt to be evaluated: %s'%FLAGS.checkpoint)

  g = tf.Graph()
  with g.as_default():
    with tf.device('/cpu:0'):
      # dataset
      dataset_to_call = getattr(datasets, FLAGS.dataset_name)
      dataset = dataset_to_call(subset='test',
                         datadir=FLAGS.data_dir)
      assert dataset.data_files()
      images, _, _, labels, _, _, _, _ = image_processing.batch_inputs(dataset,
                                                                       image_size=FLAGS.image_size,
                                                                       train=False,
                                                                       batch_size=FLAGS.batch_size)
    network_to_call = getattr(vgg, FLAGS.network)
    # Build nets
    if FLAGS.network == 'vgg_std':
      pool_stride = FLAGS.image_size / (16 * 7)
    else:
      pool_stride = math.ceil(FLAGS.image_size / 8.)
    logits, _ = network_to_call(images, num_classes=dataset.num_classes(), pool_stride=pool_stride,
                                train=True, dropout=False)

    if FLAGS.multilabel:
      scores = tf.sigmoid(logits)
    else:
      scores = tf.nn.softmax(logits)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    restorer = tf.train.Saver(tf.all_variables())
    restorer.restore(sess, FLAGS.checkpoint)
    num_iters = dataset.num_examples_per_epoch()
    tf.train.start_queue_runners(sess=sess)
    step = 0
    num_tp = 0.
    num_gt = 0.
    num_obj = 0.
    all_gt = np.zeros((num_iters, FLAGS.batch_size, dataset.num_classes()))
    all_pred = np.zeros((num_iters, FLAGS.batch_size, dataset.num_classes()))
    ap = np.zeros(dataset.num_classes())
    while step < num_iters:
      print 'Eval batch %d/%d' % (step, num_iters)
      scores_out, labels_out = sess.run([scores, labels])
      labels_out = np.maximum(labels_out, 0)
      for cls in range(1, dataset.num_classes()):
        all_gt[step, :, cls] = labels_out[:, cls]
        all_pred[step, :, cls] = scores_out[:, cls]
      scores_out = scores_out[0]
      labels_out = labels_out[0]
      pred = np.argmax(scores_out)
      gt = np.argmax(labels_out)
      if pred == gt:
        num_tp += 1
      num_gt +=1
      num_obj += (gt > 0)
      step += 1
    if FLAGS.multilabel:
      for cls in range(1, dataset.num_classes()):
        all_pred_cls = np.reshape(all_pred[:, :, cls], (FLAGS.batch_size*num_iters))
        all_gt_cls = np.reshape(all_gt[:, :, cls], (FLAGS.batch_size*num_iters))
        ap[cls] = metrics.average_presicion(all_pred_cls, all_gt_cls)
        logging.info("AP for %s : %.3f" %(dataset.category_list()[cls], ap[cls]))
      logging.info("mAP: %.3f" %(np.sum(ap)/(dataset.num_classes() - 1)))
    else:
      logging.info("Accuracy: %.3f"%(num_tp/num_gt))
      logging.info("Num salient objects: %d"%num_obj)


if __name__ == '__main__':
  if not FLAGS.checkpoint:
    print('Must provide a checkpoint!')
  else:
    tf.app.run()