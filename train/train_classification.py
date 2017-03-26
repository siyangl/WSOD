from __future__ import absolute_import
import numpy as np

import tensorflow as tf

import data.datasets as datasets
import data.bbox.image_processing as image_processing
import network.vgg as vgg
import train.learning as learning

tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
tf.app.flags.DEFINE_integer('batch_size', 16, 'Batch size.')
tf.app.flags.DEFINE_integer('max_steps', 10000, 'Training steps.')
tf.app.flags.DEFINE_string('pretrained_model', None, 'Initialize with pretrained checkpoint weights')
tf.app.flags.DEFINE_string('checkpoint', None, 'Continue training from previous checkpoint')
tf.app.flags.DEFINE_string('train_dir', './output/debug', 'Training directory')

tf.app.flags.DEFINE_string('dataset_name', None, 'Name of dataset')
tf.app.flags.DEFINE_string('image_set', 'trainval', 'Subset for training')
tf.app.flags.DEFINE_string('data_dir', None, 'Dataset directory')
tf.app.flags.DEFINE_integer('num_train_imgs', None, 'Train set size')
tf.app.flags.DEFINE_integer('num_test_imgs', None, 'Test set size')
tf.app.flags.DEFINE_integer('image_size', 224, 'Input image size')
tf.app.flags.DEFINE_bool('flip_images', False, 'Flip images on the fly left right')
tf.app.flags.DEFINE_bool('crop_images', False, 'Crop images on the fly, used for classification'
                                               'training only')
tf.app.flags.DEFINE_string('network', 'vgg_std', 'vgg_std or vgg_fcn')
tf.app.flags.DEFINE_bool('multilabel', False, 'The dataset is a multilabel dataset')
tf.app.flags.DEFINE_bool('dropout', True, 'Use dropout as regulizer')
tf.app.flags.DEFINE_float('lr_decay', -1, 'Learning rate decay ratio')
tf.app.flags.DEFINE_integer('lr_decay_step', -1, 'Learning rate decay steps')

tf.app.flags.DEFINE_integer('log_interval', 20, 'Number of step interval for log printing')
tf.app.flags.DEFINE_integer('summary_interval', 100, 'Number of step interval for summary writing')
tf.app.flags.DEFINE_integer('snapshot_interval', 10000, 'Number of step interval for model saving')
FLAGS = tf.app.flags.FLAGS


def main(unused):
  g = tf.Graph()
  with g.as_default():
    with tf.device('/cpu:0'):
      # dataset
      dataset_to_call = getattr(datasets, FLAGS.dataset_name)
      dataset = dataset_to_call(subset=FLAGS.image_set,
                                datadir=FLAGS.data_dir,
                                num_train_imgs=FLAGS.num_train_imgs,
                                num_test_imgs=FLAGS.num_test_imgs)
      assert dataset.data_files()

      images, _, _, labels, _, _, _, _ = image_processing.batch_inputs(dataset,
                                                                       image_size=FLAGS.image_size,
                                                                       flip_image=FLAGS.flip_images,
                                                                       crop_image=FLAGS.crop_images,
                                                                       train=True,
                                                                       batch_size=FLAGS.batch_size)
      labels = tf.to_float(labels)

    network_to_call = getattr(vgg, FLAGS.network)
    with tf.device('/gpu:0'):
      # Build nets
      logits, _ = network_to_call(images, num_classes=dataset.num_classes(),
                                  train=True, dropout=FLAGS.dropout)
      if not FLAGS.multilabel:
        cls_loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
      else:
        bg_mask = np.expand_dims(np.concatenate([[0], np.ones((dataset.num_classes() - 1))]), 0)
        valid_labels = tf.to_float(tf.greater(labels, -1))
        valid_labels = tf.minimum(valid_labels,
                                  tf.constant(bg_mask, dtype=tf.float32))
        cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels) * valid_labels
      cls_loss = tf.reduce_mean(cls_loss)
      reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      total_loss = tf.add_n(reg_loss) + cls_loss

      all_trainable_variables = tf.trainable_variables()
      variables_to_train = [v for v in all_trainable_variables
                            if not (v.name.startswith('vgg_16/conv1') or v.name.startswith('vgg_16/conv2'))]

    # learning rate and optimizer
    global_step = tf.get_variable('global_step', shape=[], dtype=tf.int64,
                                  initializer=tf.zeros_initializer,
                                  trainable=False)

    if FLAGS.lr_decay > 0 and FLAGS.lr_decay_step > 0:
      learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                                 global_step,
                                                 FLAGS.lr_decay_step,
                                                 FLAGS.lr_decay,
                                                 staircase=True)
    else:
      learning_rate = FLAGS.learning_rate
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)

    # Summary
    tf.scalar_summary('learning rate', learning_rate)
    tf.scalar_summary('Cls Loss', cls_loss)
    tf.scalar_summary('Total loss', total_loss)
    tf.scalar_summary('global step', global_step)
    tf.image_summary('Input', images, max_images=5)

    variables_to_restore = None
    restore_ckpt = None
    if FLAGS.checkpoint:
      variables_to_restore = tf.trainable_variables
      restore_ckpt = FLAGS.checkpoint
    elif FLAGS.pretrained_model:
      variables_to_restore = [v for v in tf.trainable_variables()
                              if not v.name.startswith('vgg_16/fc8')]
      restore_ckpt = FLAGS.pretrained_model

    learning.train(g, FLAGS.train_dir,
                   total_loss, optimizer,
                   variables_to_train, global_step,
                   num_steps=FLAGS.max_steps, log_interval=FLAGS.log_interval,
                   summary_interval=FLAGS.summary_interval, snapshot_interval=FLAGS.snapshot_interval,
                   variables_to_restore=variables_to_restore, restore_ckpt=restore_ckpt)



if __name__ == '__main__':
  assert FLAGS.dataset_name, 'Must provide a dataset'

  if FLAGS.pretrained_model and FLAGS.checkpoint:
    print('Checkpoint will override pretrained_file!')
  if not FLAGS.pretrained_model and not FLAGS.checkpoint:
    print('Training from scratch!')
  tf.app.run()
