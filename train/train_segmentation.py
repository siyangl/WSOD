from __future__ import absolute_import
import math

import tensorflow as tf

import data.seg.image_processing as image_processing
from network.deeplab import deeplab

import data.datasets as datasets
import train.learning as learning

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
tf.app.flags.DEFINE_integer('batch_size', 16, 'Batch size.')
tf.app.flags.DEFINE_integer('max_steps', 8000, 'Training steps.')
tf.app.flags.DEFINE_string('pretrained_model', None, 'Initialize with pretrained checkpoint weights')
tf.app.flags.DEFINE_string('checkpoint', None, 'Continue training from previous checkpoint')
tf.app.flags.DEFINE_string('train_dir', './output/debug', 'Training directory')

tf.app.flags.DEFINE_string('dataset_name', None, 'Name of dataset')
tf.app.flags.DEFINE_string('image_set', 'trainval', 'Subset for training')
tf.app.flags.DEFINE_string('data_dir', None, 'Dataset directory')
tf.app.flags.DEFINE_integer('num_train_imgs', 7306, 'Train set size')
tf.app.flags.DEFINE_integer('num_test_imgs', None, 'Test set size')
tf.app.flags.DEFINE_integer('image_size', 321, 'Input image size')
tf.app.flags.DEFINE_bool('flip_images', False, 'Flip images on the fly left right')
tf.app.flags.DEFINE_bool('crop_images', False, 'Crop images on the fly, used for classification'
                                               'training only')
tf.app.flags.DEFINE_bool('dropout', True, 'Use dropout as regulizer')
tf.app.flags.DEFINE_float('lr_decay', 0.1, 'Learning rate decay ratio')
tf.app.flags.DEFINE_integer('lr_decay_step', 2000, 'Learning rate decay steps')

tf.app.flags.DEFINE_integer('log_interval', 20, 'Number of step interval for log printing')
tf.app.flags.DEFINE_integer('summary_interval', 100, 'Number of step interval for summary writing')
tf.app.flags.DEFINE_integer('snapshot_interval', 4000, 'Number of step interval for model saving')
FLAGS = tf.app.flags.FLAGS


def main(unused):
  seg_size = int(math.ceil(FLAGS.image_size/8.))

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

      images, _, _, labels, roi_labels, segs, seg_masks = \
        image_processing.batch_inputs(dataset,
                                      image_size=FLAGS.image_size,
                                      flip_image=FLAGS.flip_images,
                                      train=True,
                                      batch_size=FLAGS.batch_size)
      segs = tf.image.resize_images(segs, seg_size, seg_size, method=1)
      seg_masks = tf.image.resize_images(seg_masks, seg_size, seg_size, method=1)
      labels = tf.to_float(labels)
      labels_easy = tf.maximum(labels, 0)


    with tf.device('/gpu:0'):
      # Build nets
      seg_preds, _ = deeplab(images, num_classes=dataset.num_classes(),
                             train=True, dropout=FLAGS.dropout)

      seg_preds_reshape = tf.reshape(seg_preds, [-1, dataset.num_classes()])
      segs_onehot = tf.one_hot(segs, dataset.num_classes(), on_value=1.0, off_value=0.0)
      segs_reshape = tf.reshape(segs_onehot, [-1, dataset.num_classes()])
      seg_weights_reshape = tf.reshape(seg_masks, [FLAGS.batch_size, -1])
      cue_loss = tf.nn.softmax_cross_entropy_with_logits(seg_preds_reshape, segs_reshape) #[b*h*w]
      cue_loss = tf.reshape(cue_loss, [FLAGS.batch_size, -1])
      cue_loss = cue_loss*tf.to_float(seg_weights_reshape)
      cue_loss = tf.reduce_sum(cue_loss)/tf.to_float(tf.reduce_sum(seg_masks))

      max_seg_pred = tf.reduce_max(tf.reshape(seg_preds,
                                              [FLAGS.batch_size, -1, dataset.num_classes()]), [1])
      cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(max_seg_pred[:, 1:],
                                                         tf.to_float(labels_easy[:, 1:]))
      cls_loss_pos = tf.reduce_sum(cls_loss*tf.to_float(labels_easy[:, 1:]), [1]) / \
                     tf.to_float(tf.reduce_sum(labels_easy, [1]))
      labels_neg = tf.logical_and(tf.less(labels_easy, 1), tf.greater(labels, -1))
      labels_neg = tf.to_float(labels_neg)
      cls_loss_neg = tf.reduce_sum(cls_loss*labels_neg[:, 1:], [1]) / \
                     tf.to_float(tf.reduce_sum(labels_neg, [1]))
      cls_loss = tf.reduce_mean(cls_loss_pos + cls_loss_neg)

      reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      total_loss = tf.add_n(reg_loss) + cls_loss + cue_loss

      variables_to_train = tf.trainable_variables()
      # variables_to_train = [v for v in all_trainable_variables
      #                       if not (v.name.startswith('vgg_16/conv1') or v.name.startswith('vgg_16/conv2'))]

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
    with tf.device('/cpu:0'):
      tf.scalar_summary('learning rate', learning_rate)
      tf.scalar_summary('Cue Loss', cue_loss)
      tf.scalar_summary('Cls Loss', cls_loss)
      tf.scalar_summary('Total loss', total_loss)
      tf.scalar_summary('global step', global_step)
      tf.image_summary('Input', images)
      tf.image_summary('seg_weights', tf.to_float(seg_masks))
      seg_pred_labels = tf.expand_dims(tf.argmax(seg_preds, 3), 3)
      seg_combined = tf.concat(2, [tf.to_float(segs), tf.to_float(seg_pred_labels)])
      tf.image_summary('seg_pred', tf.to_float(seg_combined))

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
