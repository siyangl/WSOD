import cv2
import numpy as np
import tensorflow as tf

import image_processing
from data.datasets import Esos, Pascal
from utils.visualizer import visualize_boxes

tf.app.flags.DEFINE_bool('is_training', False, 'The input data is shuffled if training')
tf.app.flags.DEFINE_bool('flip_images', False, 'Flip images on the fly left right')
tf.app.flags.DEFINE_bool('crop_images', False, 'Crop images on the fly, used for classification'
                                               'training only')
tf.app.flags.DEFINE_bool('visualize_bboxes', True, 'Visualize boxes')
FLAGS = tf.app.flags.FLAGS


def main(unused):
  g = tf.Graph()
  with g.as_default(), tf.device('/cpu:0'):
    # dataset = EsosData(subset='train',
    #                    datadir='./data/bbox/ESOS',
    #                    num_train_imgs=10966,
    #                    num_test_imgs=2741)
    dataset = Pascal(subset='trainval',
                         datadir='./data/bbox/VOC2007selected0410_0000',
                         num_train_imgs=3328,
                         num_test_imgs=5011)
    images, filenames, sizes, labels, bbox, roi_labels, gt_labels, gt = \
      image_processing.batch_inputs(dataset,
                                    train=FLAGS.is_training,
                                    flip_image=FLAGS.flip_images,
                                    crop_image=FLAGS.crop_images,
                                    image_size=400,
                                    batch_size=1)
    valid_rois = tf.reduce_sum(tf.to_float(tf.greater(bbox[:, :, 2]-bbox[:, :, 0], 1e-6)))
    with tf.Session() as sess:
      tf.train.start_queue_runners(sess=sess)
      for step in range(10):
        print(step)
        images_out, filenames_out, sizes_out, roi_labels_out, gt_labels_out, gts_out, bbox_out, labels_out, valid_rois_out = \
          sess.run([images, filenames, sizes, roi_labels, gt_labels, gt, bbox, labels, valid_rois])
        print(filenames_out)
        print(labels_out)
        print(sizes_out)
        for i in range(len(dataset.category_list())):
          if labels_out[0, i] > 0.5:
            print(dataset.category_list()[i])
        for i in range(len(dataset.category_list())):
          if roi_labels_out[0, i] > 0.5:
            print(dataset.category_list()[i])
        print(bbox_out[0, 100])
        print(gt_labels_out)
        print(bbox_out[:, 0:5, :])
        print('num rois: %d'%valid_rois_out)

        gt_out = (400 * gts_out[0]).astype(np.int32)
        bbox_out = (400 * bbox_out[0]).astype(np.int32)
        gt_vis = visualize_boxes(images_out[0], gt_out)
        roi_vis = visualize_boxes(images_out[0], bbox_out[:10, :])
        cv2.imshow('img', gt_vis)
        cv2.imshow('rois', roi_vis)
        cv2.waitKey()


if __name__ == "__main__":
  tf.app.run()
