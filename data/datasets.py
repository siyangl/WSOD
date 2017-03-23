from __future__ import absolute_import
import os
import tensorflow as tf

from data.dataset import Dataset


class Pascal(Dataset):
  """Pascal dataset"""

  def __init__(self, subset, datadir, num_train_imgs=None, num_test_imgs=None, contain_flipped=False):
    super(Pascal, self).__init__('Pascal', subset)
    if datadir:
      self.datadir = datadir
    else:
      self.datadir = './data/bbox/VOC2007'
    if num_train_imgs:
      self.num_train_imgs = num_train_imgs
    else:
      self.num_train_imgs = 5011,
    if num_test_imgs:
      self.num_test_imgs = num_test_imgs
    else:
      self.num_test_imgs = 4952
    self.contain_flipped = contain_flipped

  def num_classes(self):
    """Number of classes including background"""
    return len(self.category_list())

  def category_list(self):
    pascal_classes = ['__background__',
                      'aeroplane', 'bicycle', 'bird', 'boat',
                      'bottle', 'bus', 'car', 'cat', 'chair',
                      'cow', 'diningtable', 'dog', 'horse',
                      'motorbike', 'person', 'pottedplant',
                      'sheep', 'sofa', 'train', 'tvmonitor']
    return pascal_classes

  def num_examples_per_epoch(self):
    if self.subset == 'trainval' or self.subset == 'train':
      if self.contain_flipped:
        return self.num_train_imgs * 2
      else:
        return self.num_train_imgs
    else:
      return self.num_test_imgs

  def download_message(self):
    print("Need to download the dataset.")

  def available_subsets(self):
    """Returns the list of available subsets."""
    return ['trainval', 'test']

  def data_files(self):
    """Returns a python list of all (sharded) data subset files.
    Returns:
      python list of all (sharded) data set files.
    Raises:
      ValueError: if there are not data_files matching the subset.
    """
    tf_record_pattern = os.path.join(self.datadir, '%s-*' % self.subset)
    data_files = tf.gfile.Glob(tf_record_pattern)
    if not data_files:
      print('No files found for dataset %s/%s at %s' % (self.name,
                                                        self.subset,
                                                        self.datadir))

      self.download_message()
      exit(-1)
    return data_files


class ImageNet(Dataset):
  """ImageNet dataset"""

  def __init__(self, subset, datadir, num_train_imgs=None, num_test_imgs=None, contain_flipped=False):
    super(ImageNet, self).__init__('ImageNet', subset)
    if datadir:
      self.datadir = datadir
    else:
      self.datadir = './data/bbox/ImageNet'
    if num_train_imgs:
      self.num_train_imgs = num_train_imgs
    else:
      self.num_train_imgs = 15116,
    if num_test_imgs:
      self.num_test_imgs = num_test_imgs
    else:
      self.num_test_imgs = 14333
    self.contain_flipped = contain_flipped

  def num_classes(self):
    """Number of classes including background"""
    return len(self.category_list())

  def category_list(self):
    pascal_classes = ['__background__',
                      'aeroplane', 'bicycle', 'bird', 'boat',
                      'bottle', 'bus', 'car', 'cat', 'chair',
                      'cow', 'diningtable', 'dog', 'horse',
                      'motorbike', 'person', 'pottedplant',
                      'sheep', 'sofa', 'train', 'tvmonitor']
    return pascal_classes

  def num_examples_per_epoch(self):
    if self.subset == 'train' or self.subset == 'val':
      if self.contain_flipped:
        return self.num_train_imgs * 2
      else:
        return self.num_train_imgs
    else:
      return self.num_test_imgs

  def download_message(self):
    print("Need to download the dataset.")

  def available_subsets(self):
    """Returns the list of available subsets."""
    return ['train', 'val']

  def data_files(self):
    """Returns a python list of all (sharded) data subset files.
    Returns:
      python list of all (sharded) data set files.
    Raises:
      ValueError: if there are not data_files matching the subset.
    """
    tf_record_pattern = os.path.join(self.datadir, '%s-*' % self.subset)
    data_files = tf.gfile.Glob(tf_record_pattern)
    if not data_files:
      print('No files found for dataset %s/%s at %s' % (self.name,
                                                        self.subset,
                                                        self.datadir))

      self.download_message()
      exit(-1)
    return data_files


class Esos(Dataset):
  """Saliency dataset"""

  def __init__(self, subset, datadir, num_train_imgs=None, num_test_imgs=None, contain_flipped=False):
    super(Esos, self).__init__('Esos', subset)
    if datadir:
      self.datadir = datadir
    else:
      self.datadir = './data/bbox/ESOS'
    if num_train_imgs:
      self.num_train_imgs = num_train_imgs
    else:
      self.num_train_imgs = 10966,
    if num_test_imgs:
      self.num_test_imgs = num_test_imgs
    else:
      self.num_test_imgs = 2741
    self.contain_flipped = contain_flipped

  def num_classes(self):
    """Number of classes including background"""
    return len(self.category_list())

  def category_list(self):
    saliency_classes = ['scene', 'object']
    return saliency_classes

  def num_examples_per_epoch(self):
    if self.subset == 'trainval' or self.subset == 'train':
      if self.contain_flipped:
        return self.num_train_imgs * 2
      else:
        return self.num_train_imgs
    else:
      return self.num_test_imgs

  def download_message(self):
    print("Need to download the dataset.")

  def available_subsets(self):
    """Returns the list of available subsets."""
    return ['train', 'test']

  def data_files(self):
    """Returns a python list of all (sharded) data subset files.
    Returns:
      python list of all (sharded) data set files.
    Raises:
      ValueError: if there are not data_files matching the subset.
    """
    tf_record_pattern = os.path.join(self.datadir, '%s-*' % self.subset)
    data_files = tf.gfile.Glob(tf_record_pattern)
    if not data_files:
      print('No files found for dataset %s/%s at %s' % (self.name,
                                                        self.subset,
                                                        self.datadir))

      self.download_message()
      exit(-1)
    return data_files