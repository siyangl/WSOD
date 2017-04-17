import numpy as np


def generate_anchor_boxes(anchor_centers, anchor_sizes):
  if anchor_centers.shape[1] != 2:
    raise ValueError('anchor_centers should contain center_h, center_w')
  if anchor_sizes.shape[1] != 2:
    raise ValueError('anchor_sizes should contain size_h, size_w.')
  anchor_centers = anchor_centers.astype(float)
  anchor_sizes = anchor_sizes.astype(float)
  num_anchor_centers = anchor_centers.shape[0]
  num_anchor_sizes = anchor_sizes.shape[0]
  num_anchors = num_anchor_centers * num_anchor_sizes
  anchor_centers_dup = np.tile(anchor_centers, [1, num_anchor_sizes])
  anchor_centers_dup = np.reshape(anchor_centers_dup, [num_anchors, 2])
  anchor_sizes_dup = np.tile(anchor_sizes, [num_anchor_centers, 1])
  anchor_sizes_dup = np.reshape(anchor_sizes_dup, [num_anchors, 2])
  anchors = np.hstack([anchor_centers_dup, anchor_sizes_dup])
  return anchors


def generate_anchor_centers(centers_h, centers_w):
  """
  Example: centers_h = [1, 3, 5]
           centers_w = [2, 4]
           anchor_centers = [[1, 2], [1, 4], [3, 2], [3, 4], [5, 2], [5, 4]]
  centers_h: np array with shape [num_ch]
  centers_w: np array with shape [num_cw]
  returns:
    anchor_centers: shape [num_ch*nu_cw, 2]
  """
  centers_h = centers_h.astype(np.float)
  centers_w = centers_w.astype(np.float)
  num_ch = centers_h.shape[0]
  num_cw = centers_w.shape[0]
  dup_centers_h = np.tile(np.reshape(centers_h, [num_ch, 1]), [1, num_cw])
  dup_centers_w = np.tile(np.reshape(centers_w, [1, num_cw]), [num_ch, 1])
  anchor_centers = np.dstack((dup_centers_h, dup_centers_w))
  anchor_centers = np.reshape(anchor_centers, [num_ch * num_cw, 2])
  return anchor_centers


def generate_batched_anchors(anchors, batch_size):
  """
  Depulicate anchors by batch_size times
  anchors [num_anchors/img, 4]
  return batched anchors [batch_size * num_anchors/img, 4]
  batch_anchor_index: indicate the ith anchor belong to which batch [batch_size*num_anchors/img]
  """
  num_anchors = anchors.shape[0]
  batch_anchors = np.tile(anchors, (batch_size, 1))
  batch_anchor_indices = np.tile(np.arange(batch_size), (num_anchors, 1))
  batch_anchor_indices = np.transpose(batch_anchor_indices)
  batch_anchor_indices = np.reshape(batch_anchor_indices, num_anchors * batch_size)
  return batch_anchors, batch_anchor_indices


def generate_anchors_from_density_and_sizes(density, sizes):
  """
  1. generate anchors
  2. transform anchors to [hmin, wmin, hmax, wmax]
  3. Crop anchors to image range"""
  stride = 1. / density
  centers_h = np.arange(density, dtype=np.float) * stride + stride / 2
  centers_w = np.arange(density, dtype=np.float) * stride + stride / 2
  centers = generate_anchor_centers(centers_h, centers_w)
  anchors = generate_anchor_boxes(centers, sizes)
  # bbox transform
  anchors = bbox_transform(anchors)
  anchors = crop_bbox_to_image_range(anchors)
  return anchors


def crop_bbox_to_image_range(anchors):
  """
  :param anchors: in normalized coordinate
  :return: clippe anchors
  """
  anchors = np.maximum(anchors, 0.)
  anchors = np.minimum(anchors, 1.)
  return anchors


def bbox_transform(bboxes):
  """[h_c, w_c, h, w] to [hmin, wmin, hmax, wmax]"""
  assert bboxes.shape[1] == 4
  center_h, center_w, size_h, size_w = np.hsplit(bboxes, 4)
  top = center_h - size_h / 2
  bottom = center_h + size_h / 2
  left = center_w - size_w / 2
  right = center_w + size_w / 2
  bboxes_transformed = np.hstack([top, left, bottom, right])
  return bboxes_transformed


def bbox_inv_transform(bboxes):
  """[hmin, wmin, hmax, wmax] to [h_c, w_c, h, w]"""
  assert bboxes.shape[1] == 4
  bboxes = bboxes.astype(np.float)
  top, left, bottom, right = np.hsplit(bboxes, 4)
  center_h = (top + bottom) / 2
  center_w = (left + right) / 2
  size_h = (bottom - top + 1)
  size_w = (right - left + 1)
  bboxes_inv_transformed = np.hstack([center_h, center_w, size_h, size_w])
  return bboxes_inv_transformed


def point_in_box(point, bboxes, tolerance=0.):
  """
  params:
      point: [h, w] normalized coordinates, shape [2]
      bboxes: [hmin, wmin, hmax, wmax] normalized coordinates, shape [num_bboxes, 4]
      tolerance: if the point is with this distance to the bboxes, also consider as in the box
  returns:
      is_in, 1-D bool array [num_bboxes]
  """
  bboxes_enlarged = bboxes
  bboxes_enlarged[:, :2] = bboxes[:, :2] - tolerance
  bboxes_enlarged[:, 2:] = bboxes[:, 2:] + tolerance
  is_in = np.logical_and(
          np.logical_and(np.greater(point[0], bboxes_enlarged[:, 0]),
                         np.less(point[0], bboxes_enlarged[:, 2])),
          np.logical_and(np.greater(point[1], bboxes_enlarged[:, 1]),
                         np.less(point[1], bboxes_enlarged[:, 3])))
  return is_in


def bbox_unnormalize(bboxes, image_size, one_index=False):
  """
  Unnormalized bboxes
  :param bboxes: [hmin, wmin, hmax, wmax] [#bboxes, 4]
  :param image_size: [h, w]
  :param one_index: bool, convert box to 1-index
  :return: unnormalized bbox
  """

  image_size = np.reshape(image_size, (1, 2))
  size_factor = np.tile(image_size, (1, 2))
  unnormalized_bboxes = bboxes * (size_factor - 1)
  if one_index:
    unnormalized_bboxes += 1
  return unnormalized_bboxes


def bbox_normalize(bboxes, image_size, one_index=False):
  """
  Unnormalized bboxes
  :param bboxes: [hmin, wmin, hmax, wmax] [#bboxes, 4]
  :param image_size: [h, w]
  :param one_index: bool, convert box to 1-index
  :return: unnormalized bbox
  """
  bboxes = bboxes.astype(np.float)
  image_size = image_size.astype(np.float)
  image_size = np.reshape(image_size, (1, 2))
  size_factor = np.tile(image_size, (1, 2))

  if one_index:
    normalized_bboxes = (bboxes - 1) / (size_factor - 1)
  else:
    normalized_bboxes = bboxes / (size_factor - 1)
  return normalized_bboxes


def get_iou_matrix(gt, det):
  gt = gt.astype(np.float)
  det = det.astype(np.float)
  iou_matrix = np.zeros((gt.shape[0], det.shape[0]))
  for g in range(gt.shape[0]):
    gt_dup = np.tile(gt[g], (det.shape[0], 1))
    ymin = np.maximum(gt_dup[:, 0], det[:, 0])
    xmin = np.maximum(gt_dup[:, 1], det[:, 1])
    ymax = np.minimum(gt_dup[:, 2], det[:, 2])
    xmax = np.minimum(gt_dup[:, 3], det[:, 3])
    overlap = np.maximum(0, ymax - ymin) * np.maximum(0, xmax - xmin)
    union = (gt_dup[:, 2] - gt_dup[:, 0]) * (gt_dup[:, 3] - gt_dup[:, 1]) + \
            (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]) - overlap
    iou = overlap / union
    iou_matrix[g] = iou
  return iou_matrix


def get_iob_matrix(gt, det):
  gt = gt.astype(np.float)
  det = det.astype(np.float)
  # iou_matrix = np.zeros((gt.shape[0], det.shape[0]))
  iob_matrix1 = np.zeros((gt.shape[0], det.shape[0]))
  iob_matrix2 = np.zeros((gt.shape[0], det.shape[0]))
  for g in range(gt.shape[0]):
    gt_dup = np.tile(gt[g], (det.shape[0], 1))
    ymin = np.maximum(gt_dup[:, 0], det[:, 0])
    xmin = np.maximum(gt_dup[:, 1], det[:, 1])
    ymax = np.minimum(gt_dup[:, 2], det[:, 2])
    xmax = np.minimum(gt_dup[:, 3], det[:, 3])
    overlap = np.maximum(0, ymax - ymin) * np.maximum(0, xmax - xmin)
    # union = (gt_dup[:, 2] - gt_dup[:, 0]) * (gt_dup[:, 3] - gt_dup[:, 1]) + \
    #         (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]) - overlap
    iob1 = overlap / ((gt_dup[:, 2] - gt_dup[:, 0]) * (gt_dup[:, 3] - gt_dup[:, 1]))
    iob2 = overlap / ((det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])+ 1e-6)
    # iou_matrix[g] = iou
    iob_matrix1[g] = iob1
    iob_matrix2[g] = iob2
  return iob_matrix1, iob_matrix2


def boxes_to_seg(boxes, image_size):
  """
  Make box annotation to segmentation annotation
  :param boxes: unnormalized coordinates, zero index, [#boxes, 4], [hmin, wmin, hmax, wmax]
  :param image_size: [h, w]
  :return: box_map: segmentation binary map
  """
  box_map = np.zeros(image_size, dtype=np.bool)
  for b in boxes:
    box_map[b[0]:b[2]+1, b[1]:b[3]+1] = True
  return box_map


def boxes_to_objectness_map(boxes, box_scores, image_size, use_max=False):
  """
  Get objectness map, category-dependent or -independent
  :param boxes: unnormalized coordinates, zero index, list of array [#boxes, 4], [hmin, wmin, hmax, wmax]
  :param box_scores: list [#boxes]
  :param image_size: [h, w]
  :return: box_map: segmentation binary map
  """
  box_map = np.zeros(image_size)
  if use_max:
    for b, s in zip(boxes, box_scores):
      box_map[b[0]:b[2]+1, b[1]:b[3]+1] = np.maximum(box_map[b[0]:b[2]+1, b[1]:b[3]+1], s)
  else:
    for b, s in zip(boxes, box_scores):
      box_map[b[0]:b[2]+1, b[1]:b[3]+1] += s
  box_map = box_map / np.max(box_map)
  return box_map