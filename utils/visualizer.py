import png
import cv2
import numpy as np

palette = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0),
           (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
           (0.25, 0.0, 0.0), (0.75, 0.0, 0.0), (0.25, 0.5, 0.0), (0.75, 0.5, 0.0),
           (0.25, 0.0, 0.5), (0.75, 0.0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5),
           (0.0, 0.25, 0.0), (0.5, 0.25, 0.0), (0.0, 0.75, 0.0), (0.5, 0.75, 0.0),
           (0.0, 0.25, 0.5)]


def write_to_png_file(im, f):
  global palette
  palette_int = map(lambda x: map(lambda xx: int(255 * xx), x), palette)
  w = png.Writer(size=(im.shape[1], im.shape[0]), bitdepth=8, palette=palette_int)
  with open(f, "w") as ff:
    w.write(ff, im)


def visualize_heatmap(heatmap):
  """
  :param heatmap: [h, w], float
  :return: visualized image
  """
  minVal, maxVal = np.min(heatmap), np.max(heatmap)
  if minVal < 0:
    print 'Error'
  temp = (heatmap - minVal) * 255.0 / (maxVal - minVal)
  temp = cv2.convertScaleAbs(temp)
  temp = cv2.applyColorMap(temp, cv2.COLORMAP_JET)
  return temp


def visualize_labelmap(label_map, seed_mask):
  """
  :param label_map: int, shape [h, w]
  :param seed_map: int shape [h, w]
  :return: visualized image
  """

  label_map_vis = np.ones((label_map.shape[0], label_map.shape[1], 3))
  for i in range(label_map.shape[0]):
    for j in range(label_map.shape[1]):
      idx = label_map[i, j]
      if seed_mask[i, j] > 0 and idx >= 0:
        label_map_vis[i, j, :] = palette[idx]
  label_map_vis = label_map_vis[:, :, (2, 1, 0)]
  return (label_map_vis * 255).astype(np.uint8)


def visualize_boxes(img, boxes, is_RGB=True, max_boxes=10):
  """
  images will be converted to uint8 directly
  :param img: np array [h, w, c]
  :param boxes: [num_boxes, 4]
  :param max_boxes: int, max #boxes to be drawn
  :return: visualized image
  """
  if is_RGB:
    img = img[:, :, ::-1]
  img = img.astype(np.uint8)
  boxes = boxes.astype(np.int32)
  for ind in range(min(max_boxes, boxes.shape[0])):
    cv2.rectangle(img, (boxes[ind, 1], boxes[ind, 0]),
                  (boxes[ind, 3], boxes[ind, 2],), (0, 255, 0), 2)
  return img