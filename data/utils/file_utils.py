import xml.etree.ElementTree as ET
import numpy as np

from data.cfg import pascal_dir


def get_annotations_from_xml(xml_file, label_list):
  """
  :param xml_file: str, path to annotation file in xml format
  :param label_list: list of str, a list of possible labels
  :return image_size: array, image_size [h, w, c]
  :return labels: np array [num_possible_classes],
          0 for absent, 1 for present, -1 for difficult but present
  :return objects: np array [num_objs, 5],
          [cls, ymin, xmin, ymax, xmax], cls < 0 meaning diffcult objects
  """
  tree = ET.parse(xml_file)
  root = tree.getroot()
  size = root.find('size')
  height = int(size.find('height').text)
  width = int(size.find('width').text)
  channel = int(size.find('depth').text)
  labels = np.zeros(len(label_list), dtype=np.int32)
  objects = []
  for obj in root.iter('object'):
    name = obj.find('name').text
    is_difficult = (int(obj.find('difficult').text) > 0)
    label_index = label_list.index(name)
    if label_index < len(label_list):
      if not is_difficult:
        labels[label_index] = 1
      else:
        if labels[label_index] == 0:
          labels[label_index] = -1
    else:
      print('Label unfound in valid label list: %s'%name)

    object = np.zeros((5), dtype=np.int32)
    if is_difficult:
      object[0] = -label_index
    else:
      object[0] = label_index
    bbox = obj.find('bndbox')
    xmin = int(bbox.find('xmin').text)
    ymin = int(bbox.find('ymin').text)
    xmax = int(bbox.find('xmax').text)
    ymax = int(bbox.find('ymax').text)
    object[1] = ymin
    object[2] = xmin
    object[3] = ymax
    object[4] = xmax
    objects.append(object)
  objects = np.array(objects)
  return np.array([height, width, channel]), labels, objects


def get_list_from_file(file):
  with open(file) as f:
    lines = f.readlines()
  lines = [l.strip() for l in lines]
  count = 0
  while count < len(lines):
    if len(lines[count]) < 1:
      lines.remove(lines[count])
    else:
      count += 1
  return lines


def create_xml(image_name, image_size, object_bboxes, category_list):
  """ inverse of get_annotations_from_xml
  :param image_name: str
  :param object_bboxes: [#bboxes, 5], [category, hmin, wmin, hmax, wmax], absolute coordinates
  :param category_list: list of str
  :return: xml tree
  """

  tree = ET.parse('%s/Annotations_bak_true_anno/%s.xml' % (pascal_dir, image_name))
  root = tree.getroot()
  for obj in root.findall('object'):
    root.remove(obj)

  for i in range(object_bboxes.shape[0]):
    category_idx = object_bboxes[i, 0]
    box = object_bboxes[i, 1:5]
    object = ET.SubElement(root, 'object')
    name = ET.SubElement(object, 'name')
    category_text = category_list[abs(category_idx)]
    name.text = category_text
    pose = ET.SubElement(object, 'pose')
    pose.text = 'Unspecified'
    diff = ET.SubElement(object, 'difficult')
    if category_idx > 0:
      diff.text = '0'
    else:
      diff.text = '1'

    trunc = ET.SubElement(object, 'truncated')
    trunc.text = '0'
    bndbox = ET.SubElement(object, 'bndbox')
    hmin = box[0]
    wmin = box[1]
    hmax = box[2]
    wmax = box[3]

    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = str(int(wmin))
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = str(int(hmin))
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = str(int(wmax))
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = str(int(hmax))
  root_str = ET.tostring(root)
  return root_str