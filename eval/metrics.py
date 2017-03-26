import numpy as np

def average_presicion(scores, gt):
    assert(scores.shape == gt.shape)
    assert(len(scores.shape) == 1)  # 1-D array
    num_gt = np.sum(gt)
    num_gt = num_gt.astype(int)
    total_label = scores.shape[0]
    index = np.argsort(scores)
    gt_sorted = gt[index]
    precision = np.zeros(num_gt)

    count = 0
    for i in range(0, total_label):
        if gt_sorted[i] > 0.5:
            # thres = scores_sorted[i]
            binary_prediction = np.concatenate((np.zeros(i), np.ones(total_label-i)), axis=0)
            # print(binary_prediction)
            tp = np.logical_and(np.isclose(binary_prediction, gt_sorted), np.isclose(gt_sorted, 1.))
            tp = tp.astype(float)
            tp = np.sum(tp)
            precision[count] = tp/np.sum(binary_prediction)
            count += 1
    assert(count == num_gt)
    ap = np.sum(precision)/num_gt
    return ap


def get_iou_matrix(gt, det):
    iou_matrix=np.zeros((gt.shape[0], det.shape[0]))
    for g in range(gt.shape[0]):
        gt_dup = np.tile(gt[g], (det.shape[0], 1))
        ymin = np.maximum(gt_dup[:, 0], det[:, 0])
        xmin = np.maximum(gt_dup[:, 1], det[:, 1])
        ymax = np.minimum(gt_dup[:, 2], det[:, 2])
        xmax = np.minimum(gt_dup[:, 3], det[:, 3])
        overlap = np.maximum(0, ymax-ymin)*np.maximum(0, xmax-xmin)
        union = (gt_dup[:, 2] - gt_dup[:, 0])*(gt_dup[:, 3] - gt_dup[:, 1]) + \
                (det[:, 2] - det[:, 0])*(det[:, 3] - det[:, 1]) - overlap
        iou = overlap/union
        iou_matrix[g] = iou

    return iou_matrix