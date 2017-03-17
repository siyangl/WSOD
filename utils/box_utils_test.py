import numpy as np
import utils.box_utils as bu


def test_generate_anchor_centers():
    center_h = np.array([1., 3., 5.])
    center_w = np.array([2., 4.])
    centers = bu.generate_anchor_centers(center_h, center_w)
    assert np.allclose(centers,  np.array([[1., 2.],
                                           [1., 4.],
                                           [3., 2.],
                                           [3., 4.],
                                           [5., 2.],
                                           [5., 4.]]))

def test_generate_anchors():
    centers = np.array([[1., 2.],
                        [1., 4.],
                        [3., 2.],
                        [3., 4.],
                        [5., 2.],
                        [5., 4.]])
    sizes = np.array([[1., 2.], [1., 1.]])
    anchors = bu.generate_anchor_boxes(centers, sizes)
    assert np.allclose(anchors, [[1., 2., 1., 2.],
                                 [1., 2., 1., 1.],
                                 [1., 4., 1., 2.],
                                 [1., 4., 1., 1.],
                                 [3., 2., 1., 2.],
                                 [3., 2., 1., 1.],
                                 [3., 4., 1., 2.],
                                 [3., 4., 1., 1.],
                                 [5., 2., 1., 2.],
                                 [5., 2., 1., 1.],
                                 [5., 4., 1., 2.],
                                 [5., 4., 1., 1.]])


def test_generate_batch_anchors():
    anchors = np.array([[0.1, 0.1, 0.5, 0.5],
                        [0.2, 0.2, 0.4, 0.4]])
    batch_size = 4
    batch_anchors, batch_anchor_indices = bu.generate_batched_anchors(anchors, batch_size)
    assert np.allclose(batch_anchors, np.array([[ 0.1,  0.1,  0.5,  0.5],
                                                [ 0.2,  0.2,  0.4,  0.4],
                                                [ 0.1,  0.1,  0.5,  0.5],
                                                [ 0.2,  0.2,  0.4,  0.4],
                                                [ 0.1,  0.1,  0.5,  0.5],
                                                [ 0.2,  0.2,  0.4,  0.4],
                                                [ 0.1,  0.1,  0.5,  0.5],
                                                [ 0.2,  0.2,  0.4,  0.4]]))
    assert np.array_equal(batch_anchor_indices, np.array([0, 0, 1, 1, 2, 2, 3, 3]))


def test_bbox_transform():
    bboxes = np.array([[0.3, 0.4, 0.6, 0.6],
                       [0.4, 0.8, 0.4, 0.8]])
    bboxes_transform = bu.bbox_transform(bboxes)
    assert np.allclose(bboxes_transform, np.array([[0., 0.1, 0.6, 0.7],
                                                   [0.2, 0.4, 0.6, 1.2]]))


def test_generate_anchors_from_density_and_sizes():
    density = 2
    sizes = np.array([[0.6, 0.6], [0.2, 0.2]])
    anchors = bu.generate_anchors_from_density_and_sizes(density, sizes)
    assert np.allclose(anchors, np.array([[0., 0., 0.55, 0.55],
                                          [0.15, 0.15, 0.35, 0.35],
                                          [0., 0.45, 0.55, 1.],
                                          [0.15, 0.65, 0.35, 0.85],
                                          [0.45, 0., 1., 0.55],
                                          [0.65, 0.15, 0.85, 0.35],
                                          [0.45, 0.45, 1., 1.],
                                          [0.65, 0.65, 0.85, 0.85]]))

def test_point_in_box():
    point = np.array([0.3, 0.4])
    bboxes = np.array([[0.1, 0.1, 0.2, 0.2],
                       [0.2, 0.3, 0.4, 0.5],
                       [0.31, 0.3, 0.35, 0.39]])
    is_in = bu.point_in_box(point, bboxes)
    assert np.array_equal(is_in, [False, True, False])
    is_in = bu.point_in_box(point, bboxes, tolerance=0.15)
    assert np.array_equal(is_in, [False, True, True])


def main():
    test_generate_anchor_centers()
    test_generate_anchors()
    test_generate_batch_anchors()
    test_bbox_transform()
    test_generate_anchors_from_density_and_sizes()
    test_point_in_box()

if __name__ == "__main__":
    main()