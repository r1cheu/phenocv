from typing import List, Sequence, Tuple

import torch
from torchvision.ops import nms
from ultralytics.engine.results import Boxes, Results


def shift_predictions(results: List[Results], offsets: Sequence[Tuple[int,
                                                                      int]],
                      src_image_shape) -> Boxes:
    """Shifts the predicted bounding boxes based on the given offsets.

    Args:
        results (List[Results]): List of Results objects containing predicted
          bounding boxes.
        offsets (Sequence[Tuple[int, int]]): Sequence of (x, y) offsets to
          shift the bounding boxes.
        src_image_shape: Shape of the source image.

    Returns:
        Boxes: Shifted bounding boxes.

    Raises:
        ImportError: If the 'sahi' module is not installed.
    """
    try:
        from sahi.slicing import shift_bboxes
    except ImportError:
        raise ImportError('Please run "pip install -U sahi" '
                          'to install sahi first for large image inference.')

    assert len(results) == len(
        offsets), 'The `results` should has the ' 'same length with `offsets`.'

    shifted_predictions = []
    for result, offset in zip(results, offsets):
        pred_inst = result.boxes
        shifted_bboxes = shift_bboxes(pred_inst.xyxy, offset)
        conf_cls = pred_inst.data[:, -2:].clone()
        shifted_bboxes = torch.cat([shifted_bboxes, conf_cls], dim=1)
        # handle the case that no bbox is predicted
        if shifted_bboxes.numel() == 0:
            shifted_bboxes = torch.empty((0, 6), device=shifted_bboxes.device)
        shifted_predictions.append(shifted_bboxes)

    shifted_predictions = torch.cat(shifted_predictions, dim=0)
    shifted_predictions = Boxes(shifted_predictions, src_image_shape)
    return shifted_predictions


def merge_results_by_nms(results: List[Results], offsets: Sequence[Tuple[int,
                                                                         int]],
                         src_image_shape, iou_thres) -> Boxes:
    """Merge the results of object detection by applying non-maximum
    suppression (NMS).

    Args:
        results (List[Results]): List of Results objects containing the
          predicted bounding boxes and confidence scores.
        offsets (Sequence[Tuple[int, int]]): Sequence of offset tuples for
          each result.
        src_image_shape: Shape of the source image.
        iou_thres: IoU (Intersection over Union) threshold for NMS.

    Returns:
        Boxes: Merged bounding boxes after applying NMS.
    """
    shifted_instances = shift_predictions(results, offsets, src_image_shape)

    keep = nms(shifted_instances.xyxy, shifted_instances.conf, iou_thres)
    bboxes = shifted_instances.data[keep].clone()
    return Boxes(bboxes, shifted_instances.orig_shape)
