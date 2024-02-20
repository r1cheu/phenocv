from collections import namedtuple
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sahi.predict import get_sliced_prediction
from segment_anything import SamPredictor, sam_model_registry
from sklearn.cluster import DBSCAN
from torch import Tensor

from phenocv.utils import Results, read_image

from .base_yolo import YoloPredictor, YoloSahiPredictor
from .utils import compute_dist, generate_label, make_label_box_map, mask2rbox

# TODO: need to refactor this file
object_result = namedtuple('object_result',
                           ['path', 'image', 'label', 'bboxes', 'label2box'])


class YoloStubbleUav(YoloPredictor):
    """YOLOStubbleUAV is a class that performs inference using YOLO model for
    detecting stubble in UAV test_images.

    Args:
        model: The YOLO model for inference.
        classes (int): The number of classes.
        devices (int | Tuple[int]): The device(s) to use for inference.
        dist (int): The maximum distance between bounding boxes in the grid.

    Attributes:
        dist (int): The maximum distance between bounding boxes in the grid.

    Methods:
        process(result, row=None, col=None): Processes the inference result
            and returns a StubbleGrid object.
        plot(show=False, save_dir=None): Plots the bounding boxes on the
            test_images and optionally saves them.
        save_cut(save_dir, expand_scale=0.05): Saves the cropped test_images of
        the detected stubble.
    """

    def __init__(self,
                 model,
                 classes: int = 0,
                 devices: int | Tuple[int] = 0,
                 dist: int = 900) -> None:
        super().__init__(model, classes, devices)
        self.dist = dist

    def __call__(self, source, conf=0.4, iou=0.1, row=None, col=None):
        """Perform inference on the given source.

        Args:
            source: The source for inference.
            conf (float): The confidence threshold for object detection.
            iou (float): The IoU threshold for non-maximum suppression.
            row (int): The row index of the image when processing a grid of
                test_images.
            col (int): The column index of the image when processing a grid of
                test_images.
        """
        results = self.model.predict(source, conf=conf, iou=iou, verbose=False)

        if row or col is not None:
            assert len(results) == 1, ('Only one image is allowed when row or'
                                       'col is specified')

        for _result in results:
            _result = _result.cpu().numpy()
            result = self.process(_result, row, col)
            self._results.append(result)

    def process(self, result, row=None, col=None):
        """Processes the inference result and returns a StubbleGrid object.

        Args:
            result: The inference result.
            row (int): The row index of the grid to process.
                If None, all rows are processed.
            col (int): The column index of the grid to process.
                If None, all columns are processed.

        Returns:
            StubbleGrid: The processed stubble grid.
        """

        img = result.orig_img
        grid = self._make_stubble_grid(result)
        row0, row1, col0, col1 = self._find_correct_grid(grid)

        if row is not None:
            row0 = row - 1
            row1 = row + 2
        if col is not None:
            col0 = col - 1
            col1 = col + 5

        bboxes = grid[row0:row1, col0:col1]
        label = generate_label(bboxes.shape[0], bboxes.shape[1])
        bboxes = bboxes.reshape(-1, 4)

        non_zero = (bboxes == np.zeros(4)).any(axis=1)
        label2box = make_label_box_map(label[~non_zero], bboxes[~non_zero])

        return object_result(
            image=img,
            path=result.path,
            label=label,
            bboxes=bboxes,
            label2box=label2box)

    def _make_stubble_grid(self, result):
        """Creates a stubble grid from the inference result.

        Args:
            result: The inference result.

        Returns:
            np.ndarray: The stubble grid.
        """

        def cluster(array):
            sort_index = np.argsort(array)
            inverse_index = np.argsort(sort_index)

            sorted_array = array[sort_index].reshape(-1, 1)
            num = DBSCAN(eps=0.05, min_samples=2).fit_predict(sorted_array)

            return num[inverse_index]

        xywhn, xyxy = self._get_bbox(result)
        x = xywhn[:, 0]
        y = xywhn[:, 1]

        rows = cluster(y)
        cols = cluster(x)

        grid = np.zeros((rows.max() + 1, cols.max() + 1, 4))
        grid[rows, cols] = xyxy

        return grid

    def _find_correct_grid(self, grid, nrow=3, ncol=6):
        """Finds the correct grid position for processing.

        Args:
            grid (np.ndarray): The stubble grid.
            nrow (int): The number of rows in the grid.
            ncol (int): The number of columns in the grid.

        Returns:
            Tuple[int, int, int, int]: The row and column indices of the
                correct grid position.
        """

        row, col = 0, 0

        while not self._check_dist(grid[row:row + nrow, :], True):
            row += 1
            if row > grid.shape[0] - nrow:
                break

        while not self._check_dist(grid[:, col:col + nrow], False):
            col += 1
            if col > grid.shape[1] - ncol:
                break

        return row, row + nrow, col, col + ncol

    def _check_dist(self, grid, row=True):
        """Checks if the distance between bounding boxes in the grid is within
        the specified distance.

        Args:
            grid (np.ndarray): The stubble grid.
            row (bool): Whether to check the distance between rows or columns.

        Returns:
            bool: True if the distance is within the specified distance,
                False otherwise.
        """

        axis = 0 if row else 1
        x_or_y = 0 if axis == 1 else 1

        for i in range(grid.shape[axis] - 1):
            next_val = grid.take(indices=[i + 1], axis=axis)[:, x_or_y]
            current_val = grid.take(indices=[i], axis=axis)[:, x_or_y]
            if np.any(compute_dist(current_val, next_val) > self.dist):
                return False

        return True


class YoloStubbleDrone(YoloPredictor):
    """YOLOStubbleDrone is a class that performs inference using YOLO model for
    detecting stubble from image taken by drones.

    Methods:
        process(result: Results) ->
            ODresult: Performs the inference process on the given result.
    """

    def process(self, result: Results):
        img = result.orig_img
        raw_bbox = self._get_raw_bbox(result)
        bboxes = raw_bbox[:, :4]
        names = self.model.names
        label = [f'{names[cls]} {conf:.2f}' for conf, cls in raw_bbox[:, 4:]]

        return object_result(
            image=img,
            path=result.path,
            bboxes=bboxes,
            label=label,
            label2box=None)


class YoloTillerDrone(YoloStubbleDrone):
    """YOLOTillerDrone is a class that performs inference using YOLO model for
    detecting tiller from image taken by drones.

    Methods:
        process(result: Results) ->
            ODresult: Performs the inference process on the given result.
    """
    pass


class YoloSamObb(YoloPredictor):
    """
    YOLOSAMOBB is a class that convert horizontal bounding box to orientational
    bounding box using YOLO model with SAM.
    Args:
        model: The YOLO model.
        classes (int): The number of classes.
        device (int | Tuple[int]): The device to use for inference.
        sam_type (str): The type of SAM model to use.
        sam_weight (str): The path to the SAM model weights.
        max_batch_num_pred (int): The maximum number of predictions per batch.

    Attributes:
        sam: The SAM model for prediction.
        max_batch_num_pred (int): The maximum number of predictions per batch.

    Methods:
        _init_sam: Initializes the SAM model.
        process: Processes the results of object detection.

    """

    def __init__(
            self,
            model,
            classes: int = 0,
            device: int | Tuple[int] = 0,
            sam_type: str = 'vit_h',
            sam_weight: str = '~/ProJect/phenocv/weights/sam_vit_h_4b8939.pth',
            max_batch_num_pred: int = 16) -> None:
        super().__init__(model, classes, device)
        self._init_sam(sam_type, sam_weight)
        self.max_batch_num_pred = max_batch_num_pred

    def _init_sam(self, sam_type: str, sam_weight: str):
        """Initializes the SAM model.

        Args:
            sam_type (str): The type of SAM model to use.
            sam_weight (str): The path to the SAM model weights.
        """

        build_sam = sam_model_registry[sam_type]
        sam_model = SamPredictor(build_sam(checkpoint=sam_weight))
        sam_model.model = sam_model.model.to(self.device)
        self.sam = sam_model

    def process(self, result: Results):
        """Processes the results of object detection.

        Args:
            result (Results): The results of object detection.

        Returns:
            ODresult: The processed object detection results.
        """
        img = result.orig_img
        _, xyxy = self._get_bbox(result)

        self.sam.set_image(img, image_format='BGR')

        masks = []
        num_pred = xyxy.shape[0]
        N = self.max_batch_num_pred

        num_batches = int(np.ceil(num_pred /
                                  N))  # batch the boxes to avoid OOM

        for i in range(num_batches):
            left_index = i * N
            right_index = (i + 1) * N
            if i == num_batches - 1:
                batch_boxes = xyxy[left_index:]
            else:
                batch_boxes = xyxy[left_index:right_index]

            transformed_boxes = self.sam.transform.apply_boxes(
                batch_boxes, img.shape[:2])
            transformed_boxes = torch.from_numpy(transformed_boxes).to(
                self.device)
            batch_masks = self.sam.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False)[0]
            batch_masks = batch_masks.squeeze(1).cpu()
            masks.extend([*batch_masks])
        masks = torch.stack(masks, dim=0)
        r_bboxes = [mask2rbox(mask.numpy()) for mask in masks]
        r_bboxes = np.stack(r_bboxes, axis=0)

        return object_result(
            image=img,
            path=result.path,
            bboxes=r_bboxes,
            label=None,
            label2box=None)


class YoloSahiPanicleUavPredictor(YoloSahiPredictor):

    def _sahi_infer(self, source):
        """Perform Sahi inference on the given source image."""

        # Prepare the image, original image, path, old box and names based
        # on the type of source
        if isinstance(source, Results):
            img, orig_img, path, old_box, names = self._prepare_from_results(
                source)
        elif isinstance(source, (str, Path)):
            img, orig_img, path, old_box, names = self._prepare_from_path(
                source)
        else:
            raise TypeError('source must be a Results, str or Path object.')

        # Get sliced prediction
        results = get_sliced_prediction(
            img,
            self.model,
            self.slice_height,
            self.slice_width,
            self.overlap_height_ratio,
            self.overlap_width_ratio,
            postprocess_match_threshold=self.iou,
            verbose=0)

        # Convert results to YOLO format and stack them
        yolo_result = [
            self._sahi_to_yolo(result)
            for result in results.object_prediction_list]

        if len(yolo_result) == 0:
            boxes = None
        else:
            boxes = torch.stack(yolo_result, dim=0)

        # If old box exists, adjust boxes
        if old_box is not None:
            boxes = self._adjust_boxes_with_old_box(boxes, old_box)

        return Results(orig_img=orig_img, path=path, names=names, boxes=boxes)

    def _prepare_from_results(self, source):
        img = source.crop_img()[:, :, ::-1]  # convert to RGB
        orig_img = source.orig_img
        path = source.path
        old_box, names = self.update_old_box(source)
        return img, orig_img, path, old_box, names

    def _prepare_from_path(self, source):
        img = read_image(source, order='RGB')
        orig_img = img
        path = str(source)
        old_box = None
        names = self.model.model.names
        return img, orig_img, path, old_box, names

    def _adjust_boxes_with_old_box(self, boxes: Tensor, old_box):
        if boxes is None:
            return old_box

        add_box = old_box.clone().zero_()
        add_box[:, :4] = torch.cat([old_box[:, :2], old_box[:, :2]], dim=1)
        add_box = add_box.repeat(boxes.shape[0], 1).to(boxes.device)

        boxes += add_box
        return torch.cat([old_box, boxes], dim=0)

    def update_old_box(self, old_results: Results) -> Tuple[Tensor, Dict]:
        """Updates the old results with the new results.

        Args:
            old_results (Results): The old results.

        Returns:
            Results: The updated results.
        """

        old_names = old_results.names
        old_boxes = old_results.boxes.data
        new_names = self.model.model.names

        # get unique value from new_names and old_names
        names = {*old_names.values(), *new_names.values()}

        # keep the order of the new_names, add the category_id in old_names
        # to it.
        new_names_rev = {v: k for k, v in new_names.items()}
        final_dict = {
            new_names_rev[name]: name
            for name in names if name in new_names_rev}

        # if the old_names is not in the new_names, add it to the final_dict
        for v in old_names.values():
            if v not in final_dict.values():
                final_dict[len(final_dict)] = v

        final_dict_rev = {v: k for k, v in final_dict.items()}

        # update the old_boxes
        for i in range(old_boxes.shape[0]):
            category_id = int(old_boxes[i, -1])
            category = old_names[category_id]
            old_boxes[i, -1] = final_dict_rev[category]

        return torch.Tensor(old_boxes), final_dict
