from abc import ABCMeta, abstractmethod
from collections import namedtuple
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
from ultralytics.engine.results import Results

from phenocv.utils import draw_bounding_boxes, imshow, save_img

StubbleGrid = namedtuple('StubbleGrid',
                         ['path', 'image', 'label', 'bboxes', 'label2box'])


def generate_label(row: int, col: int):
    """Generate labels for each element in a 2D grid. row=2, col=2, returns
    np.array([1-1, 1-2, 2-1, 2-2]), for example.

    Args:
        row (int): The number of rows in the grid.
        col (int): The number of columns in the grid.

    Returns:
        numpy.ndarray: An array of labels for each element in the grid.
    """
    label = []
    for i in range(row):
        for j in range(col):
            label.append(f'{i+1}-{j+1}')
    return np.array(label)


def compute_dist(current, next):
    """Compute the Manhattan distance between two arrays after removing
    elements where either array is equal to 0.

    Parameters:
    i (numpy.ndarray): The first array.
    next (numpy.ndarray): The second array.

    Returns:
    numpy.ndarray: The absolute difference between the non-zero elements of
        the two arrays.
    """
    index = np.any(np.stack((current == 0, next == 0)), axis=0)

    current = current[~index]
    next = next[~index]
    return np.abs(next - current)


class YOLOInfer(metaclass=ABCMeta):
    """YOLOInfer is an abstract base class for performing inference using YOLO
    models.

    Args:
        model: The YOLO model to use for inference.
        classes (int): The number of classes in the model.
        devices (int or Tuple[int]): The device(s) to use for inference.

    Attributes:
        model: The YOLO model used for inference.
        classes (int): The number of classes in the model.
        _results (list): A list to store the processed inference results.

    Methods:
        __call__: Perform inference on the given source.
        results: Get the inference results.
        _clear: Clear the inference results.
        process: Abstract method to process the inference result.
        _get_bbox: Get the bounding boxes from the inference result.
    """

    def __init__(self,
                 model,
                 classes: int = 0,
                 devices: Union[int, Tuple[int]] = 0) -> None:
        self.model = YOLO(model).to(devices)
        self.classes = classes
        self._results = []

    def __call__(self, source, conf=0.4, iou=0.1, row=None, col=None):
        """Perform inference on the given source.

        Args:
            source: The source for inference.
            conf (float): The confidence threshold for object detection.
            iou (float): The IoU threshold for non-maximum suppression.
            row (int): The row index of the image when processing a grid of
                images.
            col (int): The column index of the image when processing a grid of
                images.
        """
        self._clear()
        results = self.model.predict(source, conf=0.4, iou=0.1, verbose=False)

        if row or col is not None:
            assert len(results) == 1, ('Only one image is allowed when row or'
                                       'col is specified')

        for _result in results:
            _result = _result.cpu().numpy()
            result = self.process(_result, row, col)
            self._results.append(result)

    @property
    def results(self):
        """Get the inference results.

        Returns:
            list: The list of inference results.
        """
        return self._results

    def _clear(self):
        """Clear the inference results."""
        self._results = []

    @abstractmethod
    def process(self, result):
        """Abstract method to process the inference result.

        Args:
            result: The inference result to process.
        """
        pass

    def _get_bbox(self, result: Results):
        """Get the bounding boxes from the inference result.

        Args:
            result (Results): The inference result.

        Returns:
            tuple: A tuple containing the normalized and unnormalized bounding
                boxes.
        """
        boxes = result.boxes
        index = boxes.cls == self.classes

        return (boxes.xywhn[index], boxes.xyxy[index])


def cut_bbox(img, bbox, scale=0):
    """Cuts out a bounding box region from an image.

    Parameters:
    img (numpy.ndarray): The input image.
    bbox (tuple): The bounding box coordinates in the format (x0, y0, x1, y1).
    scale (float): The scale factor to expand the bounding box (default: 0).

    Returns:
    numpy.ndarray: The cropped image region defined by the bounding box.
    """
    x0, y0, x1, y1 = bbox

    x0 = x0 - (x1 - x0) * scale
    x1 = x1 + (x1 - x0) * scale
    y0 = y0 - (y1 - y0) * scale
    y1 = y1 + (y1 - y0) * scale

    if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(img.shape[1], x1)
        y1 = min(img.shape[0], y1)

    return img[int(y0):int(y1), int(x0):int(x1)]


class YOLOStubbleUAV(YOLOInfer):
    """YOLOStubbleUAV is a class that performs inference using YOLO model for
    detecting stubble in UAV images.

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
        plot(show=False, save_dir=None): Plots the bounding boxes on the images
            and optionally saves them.
        save_cut(save_dir, expand_scale=0.05): Saves the cropped images of the
        detected stubble.
    """

    def __init__(self,
                 model,
                 classes: int = 0,
                 devices: int | Tuple[int] = 0,
                 dist: int = 900) -> None:
        super().__init__(model, classes, devices)
        self.dist = dist

    def process(self, result, row=None, col=None) -> StubbleGrid:
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

        label2box = dict()
        non_zero = (bboxes == np.zeros(4)).any(axis=1)
        for _label, _box in zip(label[~non_zero], bboxes[~non_zero]):
            label2box[_label] = _box

        return StubbleGrid(
            image=img,
            path=result.path,
            label=label,
            bboxes=bboxes,
            label2box=label2box)

    def plot(self, show=False, save_dir=None):
        """Plots the bounding boxes on the images and optionally saves them.

        Args:
            show (bool): Whether to show the images with bounding boxes.
            save_dir (str): The directory to save the images. If None, the
                images are not saved.

        Returns:
            List[np.ndarray]: The images with bounding boxes.
        """

        imgs = []
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir()

        for result in self._results:
            img = result.image
            bboxes = result.bboxes
            label = result.label

            img = draw_bounding_boxes(
                img,
                bboxes,
                labels=label,
                width=10,
                colors='#83D6C5',
                font_size=70)

            if save_dir is not None:
                img_path = Path(result.path)
                img_name = img_path.name
                save_img(str(save_dir / img_name), img)

            if show:
                imshow(img[:, :, ::-1])

        return imgs

    def save_cut(self, save_dir, expand_scale=0.05):
        """Saves the cropped images of the detected stubble.

        Args:
            save_dir (str): The directory to save the cropped images.
            expand_scale (float): The scale factor to expand the bounding
                boxes.
        """

        save_dir = Path(save_dir)
        save_dir.mkdir()

        for result in self._results:

            img = result.image
            bboxes = result.label2box

            for label, bbox in bboxes.items():
                bbox = cut_bbox(img, bbox, expand_scale)
                img_path = Path(result.path)
                img_name = img_path.stem + f'_{label}' + img_path.suffix
                save_img(str(save_dir / img_name), bbox)

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
