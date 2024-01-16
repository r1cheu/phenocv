import warnings
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from pathlib import Path
from typing import Tuple, Union

import cv2
import sahi
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.slicing import slice_image
from torch import Tensor
from ultralytics import YOLO
from ultralytics.engine.results import Results

from phenocv.utils import imshow
from .utils import (cut_bbox, draw_bounding_boxes, merge_results_by_nms,
                    random_rgb_color, save_img)

ODresult = namedtuple('ODresult',
                      ['path', 'image', 'label', 'bboxes', 'label2box'])


class YoloInfer(metaclass=ABCMeta):
    """YOLOInfer is an abstract base class for performing inference using YOLO
    models.

    Args:
        model: The YOLO model to use for inference.
        classes (int): The number of classes in the model.
        device (int or Tuple[int]): The device(s) to use for inference.

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
                 device: Union[int, Tuple[int]] = 0) -> None:
        self.model = YOLO(model).to(device)
        self.classes = classes
        self._results = []
        self.device = device

    def __call__(self, source, conf=0.25, iou=0.7):
        """Perform inference on the given source.

        Args:
            source: The source for inference.
            conf (float): The confidence threshold for object detection.
            iou (float): The IoU threshold for non-maximum suppression.
        """
        self._clear()
        results = self.model.predict(source, conf=conf, iou=iou, verbose=False)

        for _result in results:
            _result = _result.cpu().numpy()
            result = self.process(_result)
            self._results.append(result)

    def plot(self, show=False, save_dir=None, box_color=None):
        """Plots the bounding boxes on the images and optionally saves them.

        Args:
            box_color (str): color in hex or rgb format.
            show (bool): Whether to show the images with bounding boxes.
            save_dir (str): The directory to save the images. If None, the
                images are not saved.

        Returns:
            List[np.ndarray]: The images with bounding boxes.
        """
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir()

        if box_color is None:
            box_color = random_rgb_color()

        for result in self._results:
            img = result.image
            bboxes = result.bboxes
            label = result.label

            img = draw_bounding_boxes(
                img, bboxes, labels=label, colors=box_color)

            if save_dir is not None:
                img_path = Path(result.path)
                img_name = img_path.name
                save_img(str(save_dir / img_name), img)

            if show:
                imshow(img[:, :, ::-1])

    def save_cut(self, save_dir, show=False, labels=None, expand_scale=0.05):
        """Saves the cropped images of the detected stubble.

        Args:
            show: whether to show the cutting results using matplotlib.
            save_dir (str): The directory to save the cropped images.
            expand_scale (float): The scale factor to expand the bounding
                boxes.
            labels (List[str]): The labels to choose specific bbox to cut. If
                None, all the bboxes will be cut.
        """

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        if self._results[0].label2box is None and labels is not None:
            warnings.warn(
                'The inference results do not contain label2box. '
                'The labels will be ignored. And Cut all the bboxes.')
            labels = None

        if labels is None:
            for result in self._results:
                img = result.image
                bboxes = result.bboxes

                for idx, bbox in enumerate(bboxes):
                    bbox = cut_bbox(img, bbox, expand_scale)
                    if show:
                        imshow(bbox[:, :, ::-1])
                    img_path = Path(result.path)
                    img_name = (
                        img_path.stem + f'_{int(idx):02d}' + img_path.suffix)
                    save_img(str(save_dir / img_name), bbox)
        else:
            for result in self._results:
                img = result.image
                label2box = result.label2box

                for label in labels:
                    bbox = label2box[label]
                    bbox = cut_bbox(img, bbox, expand_scale)
                    if show:
                        imshow(bbox[:, :, ::-1])
                    img_path = Path(result.path)
                    img_name = (img_path.stem + f'_{label}' + img_path.suffix)
                    save_img(str(save_dir / img_name), bbox)

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

    def _get_bbox(self, result: Results) -> tuple[Tensor, Tensor]:
        """Get the bounding boxes from the inference result.

        Args:
            result (Results): The inference result.

        Returns:
            tuple: contain xywhn and xyxy.
        """
        boxes = result.boxes
        index = boxes.cls == self.classes

        return boxes.xywhn[index], boxes.xyxy[index]

    def _get_raw_bbox(self, result: Results) -> Tensor:
        """Get the bounding boxes from the inference result.

        Args:
            result (Results): The inference result.

        Returns:
            tuple: contain xywhn and xyxy.
        """
        boxes = result.boxes
        index = boxes.cls == self.classes

        return boxes.data[index]


class _YoloSahiInfer(YoloInfer, metaclass=ABCMeta):
    """This class represents the YOLO-based inference implementation for Sahi
    dataset.

    It inherits from the YoloInfer class and is meant to be used as a base
    class for specific Sahi inference implementations.
    NOTE: the nms in this class is implemented using the merge_results_by_nms,
        which have bugs in it and need to be fixed. Using YoLoSahiInfer
        instead.
    """

    def _sahi_infer(self,
                    img_path,
                    conf=0.25,
                    iou=0.7,
                    merge_iou=0.5,
                    patch_size=(1024, 1024),
                    overlap_ratio=(0.25, 0.25)):
        """Perform Sahi-specific inference on the given image.

        Args:
            img_path (str): The path to the input image.
            conf (float): The confidence threshold for object detection.
            iou (float): The IoU threshold for non-maximum suppression.
            merge_iou (float): The IoU threshold for merging overlapping
                results.
            patch_size (tuple): The size of each patch for image slicing.
            overlap_ratio (tuple): The ratio of overlap between patches.
        Returns:
            list: A list of Results objects containing the inference results.
        """

        img = cv2.imread(img_path)
        orig_shape = img.shape[:2]

        sliced_image_obj = slice_image(
            img,
            slice_height=patch_size[0],
            slice_width=patch_size[1],
            auto_slice_resolution=False,
            overlap_height_ratio=overlap_ratio[0],
            overlap_width_ratio=overlap_ratio[1])

        slice_results = []
        start = 0
        while True:
            end = min(start + 16, len(sliced_image_obj))
            images = []
            for sliced_image in sliced_image_obj.images[start:end]:
                images.append(sliced_image)
            slice_results.extend(
                self.model.predict(images, conf=conf, iou=0, verbose=False))

            if end >= len(sliced_image_obj):
                break
            start += 16

        nms_results = merge_results_by_nms(
            slice_results,
            sliced_image_obj.starting_pixels,
            src_image_shape=orig_shape,
            iou_thres=merge_iou)

        return [
            Results(
                orig_img=img,
                path=img_path,
                boxes=nms_results,
                names=self.model.names)
        ]

    def __call__(self, source, conf=0.25, iou=0.7):
        """Perform inference on the given source.

        Args:
            source: The source for inference.
            conf (float): The confidence threshold for object detection.
            iou (float): The IoU threshold for non-maximum suppression.
        """
        self._clear()
        results = self._sahi_infer(source, conf, iou)

        for _result in results:
            _result = _result.cpu().numpy()
            result = self.process(_result)
            self._results.append(result)


class YoloSahiInfer(YoloInfer, metaclass=ABCMeta):
    """YoloSahiInfer is a class that performs inference using the YOLOv8 model
    with Sahi post-processing.

    Args:
        model (str): The path to the pre-trained YOLOv8 model.
        classes (int): The number of classes in the model.
        conf (float): The confidence threshold for object detection.
        device (Union[int, Tuple[int]]): The device to use for inference.

    Attributes:
        model (AutoDetectionModel): The YOLOv8 model.
        classes (int): The number of classes in the model.
        _results (list): A list to store the inference results.
        device (Union[int, Tuple[int]]): The device used for inference.
    """

    def __init__(self,
                 model,
                 classes: int = 0,
                 conf: float = 0.3,
                 device: Union[int, Tuple[int]] = 0):

        self.model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model,
            confidence_threshold=conf,
            device=f'cuda:{device}')
        self.classes = classes
        self._results = []
        self.device = device

    def _sahi_infer(self,
                    source,
                    slice_height=1024,
                    slice_width=1024,
                    overlap_height_ratio=0.25,
                    overlap_width_ratio=0.25,
                    iou=0.5):
        """Perform Sahi inference on the given source image.

        Args:
            source (str): The path to the source image.
            slice_height (int): The height of each slice for image tiling.
            slice_width (int): The width of each slice for image tiling.
            overlap_height_ratio (float): The overlap ratio between adjacent
                slices in the vertical direction.
            overlap_width_ratio (float): The overlap ratio between adjacent
                slices in the horizontal direction.
            iou (float): The IoU threshold for post-processing.

        Returns:
            list: A list of Results objects containing the inference results.
        """
        img = cv2.imread(source)
        results = get_sliced_prediction(
            source,
            self.model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,  # noqa
            overlap_width_ratio=overlap_width_ratio,  # noqa
            postprocess_match_threshold=iou)

        boxes = [
            self._sahi_to_yolo(result)
            for result in results.object_prediction_list
        ]
        boxes = torch.stack(boxes, dim=0)

        return [
            Results(
                orig_img=img,
                path=source,
                names=self.model.model.names,
                boxes=boxes)
        ]

    def __call__(self,
                 source,
                 slice_height=1024,
                 slice_width=1024,
                 overlap_height_ratio=0.25,
                 overlap_width_ratio=0.25,
                 iou=0.5):
        """Perform inference on the given source image.

        Args:
            source (str): The path to the source image.
            slice_height (int): The height of each slice for image tiling.
            slice_width (int): The width of each slice for image tiling.
            overlap_height_ratio (float): The overlap ratio between adjacent
                slices in the vertical direction.
            overlap_width_ratio (float): The overlap ratio between adjacent
                slices in the horizontal direction.
            iou (float): The IoU threshold for post-processing.
        """
        self._clear()
        results = self._sahi_infer(
            source,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            iou=iou)
        for _result in results:
            _result = _result.cpu().numpy()
            result = self.process(_result)
            self._results.append(result)

    @staticmethod
    def _sahi_to_yolo(result: sahi.ObjectPrediction):
        """Convert Sahi ObjectPrediction to YOLO format.

        Args:
            result (sahi.ObjectPrediction): The Sahi ObjectPrediction object.

        Returns:
            torch.Tensor: The bounding box in YOLO format.
        """
        box = result.bbox.to_xyxy()

        score = result.score.value
        category_id = result.category.id
        box.extend([score, category_id])
        return torch.tensor(box)
