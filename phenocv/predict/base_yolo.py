"""This module contains the base class for performing inference using YOLO.

"""
import os
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Tuple, Union
from urllib.parse import urlparse

import requests
import sahi
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from torch import Tensor
from tqdm import tqdm
from ultralytics import YOLO

from phenocv.utils import Results, read_image


class YoloPredictor(metaclass=ABCMeta):
    """YOLOInfer is an abstract base class for performing inference using YOLO
    models.

    Args:
        model_weight: The YOLO weights to use for inference.
        classes (int): The number of classes in the model.
        device (int or Tuple[int]): The device(s) to use for inference.

    Attributes:
        model: The YOLO model used for inference.
        classes (int): The number of classes in the model.
        _results (list): A list to store the processed inference results.

    Methods:
        __call__: Perform inference on the given source.
        results: Get the inference results.
        process: Abstract method to process the inference result.
        _get_bbox: Get the bounding boxes from the inference result.
    """

    def __init__(
        self,
        model_weight: str,
        classes: int = 0,
        device: Union[int, Tuple[int]] = 0,
    ) -> None:

        self.classes = classes
        self._results = None

        self._init_model(model_weight, device)

    def _init_model(self, model_weight, device):

        parsed = urlparse(model_weight)
        if parsed.scheme in ('http', 'https'):
            response = requests.get(model_weight)
            model_weight = os.path.basename(parsed.path)
            with open(model_weight, 'wb') as f:
                f.write(response.content)
        elif not os.path.isfile(model_weight):
            raise ValueError("model_weight must be a URL or a file path, got "
                             f"{model_weight}")
        self.model = YOLO(model_weight).to(device)
        self.device = device

    def __call__(self, source, conf, iou):
        """Perform inference on the given source.

        Args:
            source: The source for inference.
            conf (float): The confidence threshold for object detection.
            iou (float): The IoU threshold for non-maximum suppression.
        """
        results = self.model.predict(source, conf=conf, iou=iou, verbose=False)

        for _result in results:
            _result = _result.cpu().numpy()
            result = self.process(_result)
            self._results.append(result)

        return self.results

    @abstractmethod
    def process(self, result):
        """Abstract method to process the inference Ultralytics result. if
        needed, else simply return the result.

        Args:
            result: The inference result to process.
        """
        raise NotImplementedError

    @property
    def results(self):
        """Get the inference results.

        Returns:
            list: The list of inference results.
        """
        return self._results

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


class YoloSahiPredictor:

    def __init__(
        self,
        model_type,
        model_weight,
        device: int | Tuple[int] = 0,
        conf=0.25,
        iou=0.7,
        slice_height=1024,
        slice_width=1024,
        overlap_height_ratio=0.25,
        overlap_width_ratio=0.25,
    ) -> None:

        self._results = None
        self.conf = conf
        self.iou = iou
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self._init_model(model_type, model_weight, device)

    @staticmethod
    def load_weight(model_weight: Union[str, os.PathLike]) -> str:
        """
        Load model weights from a local file or a URL.

        If model_weight is a URL, download the file and return the local path.
        If model_weight is a local file path, check its existence and return
        the path.

        Args:
            model_weight: A URL or a local file path pointing to the model
            weights.

        Returns:
            The local path to the model weights.

        Raises:
            ValueError: If model_weight is neither a URL nor a local file path.
        """
        parsed = urlparse(model_weight)

        if parsed.scheme in ('http', 'https'):
            try:
                response = requests.get(model_weight, stream=True)
                response.raise_for_status()
            except requests.RequestException as e:
                raise ValueError(f"Failed to download {model_weight}: {e}")

            # Create full path for the file to be downloaded
            local_path = os.path.join('./', os.path.basename(parsed.path))
            # if local_path exists, skip downloading
            if os.path.isfile(local_path):
                return local_path
            # Download the file with a progress bar
            total_size_in_bytes = int(
                response.headers.get('content-length', 0))
            progress_bar = tqdm(
                total=total_size_in_bytes,
                unit='iB',
                unit_scale=True,
                desc=f'Downloading {os.path.basename(local_path)}',
            )

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    progress_bar.update(len(chunk))
                    f.write(chunk)

            progress_bar.close()

            if (total_size_in_bytes != 0
                    and progress_bar.n != total_size_in_bytes):  # noqa
                raise ValueError(
                    "ERROR, something went wrong with the download")

            print(f"Downloaded file to {os.path.abspath(local_path)}")
        elif os.path.isfile(model_weight):
            local_path = model_weight
        else:
            raise ValueError("model_weight must be a URL or a file path, got "
                             f"{model_weight}")

        return local_path

    def _init_model(self, model_type, model_weight, device):

        model_weight = self.load_weight(model_weight)
        print(f'Loading model from {model_weight}')

        self.model = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=model_weight,
            confidence_threshold=self.conf,
            device=f'cuda:{device}')

        self.device = device

    def _sahi_infer(self, source: str | Path | Results):
        """Perform Sahi inference on the given source image."""
        if isinstance(source, str):
            orig_img = read_image(source)
            path = source
        elif isinstance(source, Results):
            orig_img = source.orig_img
            path = source.path
        else:
            raise TypeError('source must be str or Results.')

        names = self.model.model.names

        results = get_sliced_prediction(
            orig_img,
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

        return Results(orig_img=orig_img, path=path, names=names, boxes=boxes)

    def __call__(self, source: str | Results):
        """Perform inference on the given source image.

        Args:
            source (str | Path | np.ndarray | Results): The path to the source
            image.
        """

        results = self._sahi_infer(source)
        try:
            self._results = self.process(results)
        except NotImplementedError:
            self._results = results

        return self.results

    def process(self, result):
        """Abstract method to process the inference Ultralytics result. if
        needed, else simply return the result.

        Args:
            result: The inference result to process.
        """
        raise NotImplementedError

    @property
    def results(self):
        if self._results is None:
            raise RuntimeError('No results available. Please run inference '
                               'first.')
        return self._results

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
