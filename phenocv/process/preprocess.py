"""Preprocess module is used to do some preprocessing on the original
    test_images.

class:
    ImagePreprocessor: The base class for all image extractors. Preprocessor
        first initializes a deep learning model if needed, then process the
        image with the model or algorithms.
    PanicleUavPreprocessor: Preprocess the image from the UAV, which need to
        locate the central plot of the image. The image is first binarized
        with CIVE, then the image is cropped to the region of interest.

Typical usage example:

    preprocessor = PanicleUavPreprocessor()
    results = preprocessor('path/to/image')
    or,
    results = preprocessor.results

"""
import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch

from phenocv.utils import Results, read_image

from .base import Processor
from .utils import binarize_cive, find_min


class Preprocessor(Processor):

    def __init__(self,
                 names: Tuple | str = None):
        super().__init__()

        if names is None:
            logging.warning('No names provided, using default names')
            self.names = {0: "NoName"}
        if isinstance(names, str):
            self.names = {0: names}
        if isinstance(names, tuple):
            self.names = {i: name for i, name in enumerate(names)}

    def __call__(self, img_path: Union[str, Path]):
        img = read_image(img_path, order='RGB')
        boxes = self.process(img)

        self._result = Results(
            orig_img=img,
            names=self.names,
            path=img_path,
            boxes=boxes,
        )

        return self.result


class PrePanicleUavHW(Preprocessor):
    """Preprocess the image from the UAV, which need to locate the central
    plot of the image. The image is first binarized with CIVE, then the
    image is cropped to the region of interest.

    Attributes:
        height (int): The expected height of the plot.
        width (int): The expected width of the plot.
        window_size (int): The window size for moving average. The larger the
            window size, the smoother the curve.
        names (dict): A dictionary of the (id: category) pairs.
    """

    def __init__(
        self,
        width: int = 3800,
        height: int = 2000,
        window_size: int = 100,
        names: Tuple[str] = ('plot', ),
    ):
        super().__init__(names=names)
        self.height = height
        self.width = width
        self.window_size = window_size

    def process(self, img: np.ndarray):
        bin_image = binarize_cive(img)

        x1, x2 = find_min(bin_image, 0, self.window_size, self.width)
        y1, y2 = find_min(bin_image, 1, self.window_size, self.height)

        return torch.tensor([[x1, y1, x2, y2, 0, 0]])


class PrePanicleUav(Preprocessor):

    def __init__(self,
                 window_size: int = 100,
                 w_ratio: float = 2,
                 names: Tuple[str] = ('plot', )):
        super().__init__(names=names)
        self.window_size = window_size
        self.w_ratio = w_ratio

    def process(self, img: np.ndarray):

        bin_image = binarize_cive(img)

        up, down = self.cut_2_image(bin_image)

        y1 = find_min(up, 1, self.window_size)
        y2 = find_min(down, 1, self.window_size) + bin_image.shape[0] // 2
        width = int((y2 - y1) * self.w_ratio)
        x1, x2 = find_min(bin_image, 0, self.window_size, width)

        return torch.tensor([[x1, y1, x2, y2, 0, 0]])

    @staticmethod
    def cut_2_image(img):

        half_h = img.shape[0] // 2

        up = img[:half_h, :]
        down = img[half_h:, :]

        return up, down
