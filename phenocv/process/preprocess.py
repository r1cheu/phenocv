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
from pathlib import Path
from typing import Union

import numpy as np
import torch

from phenocv.utils import Results, read_image

from .base import Processor
from .utils import binarize_cive, min_sum, moving_average


class PanicleUavPreprocessor(Processor):
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
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.window_size = window_size
        self.names = {0: "plot"}

    def __call__(self, img_path: Union[str, Path]):
        """using the CIVE to binarize the image, then find the minimum
        sum of the binarized image to locate the plot.

        Args:
            img_path (str or Path): The path to the image.
        """
        img = read_image(img_path)
        boxes = self.process(img)

        self._result = Results(
            orig_img=img,
            names=self.names,
            path=img_path,
            boxes=boxes,
        )

        return self.result

    def process(self, img: np.ndarray):
        bin_image = binarize_cive(img)

        x = np.apply_along_axis(np.sum, 0, bin_image)
        x = moving_average(x, self.window_size)
        x1, x2 = min_sum(x, self.width, self.window_size)

        y = np.apply_along_axis(np.sum, 1, bin_image)
        y = moving_average(y, self.window_size)
        y1, y2 = min_sum(y, self.height, self.window_size)

        boxes = torch.tensor([[x1, y1, x2, y2, 0, 0]])

        return boxes
