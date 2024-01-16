from abc import ABCMeta, abstractmethod
from collections import namedtuple
from logging import warning
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

from phenocv.utils import check_path
from .utils import binarize_cive, min_sum, moving_average, resize

BoundingBox = namedtuple('BoundingBox', ['x1', 'x2', 'y1', 'y2'])


class ImageExtractor(metaclass=ABCMeta):

    def __init__(self, img_path: Union[str, Path]):

        check_path(img_path)

        self.image_path = Path(img_path)
        self.ori_image = self.read_image()
        self.ori_image_shape = self.ori_image.shape

        self._processed = False
        self._processed_image = self.ori_image
        self._processed_image_shape = self.ori_image_shape

    @abstractmethod
    def process(self):
        raise NotImplementedError('process has not been Implemented yet')

    def read_image(self):
        """Reads an image from a given path.

        Parameters:
        img_path (str): The path to the image file.

        Returns:
        numpy.ndarray: The image.
        """
        return cv2.imread(str(self.image_path))

    def save_image(self, img_path: Optional[Union[str, Path]] = None):
        """Saves an image to a given path.

        Parameters:
        img (numpy.ndarray): The image.
        img_path (str): The path to the image file.
        """
        if not self._processed:
            self.process()
        if img_path is None:
            img_path = self.image_path.stem + '_processed' + \
                       self.image_path.suffix

        cv2.imwrite(str(img_path), self._processed_image)

    @property
    def image_pil(self):
        image = cv2.cvtColor(self.ori_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)

    @property
    def processed_image_pil(self):

        if not self._processed:
            self.process()

        processed_image = cv2.cvtColor(self._processed_image,
                                       cv2.COLOR_BGR2RGB)

        return Image.fromarray(processed_image)

    @property
    def image_array(self) -> np.ndarray:
        return cv2.cvtColor(self.ori_image, cv2.COLOR_BGR2RGB)

    @property
    def processed_image_array(self) -> np.ndarray:

        if not self._processed:
            self.process()

        return cv2.cvtColor(self._processed_image, cv2.COLOR_BGR2RGB)


class H20ImageExtractor(ImageExtractor):

    def __init__(
        self,
        img_path: str | Path,
        width: int = 3800,
        height: int = 2000,
        window_size: int = 100,
    ):
        self.height = height
        self.width = width
        self.window_size = window_size
        self.xyxy = None
        super().__init__(img_path)

    def process(self):
        bin_image = binarize_cive(self.ori_image)

        x = np.apply_along_axis(np.sum, 0, bin_image)
        x = moving_average(x, self.window_size)
        x1, x2 = min_sum(x, self.width, self.window_size)

        y = np.apply_along_axis(np.sum, 1, bin_image)
        y = moving_average(y, self.window_size)
        y1, y2 = min_sum(y, self.height, self.window_size)

        self.xyxy = BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2)

        self._processed_image = self.ori_image[y1:y2, x1:x2]
        self._processed_image_shape = self._processed_image.shape
        self._processed = True


class LMJImageExtractor(ImageExtractor):

    def __init__(
        self,
        img_path: Union[str, Path],
        resize_long_side: Optional[int] = None,
    ):
        self.xyxy = None
        self.resize_len = resize_long_side
        super().__init__(img_path)

    def process(self, resize_ratio=1.0):

        bin_image = binarize_cive(self.ori_image)
        contours, _ = cv2.findContours(bin_image, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # find the contour with the largest area
        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        # find the bounding box of the contour
        x, y, w, h = cv2.boundingRect(max_contour)

        if w == 0 or h == 0:
            warning(f'Invalid contour in fig {self.image_path}' +
                    f': {x}, {y}, {w}, {h}, use the original image instead')
            self._processed_image = self.ori_image
            self._processed_image_shape = self.ori_image.shape
            self.xyxy = BoundingBox(x1=0, x2=0, y1=0, y2=0)

        else:
            # crop the image
            h_extend = int(h * 0.1)
            w_extend = int(w * 0.1)

            y1 = y - h_extend
            y2 = y + h + h_extend
            x1 = x - w_extend
            x2 = x + w + w_extend

            # make sure the crop position is within the image
            if y1 < 0:
                y1 = 0
            if y2 > self.ori_image.shape[0]:
                y2 = self.ori_image.shape[0]
            if x1 < 0:
                x1 = 0
            if x2 > self.ori_image.shape[1]:
                x2 = self.ori_image.shape[1]
            self.xyxy = BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2)
            crop_image = self.ori_image[y1:y2, x1:x2]

            # resize the image with the highest quality
            if self.resize_len is not None:
                crop_image = resize(crop_image, self.resize_len)

            self._processed_image = crop_image
            self._processed_image_shape = self._processed_image.shape

        self._processed = True


class ResizeExtractor(ImageExtractor):

    def __init__(
        self,
        img_path: Union[str, Path],
        resize_long_side: Optional[int] = 1280,
    ):
        self.resize_len = resize_long_side
        super().__init__(img_path)

    def process(self):
        self._processed_image = resize(self.ori_image, self.resize_len)
        self._processed_image_shape = self._processed_image.shape
        self._processed = True
