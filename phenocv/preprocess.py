from abc import ABCMeta, abstractmethod
from collections import namedtuple
from logging import warning
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import PIL

from .utils import check_path

BoundingBox = namedtuple('BoundingBox', ['x1', 'x2', 'y1', 'y2'])


def resize(image, target_size=1000):
    h, w = image.shape[:2]
    if h < w:
        new_h, new_w = target_size * h / w, target_size
    else:
        new_h, new_w = target_size, target_size * w / h

    new_h, new_w = int(new_h), int(new_w)

    resized_image = cv2.resize(
        image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return resized_image


def binarize_cive(img: np.ndarray) -> np.ndarray:
    """Binarize an image using the CIVE (Color Index of Vegetation Extraction)
    algorithm.

    Parameters:
    img (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The binarize image.
    """
    b, g, r = cv2.split(img)
    cive = 0.441 * r - 0.811 * g + 0.385 * b + 18.78745
    gray = cive.astype('uint8')
    _, th = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def moving_average(interval, window_size):
    """Calculate the moving average of a given interval.

    Parameters:
    interval (array-like): The input interval.
    window_size (int): The size of the moving window.

    Returns:
    array-like: The moving average of the input interval.
    """
    window = np.ones(int(window_size)) / float(window_size)
    re = np.convolve(interval, window, 'valid')
    return re


def min_sum(x, length, window_size):
    """Calculate the index range with the minimum sum of a given array.

    Parameters:
    x (array-like): The input array.
    length (int): The length of the index range.
    window_size (int): The window size for moving average.

    Returns:
    tuple: A tuple containing the start and end index of the range with
        the minimum sum.
    """
    i_sum = x[:-length] + x[length:]
    index = np.argmin(i_sum) + window_size
    return index, index + length


class ImageExtractor(metaclass=ABCMeta):

    def __init__(self, img_path: Union[str, Path]):

        check_path(img_path)

        self.image_path = Path(img_path)
        self.ori_image = self.read_image()
        self.ori_image_shape = self.ori_image.shape
        self._processed_image = None

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
        if self._processed_image is None:
            self.process()
        if img_path is None:
            img_path = self.image_path.stem + '_processed' + \
                self.image_path.suffix

        cv2.imwrite(str(img_path), self._processed_image)

    @property
    def image(self):
        image = cv2.cvtColor(self.ori_image, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(image)

    @property
    def processed_image(self):

        if self._processed_image is None:
            self.process()

        processed_image = cv2.cvtColor(self._processed_image,
                                       cv2.COLOR_BGR2RGB)

        return PIL.Image.fromarray(processed_image)


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
        super().__init__(img_path)

    def process(self):
        """Cuts a plot from an image based on the specified width, height, and
        window size. The width, height should be obtain from actual plant
        density.

        Args:
            img_path (str): The path to the image file.
            width (int): The desired width of the plot. Can be obtained from
            actual plot width.
            height (int): The desired height of the plot. Can be obtained from
                actual plot height.
            window_size (int): The size of the moving average window.

        Returns:
            numpy.ndarray: The cropped plot image.
        """

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


class LMJImageExtractor(ImageExtractor):

    def __init__(
        self,
        img_path: Union[str, Path],
        resize_long_side: Optional[int] = None,
    ):

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
            self.processed_image_shape = self.ori_image.shape
            self.xyxy = BoundingBox(x1=0, x2=0, y1=0, y2=0)

        else:
            self.crop_position = dict(x1=x, y1=y, x2=x + w, y2=y + h)
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

            # resize the image with highest quality
            if self.resize_len is not None:
                crop_image = resize(crop_image, self.resize_len)

            self._processed_image = crop_image
            self.processed_image_shape = self._processed_image.shape
