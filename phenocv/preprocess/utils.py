import cv2
import numpy as np


def resize(image: np.ndarray, target_size=1000):
    h, w = image.shape[:2]
    if h < w:
        new_h, new_w = target_size * h / w, target_size
    else:
        new_h, new_w = target_size, target_size * w / h

    new_h, new_w = int(new_h), int(new_w)
    if h > new_h:
        inter = cv2.INTER_AREA
    else:
        inter = cv2.INTER_CUBIC

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=inter)
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
