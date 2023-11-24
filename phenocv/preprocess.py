import cv2
import numpy as np


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


def cut_plot(img_path, width, height, window_size):
    """Cuts a plot from an image based on the specified width, height, and
    window size.

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

    img = cv2.imread(str(img_path))
    binary = binarize_cive(img)

    x = np.apply_along_axis(np.sum, 0, binary)
    x = moving_average(x, window_size)
    x1, x2 = min_sum(x, width, window_size)

    y = np.apply_along_axis(np.sum, 1, binary)
    y = moving_average(y, window_size)
    y1, y2 = min_sum(y, height, window_size)

    img = img[y1:y2, x1:x2]
    return img, y1, y2, x1, x2
