import os
import random
import warnings
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import _parse_colors


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


def make_label_box_map(label, boxes):

    map_dict = dict()

    for _label, box in zip(label, boxes):
        map_dict[_label] = box

    return map_dict


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


def save_img(path, img):
    cv2.imwrite(path, img)


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: Optional[List[str]] = None,
    font: Optional[str] = None,
    colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str,
                           Tuple[int, int, int]]] = None,
    fill: Optional[bool] = False,
    font_size: Optional[int] = None,
) -> np.ndarray:
    """Draw bounding boxes on an image.

    Args:
        image (np.ndarray): The image to be drawn on. It should be in
            (H, W, C) format.
        boxes (np.ndarray): A tensor of shape (N, 4) containing the boxes in
            (x1, y1, x2, y2) format.
        labels (List[str], optional): A list containing the labels of boxes.
            Defaults to None.
        colors (Union[List[Union[str, Tuple[int, int, int]]], str,
            Tuple[int, int, int]], optional):
            A list containing the colors of boxes. Defaults to None.
        fill (bool, optional): Whether to fill the boxes with colors.
            Defaults to False.
        width (int, optional): The line width of boxes. Defaults to 1.
        font_size (int, optional): The font size of labels. Defaults to None.

    Returns:
        np.ndarray[H, W, C]: The image with drawn bounding boxes.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError(f'numpy ndarray expected, got {type(image)}')
    elif image.dtype != np.uint8:
        raise ValueError(f'uint8 dtype expected, got {image.dtype}')
    elif len(image.shape) != 3:
        raise ValueError('Pass individual images, not batches')
    elif image.shape[-1] not in {1, 3}:
        raise ValueError('Only grayscale and RGB images are supported')
    elif (boxes[:, 0] > boxes[:, 2]).any() or (boxes[:, 1] > boxes[:,
                                                                   3]).any():
        raise ValueError('Boxes need to be in (xmin, ymin, xmax, ymax) format.'
                         'Use torchvision.ops.box_convert to convert them')

    num_boxes = boxes.shape[0]

    if num_boxes == 0:
        warnings.warn("boxes doesn't contain any box. No box was drawn")
        return image

    if labels is None:
        labels: Union[List[str],
                      List[None]] = [None
                                     ] * num_boxes  # type: ignore[no-redef]
    elif len(labels) != num_boxes:
        raise ValueError(
            f'Number of boxes ({num_boxes}) and labels ({len(labels)})'
            'mismatch. Please specify labels for each box.')

    colors = _parse_colors(colors, num_objects=num_boxes)

    if font is None:
        font = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')

    txt_font = ImageFont.truetype(font, size=font_size)

    # Handle Grayscale images
    if image.shape[-1] == 1:
        image = np.tile(image, (3, 1, 1))
    ndarr = image[:, :, ::-1].copy()
    img_to_draw = Image.fromarray(ndarr)
    img_boxes = boxes.astype(np.int64).tolist()
    width = int(max(np.abs(img_boxes[0][0] - img_boxes[0][2]) / 10, 1))

    if fill:
        draw = ImageDraw.Draw(img_to_draw, 'RGBA')
    else:
        draw = ImageDraw.Draw(img_to_draw)

    for bbox, color, label in zip(img_boxes, colors,
                                  labels):  # type: ignore[arg-type]
        if fill:
            fill_color = color + (100, )
            draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
        else:
            draw.rectangle(bbox, width=width, outline=color)

        if label is not None:
            margin = width + 1
            draw.text((bbox[0] + margin, bbox[1] + margin),
                      label,
                      fill=color,
                      font=txt_font)

    return np.array(img_to_draw)[:, :, ::-1]


def random_hex_color():
    """Generates a random hex color code.

    Returns:
    A string representing a random hex color code.
    """

    # Generate three random integers for red, green, and blue values.
    red = hex(random.randint(0, 255))[2:]
    green = hex(random.randint(0, 255))[2:]
    blue = hex(random.randint(0, 255))[2:]

    # Ensure each value has two characters (prepend a "0" if necessary).
    if len(red) == 1:
        red = '0' + red
    if len(green) == 1:
        green = '0' + green
    if len(blue) == 1:
        blue = '0' + blue

    # Combine the three values into a hex code.
    return '#' + red + green + blue
