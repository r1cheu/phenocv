import os
import random
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms
from torchvision.utils import _parse_colors
from ultralytics.engine.results import Boxes, Results


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
            label.append(f'{i + 1}-{j + 1}')
    return np.array(label)


def compute_dist(x, x_prime):
    """Compute the Manhattan distance between two arrays after removing
    elements where either array is equal to 0.

    Parameters:
    i (numpy.ndarray): The first array.
    next (numpy.ndarray): The second array.

    Returns:
    numpy.ndarray: The absolute difference between the non-zero elements of
        the two arrays.
    """
    index = np.any(np.stack((x == 0, x_prime == 0)), axis=0)

    x = x[~index]
    x_prime = x_prime[~index]
    return np.abs(x_prime - x)


def make_label_box_map(label, boxes):
    """Create a dictionary mapping labels to boxes.

    Args:
        label (np.ndarray): List of labels.
        boxes (Tensor): List of boxes.

    Returns:
        dict: Dictionary mapping labels to boxes.
    """
    map_dict = dict()

    for _label, box in zip(label, boxes):
        map_dict[_label] = box

    return map_dict


def cut_bbox(img, bbox, scale=0):
    """Cuts out a bounding box region from an image.

    Parameters:
    img (numpy.ndarray): The path image.
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
    line_width: Optional[int] = None,
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

    if image.shape[-1] == 1:
        image = np.tile(image, (3, 1, 1))
    ndarr = image[:, :, ::-1].copy()
    img_to_draw = Image.fromarray(ndarr)
    img_boxes = boxes.tolist()

    width = int(ndarr.shape[0] // 500) if line_width is None else line_width
    font_size = int(ndarr.shape[0] // 50) if font_size is None else font_size

    txt_font = ImageFont.truetype(font, size=font_size)

    if fill:
        draw = ImageDraw.Draw(img_to_draw, 'RGBA')
    else:
        draw = ImageDraw.Draw(img_to_draw)

    for bbox, color, label in zip(img_boxes, colors,
                                  labels):  # type: ignore[arg-type]

        fill_color = color + (
            100, ) if fill else None  # Set fill color conditionally
        if len(bbox) != 4:
            draw.polygon(bbox, width=width, outline=color, fill=fill_color)
        else:
            draw.rectangle(bbox, width=width, outline=color, fill=fill_color)

        if label is not None:
            w, h = txt_font.getsize(label)  # text width, height
            outside = bbox[1] - h >= 0  # label fits outside box
            draw.rectangle(
                (bbox[0], bbox[1] - h if outside else bbox[1], bbox[0] + w + 1,
                 bbox[1] + 1 if outside else bbox[1] + h + 1),
                fill=color,
            )
            font_color = decide_font_color(*color)
            draw.text((bbox[0], bbox[1] - h if outside else bbox[1]),
                      label,
                      fill=font_color,
                      font=txt_font)

    return np.array(img_to_draw)[:, :, ::-1]


def random_rgb_color():
    """Generates a random rgb color code.

    Returns:
    A string representing a random rgb color code.
    """
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)

    return (red, green, blue)


def mask2rbox(mask):
    """Convert a binary mask to a rotated bounding box.

    Parameters:
    mask (ndarray): Binary mask representing the object.

    Returns:
    ndarray: Rotated bounding box coordinates.
    """
    y, x = np.nonzero(mask)
    points = np.stack([x, y], axis=-1)
    rec = cv2.minAreaRect(points)
    r_bbox = cv2.boxPoints(rec)
    r_bbox = r_bbox.reshape(1, -1).squeeze()
    return r_bbox


def r_bbox2poly(bbox):
    """Draw oriented bounding boxes on the axes.

    Args:
        ax (matplotlib.Axes): The path axes.
        bboxes (ndarray): The path bounding boxes with the shape
            of (n, 5).
        color (list[tuple] | matplotlib.color): the colors for each
            bounding boxes.
        alpha (float): Transparency of bounding boxes. Default: 0.8.
        thickness (int): Thickness of lines. Default: 2.

    Returns:
        matplotlib.Axes: The result axes.
    """
    xc, yc, w, h, ag = bbox
    wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
    hx, hy = h / 2 * np.sin(ag), h / 2 * np.cos(ag)
    p1 = (xc - wx - hx, yc - wy - hy)
    p2 = (xc + wx - hx, yc + wy - hy)
    p3 = (xc + wx + hx, yc + wy + hy)
    p4 = (xc - wx + hx, yc - wy + hy)
    poly = [p1, p2, p3, p4]
    return poly


def shift_predictions(results: List[Results], offsets: Sequence[Tuple[int,
                                                                      int]],
                      src_image_shape) -> Boxes:
    """Shifts the predicted bounding boxes based on the given offsets.

    Args:
        results (List[Results]): List of Results objects containing predicted
        bounding boxes.
        offsets (Sequence[Tuple[int, int]]): Sequence of (x, y) offsets to
        shift the bounding boxes.
        src_image_shape: Shape of the source image.

    Returns:
        Boxes: Shifted bounding boxes.

    Raises:
        ImportError: If the 'sahi' module is not installed.
    """
    try:
        from sahi.slicing import shift_bboxes
    except ImportError:
        raise ImportError('Please run "pip install -U sahi" '
                          'to install sahi first for large image inference.')

    assert len(results) == len(
        offsets), 'The `results` should has the ' 'same length with `offsets`.'

    shifted_predictions = []
    for result, offset in zip(results, offsets):

        pred_inst = result.boxes
        if pred_inst.xyxy.numel() == 0:
            shifted_bboxes = torch.empty((0, 6), device=pred_inst.xyxy.device)
            shifted_predictions.append(shifted_bboxes)
            continue

        shifted_bboxes = shift_bboxes(pred_inst.xyxy, offset)
        shifted_bboxes = torch.cat([shifted_bboxes, pred_inst.data[:, -2:]],
                                   dim=1)

        shifted_predictions.append(shifted_bboxes)
    shifted_predictions = torch.cat(shifted_predictions, dim=0)
    shifted_predictions = Boxes(shifted_predictions, src_image_shape)

    return shifted_predictions


def merge_results_by_nms(results: List[Results], offsets: Sequence[Tuple[int,
                                                                         int]],
                         src_image_shape, iou_thres) -> Boxes:
    """Merge the results of object detection by applying non-maximum
    suppression (NMS).

    Args:
        results (List[Results]): List of Results objects containing the
        predicted bounding boxes and confidence scores.
        offsets (Sequence[Tuple[int, int]]): Sequence of offset tuples for
        each result.
        src_image_shape: Shape of the source image.
        iou_thres: IoU (Intersection over Union) threshold for NMS.

    Returns:
        Boxes: Merged bounding boxes after applying NMS.
    """
    shifted_instances = shift_predictions(results, offsets, src_image_shape)

    keep = nms(shifted_instances.xyxy, shifted_instances.conf, iou_thres)
    bboxes = shifted_instances.data[keep].clone()
    return bboxes


def decide_font_color(r, g, b):
    # 计算对比度
    contrast = ((r * 299) + (g * 587) + (b * 114)) / 1000 / 255
    # 根据对比度决定字体颜色
    if contrast > 0.5:
        return 0, 0, 0
    else:
        return 255, 255, 255
