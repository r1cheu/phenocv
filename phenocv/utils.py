import copy
import os
import os.path as osp
import subprocess
import sys
import time
import warnings
from argparse import Action, ArgumentParser, Namespace
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision.io import write_png
from torchvision.utils import _parse_colors


def scandir(dir_path, suffix=None, recursive=False, case_sensitive=True):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str | :obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Defaults to None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Defaults to False.
        case_sensitive (bool, optional) : If set to False, ignore the case of
            suffix. Defaults to True.

    Returns:
        A generator for all the interested files with relative paths.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    if suffix is not None and not case_sensitive:
        suffix = suffix.lower() if isinstance(suffix, str) else tuple(
            item.lower() for item in suffix)

    root = dir_path

    def _scandir(dir_path, suffix, recursive, case_sensitive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                _rel_path = rel_path if case_sensitive else rel_path.lower()
                if suffix is None or _rel_path.endswith(suffix):
                    yield rel_path
            elif recursive and os.path.isdir(entry.path):
                # scan recursively if entry.path is a directory
                yield from _scandir(entry.path, suffix, recursive,
                                    case_sensitive)

    return _scandir(dir_path, suffix, recursive, case_sensitive)


def match_yolo_annotations(ann_dir, img_dir, img_suffix='.jpg'):

    ann_dir = Path(ann_dir)
    img_dir = Path(img_dir)

    anns = scandir(ann_dir, suffix='.txt', recursive=False)
    imgs = scandir(img_dir, suffix=img_suffix, recursive=False)

    anns = sorted(list(anns))
    imgs = sorted(list(imgs))

    if len(anns) == 0 or len(imgs) == 0:
        raise FileNotFoundError('No annotations or images found.')

    for ann in anns:
        img = ann.replace('.txt', img_suffix)
        img = img_dir / img
        # remove img
        if not img.exists():
            ann = ann_dir / ann
            ann.unlink()

    for img in imgs:
        ann = img.replace(img_suffix, '.txt')
        ann = ann_dir / ann
        # remove ann
        if not ann.exists():
            img = img_dir / img
            img.unlink()


def prepare_io_dir(input_dir, output_dir, resume=False):
    """Prepare input and output directories.

    Args:
        input_dir (str | :obj:`Path`): Path of the input directory.
        output_dir (str | :obj:`Path`): Path of the output directory.

    Returns:
        A tuple containing the input and output directories.
    """
    if not isinstance(input_dir, (str, Path)):
        raise TypeError('"input_dir" must be a string or Path object')
    input_dir = Path(input_dir).resolve()

    if output_dir is None:
        output_dir = input_dir.with_name(f'{input_dir.name}_out')
    if not isinstance(output_dir, (str, Path)):
        raise TypeError('"output_dir" must be a string or Path object')
    output_dir = Path(output_dir).resolve()
    if not resume:
        try:
            output_dir.mkdir()
        except FileExistsError:
            raise FileExistsError(f'{output_dir} already exists, remove it ' +
                                  'or use --resume')
    return input_dir, output_dir


def save_pred(image, bboxes, file_name: str):
    image = image.transpose(2, 0, 1)
    image = draw_bounding_boxes(
        torch.from_numpy(image), bboxes, colors='red', fill=True, width=3)
    write_png(image, file_name)


def exec_par(cmds, max_proc=None, verbose=False):
    total = len(cmds)
    finished = 0
    running = 0
    p = []

    if max_proc is None:
        max_proc = len(cmds)

    if max_proc == 1:
        while finished < total:
            if verbose:
                print(cmds[finished], file=sys.stderr)
            op = subprocess.Popen(cmds[finished], shell=True)
            os.waitpid(op.pid, 0)
            finished += 1

    else:
        while finished + running < total:
            # launch jobs up to max
            while running < max_proc and finished + running < total:
                if verbose:
                    print(cmds[finished + running], file=sys.stderr)
                p.append(
                    subprocess.Popen(cmds[finished + running], shell=True))
                # print 'Running %d' % p[running].pid
                running += 1

            # are any jobs finished
            new_p = []
            for i in range(len(p)):
                if p[i].poll() is not None:
                    running -= 1
                    finished += 1
                else:
                    new_p.append(p[i])

            # if none finished, sleep
            if len(new_p) == len(p):
                time.sleep(1)
            p = new_p

        # wait for all to finish
        for i in range(len(p)):
            p[i].wait()


def check_path(path):

    if not isinstance(path, (str, Path)):
        raise TypeError('The input path must be a string or a Path object,' +
                        f'but got {type(path)}')

    if not Path(path).exists():
        raise FileNotFoundError(f'{path} does not exist.')


def write_file(file_path, content, mode='a'):
    """Write content to file.

    Args:
        file_path (str | :obj:`Path`): Path of the file.
        content (str): Content to be written to the file.
        mode (str, optional): Mode for opening the file. Defaults to 'w'.
    """
    with open(file_path, mode) as f:
        f.write(content)


class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val: str) -> Union[int, float, bool, Any]:
        """parse int/float/bool value in the string."""
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        if val == 'None':
            return None
        return val

    @staticmethod
    def _parse_iterable(val: str) -> Union[list, tuple, Any]:
        """Parse iterable values in the string.

        All elements inside '()' or '[]' are treated as iterable values.

        Args:
            val (str): Value string.

        Returns:
            list | tuple | Any: The expanded list or tuple from the string,
            or single value if no iterable values are found.

        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.

            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count('(') == string.count(')')) and (
                    string.count('[') == string.count(']')), \
                f'Imbalanced brackets exist in {string}'
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if ((char == ',') and (pre.count('(') == pre.count(')'))
                        and (pre.count('[') == pre.count(']'))):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip('\'\"').replace(' ', '')
        is_tuple = False
        if val.startswith('(') and val.endswith(')'):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith('[') and val.endswith(']'):
            val = val[1:-1]
        elif ',' not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1:]

        if is_tuple:
            return tuple(values)

        return values

    def __call__(self,
                 parser: ArgumentParser,
                 namespace: Namespace,
                 values: Union[str, Sequence[Any], None],
                 option_string: str = None):
        """Parse Variables in string and add them into argparser.

        Args:
            parser (ArgumentParser): Argument parser.
            namespace (Namespace): Argument namespace.
            values (Union[str, Sequence[Any], None]): Argument string.
            option_string (list[str], optional): Option string.
                Defaults to None.
        """
        # Copied behavior from `argparse._ExtendAction`.
        options = copy.copy(getattr(namespace, self.dest, None) or {})
        if values is not None:
            for kv in values:
                key, val = kv.split('=', maxsplit=1)
                options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)


def imshow(img: np.ndarray, figsz=(20, 12)):
    fig, ax = plt.subplots(figsize=figsz)
    ax.imshow(img)
    ax.axis('off')
    ax.tick_params(
        bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.show()


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
    width: int = 1,
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
