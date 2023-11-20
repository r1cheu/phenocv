import os
import os.path as osp
from pathlib import Path

import torch
from torchvision.io import write_png
from torchvision.utils import draw_bounding_boxes


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


def prepare_io_dir(input_dir, output_dir):
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

    try:
        output_dir.mkdir()
    except FileExistsError:
        print(f'{output_dir} already exists, remove it first.')
        exit(1)
    return input_dir, output_dir


def save_pred(image, bboxes, file_name: str):
    image = image.transpose(2, 0, 1)
    image = draw_bounding_boxes(
        torch.from_numpy(image), bboxes, colors='red', fill=True, width=3)
    write_png(image, file_name)
