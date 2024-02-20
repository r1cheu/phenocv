import contextlib
import copy
import os
import os.path as osp
import subprocess
import sys
import time
import warnings
from importlib import resources
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import pandas as pd
import requests
import torch
from matplotlib import pyplot as plt
from ultralytics.engine.results import Results as _Results


def prepare_df(pred_data: Union[str, Path, pd.DataFrame]) -> (pd.DataFrame):
    if isinstance(pred_data, str):
        pred_data = Path(pred_data)
    if isinstance(pred_data, Path):
        pred_data = pd.read_csv(pred_data)
    if not isinstance(pred_data, pd.DataFrame):
        raise TypeError(f'pred_data should be str, Path or DataFrame, but got '
                        f' {type(pred_data)}')
    if 'date' in pred_data.columns:
        pred_data['date'] = pd.to_datetime(pred_data['date'])

    return pred_data


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
        A list of all the interested files with relative paths.
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
    found_files = False

    def _scandir(dir_path, suffix, recursive, case_sensitive):
        nonlocal found_files
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                _rel_path = rel_path if case_sensitive else rel_path.lower()
                if suffix is None or _rel_path.endswith(suffix):
                    found_files = True
                    yield rel_path
            elif recursive and os.path.isdir(entry.path):
                # scan recursively if entry.path is a directory
                yield from _scandir(entry.path, suffix, recursive,
                                    case_sensitive)

    files = sorted(list(_scandir(dir_path, suffix, recursive, case_sensitive)))
    files = [str(Path(root) / file) for file in files]
    if not found_files:
        raise FileNotFoundError(f'No files with suffix {suffix} found in the '
                                f'directory {dir_path}. Please check the '
                                f'suffix and directory.')
    return files


def match_yolo_annotations(ann_dir, img_dir, img_suffix='.jpg'):
    ann_dir = Path(ann_dir)
    img_dir = Path(img_dir)

    anns = scandir(ann_dir, suffix='.txt', recursive=False)
    imgs = scandir(img_dir, suffix=img_suffix, recursive=False)

    anns = sorted(list(anns))
    imgs = sorted(list(imgs))

    if len(anns) == 0 or len(imgs) == 0:
        raise FileNotFoundError('No annotations or test_images found.')

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
    """Prepare path and output directories.

    Args:
        input_dir (str | :obj:`Path`): Path of the path directory.
        output_dir (str | :obj:`Path`): Path of the output directory.

    Returns:
        A tuple containing the path and output directories.
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
    if isinstance(path, (str, Path)):
        if not Path(path).exists():
            raise FileNotFoundError(f'{path} does not exist.')
    else:
        raise TypeError('Input must be a string, Path object or a 3D array.')


def read_image(img: Union[str, Path], order='BGR'):
    """Read an image from file or a numpy array.

    Args:
        img (str | :obj:`Path` | :obj:`np.ndarray`): Path of the image file or
            a numpy array.
        order (str, optional): Order of channel. Defaults to 'BGR'.

    Returns:
        np.ndarray: Loaded image array. BGR order.
    """

    if isinstance(img, (str, Path)):

        img = str(img)
        if not Path(img).exists():
            raise FileNotFoundError(f'{img} does not exist.')
        img = cv2.imread(img)

        if order.lower() == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    else:
        raise TypeError('Input must be a string or Path object')

    return img


def save_img(img: np.ndarray, file_path: Union[str, Path], order='BGR'):
    """Save an image to file.

    Args:
        img (np.ndarray): Image array to be saved.
        file_path (str | :obj:`Path`): Path of the image file.
        order (str, optional): Order of channel. Defaults to 'BGR'.
    """
    if order.lower() == 'rgb':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(file_path), img)


class Results(_Results):

    def crop_img(self):
        """Return the cropped image in np.ndarray format. Only works for single
        box case.
        """
        boxes = copy.deepcopy(self.boxes)
        if not boxes.data.shape[0] == 1:
            warnings.warn('Only works for single box case. For multiple '
                          'boxes, it will return the box with the highest '
                          'conf, be careful.')
            box = boxes.xyxy[boxes.conf.argmax()].int().tolist()
        else:
            box = boxes.xyxy[0].int().tolist()
        img = self.orig_img.copy()

        return img[box[1]:box[3], box[0]:box[2]]

    @classmethod
    def from_yolo(cls, results: _Results):
        """Convert original ultralytics Results to custom."""
        return cls(
            orig_img=results.orig_img,
            path=results.path,
            names=results.names,
            boxes=results.boxes.data,
            masks=results.masks,
            probs=results.probs,
            keypoints=results.keypoints,
        )

    def num_bbox(self):
        num_dict = {}
        bbox = copy.deepcopy(self.boxes)
        for k, v in self.names.items():
            num = torch.sum(bbox.cls == k).item()
            num_dict[v] = num
        return num_dict


def check_latest_pypi_version(package_name="phenocv"):
    """
    Returns the latest version of a PyPI package without downloading or
    installing it.

    Parameters:
        package_name (str): The name of the package to find the latest
        version for.

    Returns:
        (str): The latest version of the package.
    """
    with contextlib.suppress(Exception):
        requests.packages.urllib3.disable_warnings(
        )  # Disable the InsecureRequestWarning
        response = requests.get(
            f"https://pypi.org/pypi/"
            f"{package_name}/json", timeout=3)
        if response.status_code == 200:
            return response.json()["info"]["version"]


def get_config_path(config_name: Union[str, Path]):

    config_name = Path(config_name) if isinstance(config_name,
                                                  str) else (config_name)

    def check_config(path, config):
        path = Path(str(resources.files('phenocv'))) / path
        config = path / config

        if not config.exists():
            raise FileNotFoundError(f'{config} does not exist.')

        return config

    if config_name.exists():
        return config_name

    if config_name.suffix == '.yaml':
        return check_config('configs/yolo', config_name)

    elif config_name.suffix == '.cfg':
        return check_config('configs/', config_name)

    else:
        raise ValueError('Wrong config, config should end with .yaml or '
                         f'.cfg, but got {config_name}')


def imshow(img: np.ndarray, figsz=(20, 12)):
    fig, ax = plt.subplots(figsize=figsz)
    fig.set_facecolor('#181818')
    ax.imshow(img)
    ax.axis('off')
    ax.tick_params(
        bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.show()
