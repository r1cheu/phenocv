from pathlib import Path

import numpy as np
import pandas as pd

from phenocv.utils import scandir


def format_detection_results(dir_path,
                             suffix='.csv',
                             save_out=False,
                             date_re=r'(\d{8})',
                             id_re=r'(GP\d{3})'):
    if isinstance(dir_path, (str, Path)):
        dir_path = Path(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    file_paths = scandir(dir_path, suffix=suffix, recursive=True)
    data_list = []

    for file_path in file_paths:
        file_path = dir_path / file_path
        data = pd.read_csv(file_path, header=None, names=['source', 'values'])
        data['date'] = pd.to_datetime(
            data['source'].str.extract(date_re, expand=False), format='%Y%m%d')
        data['ID'] = data['source'].str.extract(id_re, expand=False)
        data_list.append(data)

    data_list = pd.concat(data_list, ignore_index=True)
    if save_out:
        data_list.to_csv(index=False)

    return data_list


def ascent_idx(values, window_size=5):
    # check values if np.ndarray
    if not isinstance(values, np.ndarray):
        raise ValueError(
            f'values should be {np.ndarray} but got {type(values)}')
    ascent_values = (values[1:] > values[:-1])
    ascent_index = np.where(
        np.convolve(ascent_values, np.ones(window_size), mode='valid') ==
        window_size)[0][0]

    # handle if ascent_idx==0
    if values[ascent_index] == 0:
        ascent_index += 1

    return ascent_index
