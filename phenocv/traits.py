import re
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .utils import scandir


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


class TraitExtractor(metaclass=ABCMeta):

    def __init__(self, data_path: Path):
        data = pd.read_table(data_path, parse_dates=['date'])
        check_data = (data.columns == ['source', 'values', 'date', 'ID']).all()

        if not check_data:
            raise ValueError(
                "The trait file's header is [source, values, date, ID],"
                f'but found {list(data.columns)}')
        self._data = data
        self._result = {}

    @abstractmethod
    def _data_filtering(self):
        raise NotImplementedError('filtering has not been Implemented yet')

    @abstractmethod
    def _trait_extract(self):
        raise NotImplementedError('tarit_extract has not been Implemented yes')

    def get_trait(self):
        self._data_filtering()
        self._trait_extract()
        return self._result

    def get_data(self):
        return self._data


class HeadingDateExtractor(TraitExtractor):

    def __init__(self,
                 data_path: Path,
                 seeding_date: Union[str, int],
                 heading_stage=(0.1, 0.8),
                 percents=(0.1, 0.3, 0.5, 0.8)):

        date_pattern = r'^\d{4}-\d{2}-\d{2}$'
        match = re.match(date_pattern, seeding_date)
        if not match:
            raise ValueError(
                'The format of seeding_date should be YYYY-MM-DD' +
                f', but got {seeding_date}')

        if not all(stage in percents for stage in heading_stage):
            raise ValueError('heading_stage should be included in percents')

        self._seeding_date = np.datetime64(seeding_date)
        self._precents = percents
        self._heading_stage = sorted(heading_stage)

        super().__init__(data_path=data_path)

    def _data_filtering(self):
        max_row = self._data[self._data['values'] == 0].index.max()
        if max_row is not np.nan:
            self._data.loc[:max_row, 'values'] = 0
        self._filtering = True

    def _trait_extract(self):

        if not self._filtering:
            self._data_filtering()

        values = self._data['values'].to_numpy()
        date = self._data['date'].to_numpy().astype('datetime64[D]')

        self._result['max'] = values.max()
        self._result['num_sources'] = len(values)

        for percent in self._precents:
            index = self._round_percent(values, percent)
            _date = (date[index] - self._seeding_date).astype(int)
            self._result[self._percent(percent)] = _date

        self._cal_heading_stage()

    def _round_percent(self, values: np.ndarray, percent: float):

        max_index = values.argmax()

        if percent < 0 or percent > 1:
            raise ValueError(
                f'percent should between 0 and 1, but got {percent}')

        percent_value = self._result['max'] * percent
        percent_index = np.abs(values[:max_index] - percent_value).argmin()

        return percent_index

    @classmethod
    def _percent(self, percent: float):
        return f'{int(percent * 100)}%'

    def _cal_heading_stage(self):

        hstart = self._percent(self._heading_stage[0])
        hend = self._percent(self._heading_stage[1])

        self._result[
            f'{hstart}-{hend}'] = self._result[hend] - self._result[hstart]
