import re
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


class TraitExtractor(metaclass=ABCMeta):
    """Abstract base class for trait extraction.

    Args:
        data_path (Path): The path to the trait data file.
            The file should be tab-delimited.
            The header should be [source, values, date, ID].

    Attributes:
        _data (DataFrame): The trait data.
        _result (dict): The extracted traits.

    Methods:
        _data_filtering: Abstract method for filtering the data.
        _trait_extract: Abstract method for extracting traits.
        get_trait: Method to get the extracted traits.
        get_data: Method to get the trait data.
    """

    def __init__(self, data_path: Path):

        self._data = self._read_data(data_path)
        self._result = {}

    @abstractmethod
    def _read_data(self, data_path):
        raise NotImplementedError('filtering has not been Implemented yet')

    @abstractmethod
    def _data_filtering(self):
        raise NotImplementedError('filtering has not been Implemented yet')

    @abstractmethod
    def _trait_extract(self):
        raise NotImplementedError('tarit_extract has not been Implemented yes')

    def get_trait(self):
        """Get the extracted traits.

        Returns:
            dict: The extracted traits.
        """
        self._data_filtering()
        self._trait_extract()
        return self._result

    def get_data(self):
        """Get the trait data.

        Returns:
            DataFrame: The trait data.
        """
        return self._data


class HeadingDateExtractor(TraitExtractor):
    """A class for extracting heading date traits from data.

    Parameters:
    - data_path (Path):
        The path to the data file.
    - seeding_date (Union[str, int]):
        The seeding date in the format 'YYYY-MM-DD'.
    - heading_stage (Tuple[float, float]):
        The range of heading stage as a tuple of two floats.
    - percents (Tuple[float]):
        The list of percents to calculate.


    Attributes:
    - _seeding_date (numpy.datetime64): The seeding date.
    - _precents (Tuple[float]): The list of percents to calculate.
    - _heading_stage (Tuple[float]): The range of heading stage.
    - _filtering (bool): Indicates if data filtering has been performed.
    - _result (Dict[str, Any]):
        The dictionary to store the extracted traits.

    Methods:
    - _data_filtering():
        Performs data filtering.
    - _trait_extract():
        Extracts the heading date traits.
    - _round_percent(values: np.ndarray, percent: float):
        Rounds the percent value to the nearest index.
    - _percent(percent: float):
        Converts the percent value to a string representation.
    - _cal_heading_stage(): Calculates the heading stage traits.
    """

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
        self._precents = sorted(percents)
        self._heading_stage = sorted(heading_stage)

        super().__init__(data_path=data_path)
        check_data = (self._data.columns == ['source', 'values', 'date',
                                             'ID']).all()

        if not check_data:
            raise ValueError(
                "The trait file's header is [source, values, date, ID],"
                f'but found {list(self._data.columns)}')

    def _read_data(self, data_path):
        with open(data_path) as f:
            first_line = f.readline()
            first_line = first_line.strip().split('\t')

        check_data = (first_line == ['source', 'values', 'date', 'ID'])

        if not check_data:
            raise ValueError(
                "The trait file's header is [source, values, date, ID],"
                f'but found {first_line}')

        data = pd.read_table(data_path, parse_dates=['date'])

        return data

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

        for idx, percent in enumerate(self._precents):
            if idx == 0:
                pre_idx = 0
            index = self._round_percent(values, percent, pre_idx)
            _date = (date[index] - self._seeding_date).astype(int)
            pre_idx = index
            self._result[self._percent_format(percent)] = _date

        self._cal_heading_stage()

    def _round_percent(self, values: np.ndarray, percent: float, pre_idx):

        max_index = values.argmax()

        if percent < 0 or percent > 1:
            raise ValueError(
                f'percent should between 0 and 1, but got {percent}')

        percent_value = self._result['max'] * percent
        percent_index = np.abs(values[pre_idx:max_index] -
                               percent_value).argmin() + pre_idx

        return percent_index

    @staticmethod
    def _percent_format(percent: float):
        return f'{int(percent * 100)}%'

    def _cal_heading_stage(self):

        hstart = self._percent_format(self._heading_stage[0])
        hend = self._percent_format(self._heading_stage[1])

        self._result[
            f'{hstart}-{hend}'] = self._result[hend] - self._result[hstart]
