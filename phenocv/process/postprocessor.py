from typing import Union

import numpy as np
import pandas as pd

from phenocv.utils import prepare_df

from .base import Processor


class PaniclePostprocessor(Processor):
    """
    A class for post-processing panicle data.

    Args:
        start_date (Union[str, int]): The start date of the data range.
        end_date (Union[str, int]): The end date of the data range.
        seeding_date (Union[str, int]): The seeding date.
    """

    def __init__(self, start_date: Union[str, int], end_date: Union[str, int],
                 seeding_date: Union[str, int]):
        super().__init__()

        self._start_date = str(start_date)
        self._end_date = str(end_date)
        self.seeding_date = pd.to_datetime(str(seeding_date))

    def __call__(self, data):
        data = prepare_df(data)
        self._result = self.process(data)
        return self.result

    def process(self, data):
        """
        Interpolate missing values in the data.

        Args:
            data (pd.DataFrame): The data to be interpolated.

        Returns:
            pd.DataFrame: The interpolated data.
        """
        # create a dataframe with all the dates, handle the missing values
        full_date = pd.Series(
            pd.date_range(self._start_date, self._end_date), name='date')

        _result = pd.merge(full_date, data, how='left', on='date')

        _result['days'] = (_result['date'] - self.seeding_date).dt.days
        _result['interpolate'] = np.isnan(_result['value'])

        _result['value'] = _result['value'].interpolate(method='linear')
        _result['value'] = _result['value'].bfill()  # fill the first value

        _result['id'] = _result['id'].bfill()  # convert to int
        _result['id'] = _result['id'].ffill()

        _result = _result[['date', 'days', 'id', 'value', 'interpolate']]
        return _result
