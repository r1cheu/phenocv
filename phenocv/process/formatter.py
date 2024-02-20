import re
from abc import ABCMeta, abstractmethod
from typing import Any, Dict

import pandas as pd

from .base import Processor


class Formatter(Processor, metaclass=ABCMeta):

    def update(self, item: Dict[str, Any]):
        item = self._format(item)
        self._result.append(item)

    def clear(self):
        self._result = []

    def __call__(self):

        self._result = pd.DataFrame(self._result)
        return self.result

    @abstractmethod
    def _format(self, item: Dict[str, Any]) -> Dict[str, Any]:
        pass


class NaiveFormatter(Formatter):

    def _format(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return item


class IdDateFormatter(Formatter):

    def __init__(
        self,
        id_pattern: str,
        date_pattern: str = r'\d{8}',
    ):
        super().__init__()
        self.id_pattern = re.compile(id_pattern)
        self.date_pattern = re.compile(date_pattern)

    @staticmethod
    def extract_data(pattern: re.Pattern, source: str, data_type: str):
        try:
            return pattern.search(source).group()
        except AttributeError:
            raise ValueError(f'Cannot find {data_type} using regex pattern'
                             f' {pattern.pattern} in {source}')

    def _format(self, item: Dict[str, Any]) -> Dict[str, Any]:

        source = item['source']
        value = item['value']

        _id = self.extract_data(self.id_pattern, source, 'id')
        date = self.extract_data(self.date_pattern, source, 'date')

        return {'source': source, 'id': _id, 'date': date, 'value': value}
