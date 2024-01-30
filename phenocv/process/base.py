from typing import Union, Dict

import pandas as pd

from phenocv.utils import Results


class Processor:
    """The base class for all image extractors. Preprocessor first initializes
    a deep learning model if needed, then process the image with the model
    or algorithms.

    Attributes:
        _results (Union[pd.DataFrame, Results, Dict]): The after processed
            results.

    Methods:
        __call__: Processes the image with the model or algorithms return the
            image in results form.
        results: Returns the results of the image processing.
    """

    def __init__(self):
        self._result = []

    def process(self, *arg, **kwargs):
        raise NotImplementedError

    @property
    def result(self) -> Union[pd.DataFrame, Results, Dict]:
        """Returns the results of the image processing.
        Raises:
            RuntimeError: If the image has not been processed yet.
        Returns:
            The results of the image processing.
        """

        if isinstance(self._result, (Results, pd.DataFrame, Dict)):
            return self._result
        else:
            raise RuntimeError('The data has not been processed yet')

    def save(self, path):
        if isinstance(self.result, Results):
            raise TypeError('Results cannot be saved')

        if isinstance(self.result, pd.DataFrame):
            self.result.to_csv(path, index=False)

        if isinstance(self.result, Dict):
            with open(path, 'a') as f:
                f.write(','.join(map(str, self.result.values())) + '\n')
