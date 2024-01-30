"""
This module is used to extract traits from data that passed from the
postprocess module. It takes in a pd.DataFrame or data csv path and returns a
dictionary of traits.
"""
import ast
from functools import partial
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit

from .base import Processor
from .utils import cut_up_curve, remove_smaller, logistic, inverse_logistic


class PanicleExtractor(Processor):

    def __init__(self,
                 seeding_date,
                 heading_stage: str | Tuple[int] = (0.1, 0.8),
                 percents: str | Tuple[float] = (0.1, 0.3, 0.5, 0.8)):
        super().__init__()
        self.seeding_date = pd.to_datetime(seeding_date)
        self._result = dict()

        if isinstance(heading_stage, str):
            heading_stage = ast.literal_eval(heading_stage)
        if isinstance(percents, str):
            percents = ast.literal_eval(percents)

        self.heading_stage = heading_stage
        self.percents = percents
        self.logistic = None

    def __call__(self, data: Union[str, Path, pd.DataFrame]):
        self.data = data
        self.process(data)
        return self.data

    def process(self, data: Union[str, Path, pd.DataFrame]):

        _inverse_logistic, maximum = self.fit_logistic(data)
        self._result['maximum'] = maximum
        self._result['id'] = data['id'].iloc[-1]
        for percent in self.percents:
            self._result[self._percent_format(percent)] = (
                _inverse_logistic(percent * maximum))

        self._cal_heading_stage()

    @staticmethod
    def before_fit(data):

        data = data.copy()
        data = data[data['interpolate'] == False] # noqa

        maximum = data['value'].max()
        min_days = data['days'].min()
        max_days = data['days'].max()

        start, end = cut_up_curve(data)
        growth_curve = data.loc[start:end, :]

        days = growth_curve['days'].to_numpy()
        values = growth_curve['value'].to_numpy()
        mask = remove_smaller(values)

        while True:
            try:
                mask_values = values[mask]
                mask_days = days[mask]
            except IndexError:
                break
            if np.all(mask_values == values):
                break
            else:
                values = mask_values
                days = mask_days
                mask = remove_smaller(values)

        return min_days, max_days, days, values, maximum

    def fit_logistic(self, data):

        min_days, max_days, days, values, maximum = self.before_fit(data)

        params, _ = curve_fit(logistic, days, values,
                              bounds=([maximum, min_days, 0],
                                      [maximum+0.0001, max_days, 10]))

        self.logistic = partial(logistic,
                                K=params[0],
                                x0=params[1],
                                r=params[2])

        return partial(inverse_logistic, *params), int(params[0])

    @staticmethod
    def _percent_format(percent: float):
        return f'{int(percent * 100)}%'

    def _cal_heading_stage(self):
        start_percent = self._percent_format(self.heading_stage[0])
        end_percent = self._percent_format(self.heading_stage[1])

        self._result[
            f'{start_percent}-{end_percent}'] = (self._result[end_percent] -
                                                 self._result[start_percent])

    def clear(self):
        self._result = dict()
        self.logistic = None

    def plot(self, save_path):

        fig, axes = plt.subplots(figsize=(15, 6))
        impute_index = self.data['interpolate'].values

        def _plot(ax, x, y, index, color='black'):
            ax.plot(x[~index], y[~index], 'o', color=color)
            ax.plot(x, y, color=color, linestyle='-')
            ax.plot(x[index], y[index], 'o', color=color,
                    markerfacecolor='white')
            ax.plot(x, self.logistic(x), color='b',
                    linestyle='--')

        def _format_ticks(ax, days, dates):
            selected_days = days[::2]
            selected_dates = dates[::2]

            axes.set_xticks(selected_days)
            axes.set_xticklabels(pd.to_datetime(selected_dates).dt.strftime(
                '%m-%d'), rotation=45)

            ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
            ax.tick_params(axis='x', which='minor', length=2)

            ax.set_xlim(34, 127)

        _plot(axes, self.data['days'], self.data['value'], impute_index)
        _format_ticks(axes, self.data['days'], self.data['date'])
        axes.set_title(self.data['id'].iloc[0])

        for key in [f"{int(i * 100)}%" for i in self.percents]:
            axes.axvline(self.result[key], color='b', linestyle='--')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
