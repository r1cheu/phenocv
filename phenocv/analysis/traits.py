import os.path as osp
from abc import ABCMeta
from pathlib import Path
from pprint import pprint
from typing import Optional, Union

from tqdm import tqdm

from phenocv.registry import Config, Registry
from phenocv.utils import (Results, check_path, get_config_path, save_img,
                           scandir)


class Analyzer(metaclass=ABCMeta):
    Attrs = []

    def __init__(
        self,
        cfg: Path | str,
        save_file: Optional[Path | str] = None,
        override: Optional[dict] = None,
    ):

        cfg_path = get_config_path(cfg)
        self.cfg = Config().from_disk(cfg_path)

        if override is not None:
            override = Config(override)
            self.cfg = Config(self.cfg).merge(override)

        self.traits = {}
        resolver = Registry.resolve(self.cfg)
        for name, obj in resolver.items():
            setattr(self, name, obj)

        self.save_file = Path(save_file) if save_file is not None else None

        if len(self.Attrs) != 0:
            self._check_attrs(attrs=self.Attrs)

    def _check_attrs(self, attrs: list[str] = None):

        if attrs is None:
            return
        else:
            for attr in attrs:
                if not hasattr(self, attr):
                    print(
                        f'{attr} is not defined, '
                        f'please check your config file.\n Your config is: \n')
                    pprint(self.cfg)
                    raise AttributeError

    @staticmethod
    def prepare_dir(img_dir: Union[str, Path], save_pred=False):
        check_path(img_dir)
        _img_dir = Path(img_dir)
        _name = _img_dir.name
        result_dir = _img_dir.parent / (_name + '_result')
        result_dir.mkdir(exist_ok=True)

        if save_pred:
            pred_dir = result_dir / 'pred'
            pred_dir.mkdir(exist_ok=True)
        else:
            pred_dir = None

        return _img_dir, result_dir, pred_dir, _name

    def process_images(self, img_paths, classes, pred_dir=None):
        """
        Process the test_images in the given directory.
        """
        pbar = tqdm(
            img_paths, desc="Starting Traits Processing", dynamic_ncols=True)
        for i, img_path in enumerate(pbar):
            img_name = osp.basename(img_path)
            result = self.predict(img_path)
            if pred_dir is not None:
                save_img(
                    result.plot(conf=True, labels=True, line_width=5),
                    pred_dir / img_name,
                    order='rgb')

            num_boxes = result.num_bbox()[classes] if (classes in
                                                       result.num_bbox()) \
                else len(result)

            self.formatter.update(dict(
                source=img_path,
                value=num_boxes,
            ))

            # update the progress bar with current image info
            pbar.set_description(f"Processing {img_name}")
            pbar.set_postfix({f'num of {classes}': num_boxes})

        pbar.close()
        print(
            f'Processing Done, check the result: {pred_dir.parent.absolute()} '
            f'and {self.save_file}')

    def predict(self, img: Union[str, Path]) -> Results:

        result = img
        if hasattr(self, 'preprocessor'):
            result = self.preprocessor(result)

        if hasattr(self, 'predictor'):
            result = self.predictor(result)
        else:
            raise AttributeError('predictor is not defined')

        return result

    def postprocess(
        self,
        save_format: Optional[Path | str] = None,
        save_post: Optional[Path | str] = None,
        save_extract: Optional[Path | str] = None,
    ):
        if hasattr(self, 'formatter'):
            result = self.formatter()

            if save_format is not None and hasattr(self.formatter, 'save'):
                self.formatter.save(save_format)
        else:
            raise AttributeError('formatter is not defined')

        if hasattr(self, 'postprocessor'):
            result = self.postprocessor(result)
            if save_post is not None and hasattr(self.postprocessor, 'save'):
                self.postprocessor.save(save_post)

        if hasattr(self, 'extractor'):
            result = self.extractor(result)
            if save_extract is not None and hasattr(self.extractor, 'save'):
                self.extractor.save(save_extract)

        return result

    def clear(self):
        for attr in dir(self):
            if attr.startswith('_'):
                continue
            if hasattr(getattr(self, attr), 'clear'):
                getattr(self, attr).clear()


# TODO: add a class to simply counts the number of panicles in an image


class PanicleNumAnalyzer(Analyzer):

    Attrs = ['predictor', 'formatter']

    def __call__(self,
                 img_dir: Union[str, Path],
                 classes: Optional[str] = 'panicle',
                 img_suffix: str = 'jpg',
                 save_pred: bool = False):

        img_dir, result_dir, pred_dir, _id = self.prepare_dir(
            img_dir, save_pred=save_pred)
        img_paths = scandir(img_dir, suffix=img_suffix)

        self.process_images(img_paths, classes, pred_dir=pred_dir)
        self.traits = self.postprocess(save_format=self.save_file)
        return self.traits


class PanicleHeadingDateAnalyzer(Analyzer):
    Attrs = [
        'preprocessor', 'predictor', 'formatter', 'postprocessor', 'extractor']

    def __call__(self,
                 img_dir: Union[str, Path],
                 img_suffix: str = 'jpg',
                 classes: Optional[str] = 'panicle',
                 save_pred: bool = False):
        """
        Analyze the test_images in the given directory.
        """

        img_dir, result_dir, pred_dir, _id = self.prepare_dir(
            img_dir, save_pred=save_pred)
        img_paths = scandir(img_dir, suffix=img_suffix)

        raw_csv = result_dir / f'{_id}_raw.csv'
        interp_csv = result_dir / f'{_id}_interp.csv'
        _img = result_dir / f'{_id}.png'

        self.process_images(img_paths, classes, pred_dir=pred_dir)

        self.traits = self.postprocess(raw_csv, interp_csv, self.save_file)
        self.extractor.plot(_img)
        self.clear()

        return self.traits
