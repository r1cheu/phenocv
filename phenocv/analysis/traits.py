from abc import ABCMeta
from pathlib import Path
from pprint import pprint
from typing import Optional, Union

from tqdm import tqdm

from phenocv.registry import Config, Register
from phenocv.utils import (Results, check_path, get_config_path, save_img,
                           scandir)


class Analyzer(metaclass=ABCMeta):

    def __init__(self, cfg: Path | str):

        cfg_path = get_config_path(cfg)
        self.cfg = Config().from_disk(cfg_path)
        self.traits = {}
        resolver = Register.resolve(self.cfg)
        for name, obj in resolver.items():
            setattr(self, name, obj)

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


class PanicleAnalyzer(Analyzer):

    def __init__(
        self,
        cfg: Path | str,
        save_file: Optional[Path | str] = None,
        out_dir_suffix: str = '_result',
    ):

        super().__init__(cfg)
        self.out_dir_suffix = out_dir_suffix
        self.save_file = Path(save_file) if save_file is not None else None

        self._check_attrs(attrs=[
            'preprocessor', 'predictor', 'formatter', 'postprocessor',
            'extractor'])

    def prepare_dir(self, img_dir: Union[str, Path]):
        check_path(img_dir)
        img_dir = Path(img_dir)
        _id = img_dir.name
        result_dir = img_dir.parent / (img_dir.name + self.out_dir_suffix)
        result_dir.mkdir()

        return img_dir, result_dir, _id

    def __call__(
        self,
        img_dir: Union[str, Path],
        img_suffix: str = 'jpg',
        classes: Optional[str] = 'panicle',
        save_pred: bool = False,
    ):

        img_dir, result_dir, _id = self.prepare_dir(img_dir)
        img_paths = sorted(list(scandir(img_dir, suffix=img_suffix)))

        if save_pred:
            pred_dir = result_dir / 'pred'
            pred_dir.mkdir()

        raw_csv = result_dir / f'{_id}_raw.csv'
        interp_csv = result_dir / f'{_id}_interp.csv'
        _img = result_dir / f'{_id}.png'

        pbar = tqdm(
            img_paths, desc="Starting Traits Processing", dynamic_ncols=True)
        for i, img_path in enumerate(pbar):
            result = self.predict(str(img_dir / img_path))
            if save_pred:
                save_img(
                    result.plot(conf=True, labels=True, line_width=5),
                    pred_dir / img_path)

            num_boxes = result.num_bbox[classes] if (
                classes in result.num_bbox) else (len(result))

            self.formatter.update(dict(
                source=img_path,
                value=num_boxes,
            ))

            # update the progress bar with current image info
            pbar.set_description(f"Processing {img_path}")

        self.traits = self.postprocess(raw_csv, interp_csv, self.save_file)

        self.extractor.plot(_img)
        self.clear()

        return self.traits
