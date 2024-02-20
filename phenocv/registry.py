from typing import Tuple, Union

import catalogue
from confection import Config, registry

from phenocv.predict import YoloSahiPanicleUavPredictor
from phenocv.process import (IdDateFormatter, NaiveFormatter, PanicleExtractor,
                             PaniclePostprocessor, PanicleUavPreprocessor)

registry.preprocessor = catalogue.create("phenocv", "preprocessor")
registry.predictor = catalogue.create("phenocv", "predictor")
registry.formatter = catalogue.create("phenocv", "formatter")
registry.postprocessor = catalogue.create("phenocv", "postprocessor")
registry.extractor = catalogue.create("phenocv", "extractor")


@registry.preprocessor.register('panicle_uav')
def panicle_uav_preprocessor(
    width: int = 3800,
    height: int = 2000,
    window_size: int = 100,
) -> PanicleUavPreprocessor:
    return PanicleUavPreprocessor(
        width=width,
        height=height,
        window_size=window_size,
    )


@registry.predictor.register('yolo_sahi_panicle_uav')
def yolo_sahi_panicle_uav_predictor(
    model_type: str,
    model_weight: str,
    device: int | Tuple[int] = 0,
    conf=0.25,
    iou=0.7,
    slice_height=1024,
    slice_width=1024,
    overlap_height_ratio=0.25,
    overlap_width_ratio=0.25,
) -> YoloSahiPanicleUavPredictor:

    return YoloSahiPanicleUavPredictor(
        model_type=model_type,
        model_weight=model_weight,
        device=device,
        conf=conf,
        iou=iou,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )


@registry.formatter.register('id_date')
def id_date_formatter(
    id_pattern: str,
    date_pattern: str = r'\d{8}',
) -> IdDateFormatter:
    return IdDateFormatter(
        id_pattern=id_pattern,
        date_pattern=date_pattern,
    )


@registry.formatter.register('naive')
def naive_formatter() -> NaiveFormatter:
    return NaiveFormatter()


@registry.postprocessor.register('panicle')
def panicle_postprocessor(
    start_date: Union[str, int],
    end_date: Union[str, int],
    seeding_date: Union[str, int],
) -> PaniclePostprocessor:
    return PaniclePostprocessor(
        start_date=start_date,
        end_date=end_date,
        seeding_date=seeding_date,
    )


@registry.extractor.register('panicle')
def panicle_extractor(
    seeding_date, heading_stage=(0.1, 0.8), percents=(0.1, 0.3, 0.5, 0.8)
) -> PanicleExtractor:
    return PanicleExtractor(
        seeding_date=seeding_date,
        heading_stage=heading_stage,
        percents=percents)


Registry = registry

__all__ = ['Registry', 'Config']
