import albumentations as A
from ultralytics import YOLO
from ultralytics.data.augment import Albumentations
from ultralytics.engine.trainer import BaseTrainer


def callback_custom_albumentations(trainer: BaseTrainer):
    # Build custom albumentations transform
    T = [
        A.Blur(always_apply=False, p=0.1, blur_limit=(3, 7)),
        A.MedianBlur(always_apply=False, p=0.1, blur_limit=(3, 7)),
        A.ToGray(always_apply=False, p=0.01),
        A.CLAHE(
            always_apply=False,
            p=0.01,
            clip_limit=(1, 4.0),
            tile_grid_size=(8, 8)),
        A.RandomBrightnessContrast(
            always_apply=False,
            p=0.1,
            brightness_limit=(-0.2, 0.2),
            contrast_limit=(-0.2, 0.2),
            brightness_by_max=True),
        A.RandomGamma(
            always_apply=False, p=0.0, gamma_limit=(80, 120), eps=None),
        A.ImageCompression(
            always_apply=False,
            p=0.0,
            quality_lower=75,
            quality_upper=100,
            compression_type=0),
    ]

    transform = A.Compose(
        T,
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']),
    )
    # Override albumentations transform in trainer
    for t in trainer.train_loader.dataset.transforms.transforms:
        if isinstance(t, Albumentations):
            print('Overriding default albumentations pipeline...')
            print(f'Original albumentations transform:\n{t.transform}')
            t.transform = transform
            print(f'New albumentations transform:\n{t.transform}')
            break
    trainer.train_loader.reset()


def main():
    model = YOLO('yolov5x6u.pt')
    model.add_callback('on_pretrain_routine_end',
                       callback_custom_albumentations)
    model.train(
        data='~/ProJect/phenocv/configs/yolo/panicle_drone.yaml',
        flipud=0.5,
        imgsz=1280,
        device='0,1,2,3',
    )


if __name__ == '__main__':
    main()
