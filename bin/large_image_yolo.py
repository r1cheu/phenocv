#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path

import cv2
from sahi.slicing import slice_image
from tqdm import tqdm
from ultralytics import YOLO

from phenocv import postprocess, utils


def get_args():
    args = ArgumentParser()
    args.add_argument(
        'input_dir', type=str, default=None, help='input image dir')
    args.add_argument('weights', type=str, default=None, help='model weights')
    args.add_argument('--save_image', action='store_true', help='save image')
    args.add_argument('--suffix', type=str, default='jpg', help='image suffix')
    args.add_argument(
        '--output_dir', type=str, default=None, help='output image dir')
    args.add_argument(
        '--device', type=str, default='cude:0', help='device to use')
    args.add_argument(
        '--conf-thr', type=float, default=0.3, help='score threshold')
    args.add_argument(
        '--overlap-iou-thr', type=float, default=0.25, help='iou threshold')
    args.add_argument(
        '--patch-size', type=int, default=1000, help='patch size')
    args.add_argument(
        '--patch-overlap', type=float, default=0.25, help='patch overlap')
    args.add_argument('--batch-size', type=int, default=16, help='batch size')
    return args.parse_args()


def main():
    args = get_args()

    # prepare directories and image paths
    input_dir, output_dir = utils.prepare_io_dir(args.input_dir,
                                                 args.output_dir)

    # get image paths
    img_paths = utils.scandir(input_dir, suffix=args.suffix, recursive=False)

    model = YOLO(args.weights)

    pbar = tqdm(img_paths)
    for img_path in pbar:
        pbar.set_description(f'Predict on {img_path}')
        img = cv2.imread(str(input_dir / img_path))
        height, width = img.shape[:2]

        sliced_image_obj = slice_image(
            img,
            slice_height=args.patch_size,
            slice_width=args.patch_size,
            auto_slice_resolution=False,
            overlap_height_ratio=args.patch_overlap,
            overlap_width_ratio=args.patch_overlap)

        slice_results = []
        start = 0
        while True:
            end = min(start + args.batch_size, len(sliced_image_obj))
            images = []
            for sliced_image in sliced_image_obj.images[start:end]:
                images.append(sliced_image)
            slice_results.extend(
                model.predict(images, conf=0.3, iou=0.5, verbose=False))

            if end >= len(sliced_image_obj):
                break
            start += args.batch_size

        nms_result = postprocess.merge_results_by_nms(
            slice_results,
            sliced_image_obj.starting_pixels,
            src_image_shape=(height, width),
            iou_thres=args.overlap_iou_thr)
        if args.save_image:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            utils.save_pred(img, nms_result.xyxy,
                            str(output_dir / Path(img_path).stem) + '.png')

        with open(output_dir / 'summary.csv', 'a') as file:
            file.write(f'{img_path},{nms_result.data.shape[0]}\n')

    print(f"Prediction finished!\nCheck the results at '{output_dir}'")


if __name__ == '__main__':
    main()
