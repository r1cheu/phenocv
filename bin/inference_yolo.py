#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from phenocv import utils


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        'input_dir', type=str, default=None, help='input image dir')
    parser.add_argument(
        'weights', type=str, default=None, help='model weights')
    parser.add_argument('--save-image', action='store_true', help='save image')
    parser.add_argument(
        '--suffix', type=str, default='jpg', help='image suffix')
    parser.add_argument(
        '--conf-thr', type=float, default=0.3, help='score threshold')
    parser.add_argument(
        '--iou-thr', type=float, default=0.5, help='iou threshold')
    parser.add_argument(
        '--output-dir', type=str, default=None, help='output image dir')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='device to use')
    parser.add_argument(
        '--batch-size', type=int, default=16, help='batch size')
    return parser.parse_args()


def main():
    args = get_args()

    # prepare directories and image paths
    input_dir, output_dir = utils.prepare_io_dir(args.input_dir,
                                                 args.output_dir)

    # get image paths
    img_paths = utils.scandir(input_dir, suffix=args.suffix, recursive=False)
    img_paths = sorted(img_paths)  # sort the img_paths

    # split the img_path to step size of batch_size
    img_paths = [
        img_paths[i:i + args.batch_size]
        for i in range(0, len(img_paths), args.batch_size)
    ]
    model = YOLO(args.weights).to(args.device)

    pbar = tqdm(img_paths, total=len(img_paths))
    for img_path in pbar:
        pbar.set_description(f'Predict on batch of {len(img_path)} images')
        imgs = [cv2.imread(str(input_dir / i)) for i in img_path]

        results = model.predict(
            imgs, conf=args.conf_thr, iou=args.iou_thr, verbose=False)
        for img, result, ipath in zip(imgs, results, img_path):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if args.save_image:
                utils.save_pred(img, result.boxes.xyxy,
                                str(output_dir / Path(ipath).stem) + '.png')

            with open(output_dir / 'summary.csv', 'a') as file:
                file.write(
                    f'{str(input_dir/ipath)},{result.boxes.data.shape[0]}\n')

    print(f"Prediction finished!\nCheck the results at '{output_dir}'")


if __name__ == '__main__':
    main()
