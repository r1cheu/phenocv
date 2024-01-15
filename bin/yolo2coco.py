#!/usr/bin/env python3
from argparse import ArgumentParser

from phenocv.convert import YOLOtoCOCO


def main():
    parser = ArgumentParser('Datasets converter from YOLOV5 to COCO')
    parser.add_argument(
        'data_dir',
        type=str,
        default='datasets/YOLOV5',
        help='Dataset root path')
    args = parser.parse_args()

    YOLOtoCOCO(args.data_dir)()


if __name__ == '__main__':
    main()
