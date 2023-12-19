#!/usr/bin/env python3
from argparse import ArgumentParser

from phenocv.convert import YOLO2COCO


def main():
    parser = ArgumentParser('Datasets converter from YOLOV5 to COCO')
    parser.add_argument(
        'data_dir',
        type=str,
        default='datasets/YOLOV5',
        help='Dataset root path')
    parser.add_argument(
        '--mode_list',
        type=str,
        default='train,val',
        help='generate which mode')
    args = parser.parse_args()

    converter = YOLO2COCO(args.data_dir)
    converter(mode_list=args.mode_list.split(','))


if __name__ == '__main__':
    main()
