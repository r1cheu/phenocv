#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

from phenocv.convert import COCO2LS


def main():
    parser = ArgumentParser('Datasets converter from YOLOV5 to COCO')
    parser.add_argument(
        'input_json',
        type=str,
        default='datasets/YOLOV5',
        help='Dataset root path')
    parser.add_argument(
        '-u',
        '--image_root_url',
        type=str,
        help='root URL path where test_images will be hosted' +
        ', e.g.: http://example.com/images',
        default='/data/local-files/?d=')
    parser.add_argument(
        '--image_root_dir',
        type=str,
        help='root directory path where test_images will be stored',
        default=None,
    )
    args = parser.parse_args()
    if args.image_root_dir is not None:
        # get relative path to home dir
        image_root_dir = Path(args.image_root_dir).relative_to(Path.home())
        args.image_root_url = args.image_root_url + str(image_root_dir)
    converter = COCO2LS(args.input_json, args.image_root_url)
    converter()


if __name__ == '__main__':
    main()
