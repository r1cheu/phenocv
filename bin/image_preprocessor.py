#!/usr/bin/env python3
import os
from argparse import ArgumentParser

from tqdm import tqdm

from phenocv.preprocess import (H20ImageExtractor, LMJImageExtractor,
                                ResizeExtractor)
from phenocv.utils import DictAction, prepare_io_dir, scandir, write_file


def get_args():
    parser = ArgumentParser(description='Extract plots from original images.')
    parser.add_argument('input_dir', type=str, help='Path to input directory.')
    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        default=None,
        help='Path to output directory.')
    parser.add_argument('-t', '--type', default=None, help='Extractor type')
    parser.add_argument(
        '--save',
        action='store_true',
        help='whether to save the extracted plots')
    parser.add_argument(
        '--save_txt',
        action='store_true',
        help='whether to save the extracted plots')
    parser.add_argument(
        '--suffix',
        type=str,
        default='.jpg',
        help='Suffix of images to extract.')
    parser.add_argument(
        '--resume', action='store_true', help='whether to resume')
    parser.add_argument(
        '--ext_args',
        nargs='+',
        action=DictAction,
        help='kwargs for Extractor')
    return parser.parse_args()


def main():

    args = get_args()
    # prepare output directory
    input_dir, output_dir = prepare_io_dir(
        args.input_dir, args.output_dir, resume=args.resume)
    # get image paths
    img_paths = list(scandir(input_dir, suffix=args.suffix))
    init_args = args.ext_args

    if args.type.lower() == 'h20':
        extractor = H20ImageExtractor
    elif args.type.lower() == 'lmj':
        extractor = LMJImageExtractor
    elif args.type == 'resize':
        extractor = ResizeExtractor
    else:
        raise ValueError(
            'Args type should chose from LMJ, H20 or Resize, but got' +
            f'{args.type}')

    if args.save_txt:
        txt_path = output_dir / 'extract.txt'
        write_file(txt_path, 'source\ty1\ty2\tx1\tx2\n')

    pbar = tqdm(img_paths)

    for img_path in pbar:
        pbar.set_description(f'Processing {img_path}')
        in_image = input_dir / img_path
        out_image = output_dir / img_path

        if os.path.exists(out_image) and args.resume:
            print(f'{out_image} exists, skip process.')
            continue

        if init_args is None:
            img = extractor(in_image)
        else:
            img = extractor(in_image, **init_args)

        if args.save:
            img.save_image(out_image)

        if args.save_txt:
            xyxy = img.xyxy
            write_file(
                txt_path,
                f'{img_path}\t{xyxy.y1}\t{xyxy.y2}\t{xyxy.x1}\t{xyxy.x2}\n')

    print('Result saved to', output_dir)


if __name__ == '__main__':
    main()
