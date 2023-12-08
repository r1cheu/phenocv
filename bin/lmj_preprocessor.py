#!/usr/bin/env python
from argparse import ArgumentParser

from tqdm import tqdm

from phenocv import preprocess, utils


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        'input_dir', type=str, default=None, help='input image dir')
    parser.add_argument(
        '--suffix', type=str, default='jpg', help='image suffix')
    parser.add_argument(
        '-o', '--output-dir', type=str, default=None, help='output image dir')
    parser.add_argument(
        '-s',
        '--save',
        action='store_true',
        help='whether to save the extracted plots')
    parser.add_argument(
        '--resize-long-side', type=int, default=1024, help='resize long side')
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    input_dir, output_dir = utils.prepare_io_dir(args.input_dir,
                                                 args.output_dir)

    img_paths = utils.scandir(args.input_dir, suffix=args.suffix)

    pbar = tqdm(img_paths)

    for img_path in pbar:
        pbar.set_description(f'Extracting {img_path}')
        img = preprocess.LMJImagePreprocessor(
            input_dir / img_path, resize_long_side=args.resize_long_side)
        if args.save:
            img.save_image(output_dir / img_path)
