from argparse import ArgumentParser

import cv2
from tqdm import tqdm

from phenocv import preprocess, utils


def get_args():
    parser = ArgumentParser(description='Extract plots from original images.')
    parser.add_argument('input_dir', type=str, help='Path to input directory.')
    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        default=None,
        help='Path to output '
        'directory.')
    parser.add_argument(
        '--suffix',
        type=str,
        default='.jpg',
        help='Suffix '
        'of images to extract.')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='whether to recursively search for images in input_dir')
    return parser.parse_args()


def main():

    args = get_args()
    # prepare output directory
    input_dir, output_dir = utils.prepare_io_dir(args.input_dir,
                                                 args.output_dir)
    # get image paths
    img_paths = utils.scandir(
        input_dir, suffix=args.suffix, recursive=args.recursive)
    pbar = tqdm(img_paths)
    for img_path in pbar:
        pbar.set_description(f'Processing {img_path}')
        img = preprocess.cut_plot(input_dir / img_path, 3800, 2000, 100)
        cv2.imwrite(str(output_dir / img_path), img)

    print('Processed images saved to', output_dir)


if __name__ == '__main__':
    main()
