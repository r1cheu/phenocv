from argparse import ArgumentParser
from pathlib import Path

import cv2

from phenocv import preprocess, utils


def get_args():
    parser = ArgumentParser(description='Extract plots from original images.')
    parser.add_argument('input_dir', type=str, help='Path to input directory.')
    parser.add_argument('output_dir', type=str, default=None,
                        help='Path to output directory.')
    parser.add_argument('--suffix', type=str, default='.jpg', help='Suffix '
                                                                   'of images to extract.')
    parser.add_argument('--recursive',
                        action='store_true',
                        help='whether to recursively search for images in input_dir')
    return parser.parse_args()


def main():

    args = get_args()
    # prepare output directory
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir or f'{input_dir}_plots').resolve()

    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f'{output_dir} already exists, remove it first.')

    # get image paths
    img_paths = utils.scandir(args.input_dir,
                              suffix=args.suffix,
                              recursive=args.recursive,
                              case_sensitive=True)

    for img_path in img_paths:
        img = preprocess.cut_plot(input_dir/img_path, 3800, 2000, 100)
        cv2.imwrite(str(output_dir/img_path), img)


if __name__ == "__main__":
    main()