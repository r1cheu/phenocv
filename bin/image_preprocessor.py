#!/usr/bin/env python3
from argparse import ArgumentParser

from tqdm import tqdm

from phenocv import utils
from phenocv.preprocess import H20ImageExtractor, LMJImageExtractor


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
        help='Suffix '
        'of images to extract.')
    return parser.parse_args()


def main():

    args = get_args()
    # prepare output directory
    input_dir, output_dir = utils.prepare_io_dir(args.input_dir,
                                                 args.output_dir)
    # get image paths
    img_paths = utils.scandir(input_dir, suffix=args.suffix)

    if args.type.lower() == 'h20':
        extractor = H20ImageExtractor
    elif args.type.lower() == 'lmj':
        extractor = LMJImageExtractor
    else:
        raise ValueError(
            f'Args type should be LMJ or H20, but got {args.type}')

    if args.save_txt:
        txt_path = output_dir / 'extract.txt'
        utils.write_file(txt_path, 'source\ty1\ty2\tx1\tx2\n')

    pbar = tqdm(img_paths)

    for img_path in pbar:
        pbar.set_description(f'Processing {img_path}')
        img = extractor(input_dir / img_path)

        if args.save:
            img.save_image(output_dir / img_path)

        if args.save_txt:
            xyxy = img.xyxy
            utils.write_file(
                txt_path,
                f'{img_path}\t{xyxy.y1}\t{xyxy.y2}\t{xyxy.x1}\t{xyxy.x2}\n')

    print('Result saved to', output_dir)


if __name__ == '__main__':
    main()
