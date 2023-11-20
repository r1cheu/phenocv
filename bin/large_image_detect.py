from argparse import ArgumentParser
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from phenocv.preprocess import binarize_cive, moving_average, min_sum, cut_plot


def get_args():
    args = ArgumentParser()
    args.add_argument('input_dir',
                      type=str,
                      default=None,
                      help='input image dir')
    args.add_argument('weights',
                      type=str,
                      default=None,
                      help='model weights')
    args.add_argument('--save_image',
                      action='store_true',
                      help='save image')
    args.add_argument('--glob_pattern',
                      type=str,
                      default='*.jpg',
                      help='glob pattern')
    args.add_argument('--suffix',
                      type=str,
                      default='jpg',
                      help='image suffix')
    args.add_argument('--out_dir',
                      type=str,
                      default=None,
                      help='output image dir')
    args.add_argument('--device',
                      type=str,
                      default='cude:0',
                      help='device to use')
    args.add_argument('--score-thr',
                      type=float,
                      default=0.3,
                      help='score threshold')
    args.add_argument('--overlap-iou-thr',
                      type=float,
                      default=0.25,
                      help='iou threshold')
    args.add_argument('--patch-size',
                      type=int,
                      default=1000,
                      help='patch size')
    args.add_argument('--patch-overlap',
                      type=float,
                      default=0.25,
                      help='patch overlap')
    args.add_argument('--batch-size',
                      type=int,
                      default=16,
                      help='batch size')
    return args.parse_args()


def main():
    args = get_args()

    # prepare directories and image paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        print(f'{args.output_dir} does not exist. Creating it...')
        output_dir.mkdir(parents=True)

    image_path = list(input_dir.rglob(args.glob_pattern))

    # load model

    model = AutoDetectionModel.from_pretrained(
        model_type=args.model_type,
        model_path=args.weights,
    )