import glob
import os
import os.path as osp
import shutil
from argparse import ArgumentParser
from multiprocessing import Pool


def get_args():
    parser = ArgumentParser(description='This script is used to rearrange'
                                        ' images by ID')
    parser.add_argument('input_dir',
                        type=str,
                        help='Directory containing images')
    parser.add_argument('output_dir',
                        type=str,
                        help='Output directory')
    return parser.parse_args()


def worker(sample, input_dir, args):
    input_path = osp.join(args.input_dir, input_dir, f'*{sample}.JPG')
    output_path = osp.join(args.output_dir, sample)
    os.makedirs(output_path, exist_ok=True)
    for file in glob.glob(input_path):
        shutil.copy(file, output_path)


def main():
    args = get_args()
    input_dirs = sorted(os.listdir(args.input_dir))
    sample_list = [osp.basename(image).split('-')[-1].split('.')[0]
                   for image in os.listdir(osp.join(args.input_dir, input_dirs[0]))]

    with Pool() as p:
        for input_dir in input_dirs:
            p.starmap(worker, [(sample, input_dir, args) for sample in sample_list])


if __name__ == '__main__':
    main()
