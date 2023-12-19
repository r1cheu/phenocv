# @Author: SWHL
# @Contact: liekkaskono@163.com
# modified from https://github.com/RapidAI/LabelConvert/blob/main/label_convert/yolov5_to_coco.py # noqa
import json
import shutil
import time
import warnings
from pathlib import Path
from typing import Union

import cv2
from tqdm import tqdm


class YOLO2COCO:

    def __init__(self, data_dir):
        self.raw_data_dir = Path(data_dir)

        self.verify_exists(self.raw_data_dir / 'images')
        self.verify_exists(self.raw_data_dir / 'labels')

        save_dir_name = f'{Path(self.raw_data_dir).name}_COCO'
        self.output_dir = self.raw_data_dir.parent / save_dir_name
        self.mkdir(self.output_dir)

        self._init_json()

    def __call__(self, mode_list: list):
        if not mode_list:
            raise ValueError('mode_list is empty!!')

        for mode in mode_list:
            # Read the image txt.
            txt_path = self.raw_data_dir / f'{mode}.txt'
            self.verify_exists(txt_path)
            img_list = self.read_txt(txt_path)

            # Create the directory of saving the new image.
            save_img_dir = self.output_dir / f'{mode}'
            self.mkdir(save_img_dir)

            # Generate json file.
            anno_dir = self.output_dir / 'annotations'
            self.mkdir(anno_dir)

            save_json_path = anno_dir / f'{mode}.json'
            json_data = self.convert(img_list, save_img_dir, mode)

            self.write_json(save_json_path, json_data)
        print(f'Successfully convert, detail in {self.output_dir}')

    def _init_json(self):
        classes_path = self.raw_data_dir / 'classes.txt'
        self.verify_exists(classes_path)
        self.categories = self._get_category(classes_path)

        self.type = 'instances'
        self.annotation_id = 1

        self.cur_year = time.strftime('%Y', time.localtime(time.time()))
        self.info = {
            'year': int(self.cur_year),
            'version': '1.0',
            'description': 'For object detection',
            'date_created': self.cur_year,
        }

    def _get_category(self, classes_path):
        class_list = self.read_txt(classes_path)
        categories = []
        for i, category in enumerate(class_list, 1):
            categories.append({
                'supercategory': category,
                'id': i,
                'name': category,
            })
        return categories

    def convert(self, img_list, save_img_dir, mode):
        images, annotations = [], []
        for img_id, img_path in enumerate(tqdm(img_list, desc=mode), 1):
            image_dict = self.get_image_info(img_path, img_id, save_img_dir)
            images.append(image_dict)

            label_path = self.raw_data_dir / 'labels' / f'{Path(img_path).stem}.txt'  # noqa
            annotation = self.get_annotation(label_path, img_id,
                                             image_dict['height'],
                                             image_dict['width'])
            annotations.extend(annotation)

        json_data = {
            'info': self.info,
            'images': images,
            'type': self.type,
            'annotations': annotations,
            'categories': self.categories,
        }
        return json_data

    def get_image_info(self, img_path, img_id, save_img_dir):
        img_path = Path(img_path)
        if self.raw_data_dir.as_posix() not in img_path.as_posix():
            # relative path (relative to the raw_data_dir)
            # e.g. images/images(3).jpg
            img_path = self.raw_data_dir / img_path

        self.verify_exists(img_path)
        new_img_name = f'{img_path.stem}.jpg'
        save_img_path = save_img_dir / new_img_name
        img_src = cv2.imread(str(img_path))
        if img_path.suffix.lower() == '.jpg':
            shutil.copyfile(img_path, save_img_path)
        else:
            cv2.imwrite(str(save_img_path), img_src)

        height, width = img_src.shape[:2]
        image_info = {
            'date_captured': self.cur_year,
            'file_name': new_img_name,
            'id': img_id,
            'height': height,
            'width': width,
        }
        return image_info

    def get_annotation(self, label_path: Path, img_id, height, width):

        def get_box_info(vertex_info, height, width):
            cx, cy, w, h = (float(i) for i in vertex_info)

            cx = cx * width
            cy = cy * height
            box_w = w * width
            box_h = h * height

            # left top
            x0 = max(cx - box_w / 2, 0)
            y0 = max(cy - box_h / 2, 0)

            # right bottom
            x1 = min(x0 + box_w, width)
            y1 = min(y0 + box_h, height)

            segmentation = [[x0, y0, x1, y0, x1, y1, x0, y1]]
            bbox = [x0, y0, box_w, box_h]
            area = box_w * box_h
            return segmentation, bbox, area

        if not label_path.exists():
            annotation = [{
                'segmentation': [],
                'area': 0,
                'iscrowd': 0,
                'image_id': img_id,
                'bbox': [],
                'category_id': -1,
                'id': self.annotation_id,
            }]
            self.annotation_id += 1
            return annotation

        annotation = []
        label_list = self.read_txt(str(label_path))
        for i, one_line in enumerate(label_list):
            label_info = one_line.split(' ')
            if len(label_info) < 5:
                warnings.warn(
                    f'The {i + 1} line of the {label_path} has been corrupted.'
                )
                continue

            category_id, vertex_info = label_info[0], label_info[1:]
            segmentation, bbox, area = get_box_info(vertex_info, height, width)
            annotation.append({
                'segmentation': segmentation,
                'area': area,
                'iscrowd': 0,
                'image_id': img_id,
                'bbox': bbox,
                'category_id': int(category_id) + 1,
                'id': self.annotation_id,
            })
            self.annotation_id += 1
        return annotation

    @staticmethod
    def read_txt(txt_path):
        with open(str(txt_path), encoding='utf-8') as f:
            data = list(map(lambda x: x.rstrip('\n'), f))
        return data

    @staticmethod
    def mkdir(dir_path):
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def verify_exists(file_path: Union[Path, str]):
        if not Path(file_path).exists():
            raise FileNotFoundError(f'The {file_path} is not exists!!!')

    @staticmethod
    def write_json(json_path, content: dict):
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False)
