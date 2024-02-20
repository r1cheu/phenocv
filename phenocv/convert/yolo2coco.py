# @Author: SWHL
# @Contact: liekkaskono@163.com
# modified from https://github.com/RapidAI/LabelConvert/blob/main/label_convert/yolov5_to_coco.py # noqa
import shutil
import time
import warnings
from pathlib import Path

import cv2
from tqdm import tqdm

from phenocv.utils import check_img_input

from .base import YOLOto


class YOLOtoCOCO(YOLOto):

    def __init__(self, data_dir: Path | str):
        super().__init__(data_dir)
        self._init_coco_json()

    def _init_coco_json(self):

        classes_path = self.raw_data_dir / 'classes.txt'

        self._type = 'instance'
        self._annotation_id = 1
        self.categories = self._get_category(classes_path)

        self._cur_date = time.strftime('%Y/%m/%d', time.localtime(time.time()))
        self.info = {
            'year': int(self._cur_date.split('/')[0]),
            'version': '1.0',
            'description': 'For object detection',
            'date_created': self._cur_date, }

    def _get_category(self, classes_path):
        class_list = self.read_txt(classes_path)
        categories = []
        for i, category in enumerate(class_list, 1):
            categories.append({
                'supercategory': category,
                'id': i,
                'name': category, })
        return categories

    def __call__(self):

        for mode in self.mode_list:
            # Read the image txt.
            txt_path = self.raw_data_dir / f'{mode}.txt'
            img_list = self.read_txt(txt_path)

            # Create the directory of saving the new image.
            save_img_dir = self.output_dir / f'{mode}'
            save_img_dir.mkdir()
            # Generate json file.
            anno_dir = self.output_dir / 'annotations'
            anno_dir.mkdir()

            save_json_path = anno_dir / f'{mode}.json'
            json_data = self.convert(img_list, save_img_dir, mode)

            self.write_json(save_json_path, json_data)
        print(f'Successfully convert, detail in {self.output_dir}')

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
            'test_images': images,
            'type': self._type,
            'annotations': annotations,
            'categories': self.categories, }
        return json_data

    def get_image_info(self, img_path, img_id, save_img_dir):
        img_path = Path(img_path)
        if self.raw_data_dir.as_posix() not in img_path.as_posix():
            # relative path (relative to the raw_data_dir)
            # e.g. test_images/test_images(3).jpg
            img_path = self.raw_data_dir / img_path

        check_img_input(img_path)

        new_img_name = f'{img_path.stem}.jpg'
        save_img_path = save_img_dir / new_img_name
        img_src = cv2.imread(str(img_path))

        if img_path.suffix.lower() == '.jpg':
            shutil.copyfile(img_path, save_img_path)
        else:
            cv2.imwrite(str(save_img_path), img_src)

        height, width = img_src.shape[:2]
        image_info = {
            'file_name': new_img_name,
            'id': img_id,
            'height': height,
            'width': width, }
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
                'bbox': [0, 0, 0, 0],
                'category_id': -1,
                'id': self._annotation_id, }]
            self._annotation_id += 1
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
                'id': self._annotation_id, })
            self._annotation_id += 1
        return annotation
