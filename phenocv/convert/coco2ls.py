import json
import os
import uuid
from pathlib import Path

from phenocv.utils import check_img_input

try:
    from label_studio_converter.imports.label_config import \
        generate_label_config
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        'label_studio_converter is not installed, ' +
        'run `pip install label_studio_converter` to install')


class COCO2LS:

    def __init__(self,
                 input_json,
                 image_root_url,
                 use_super_categories=False,
                 out_type='predictions',
                 from_name='label',
                 to_name='image'):

        check_img_input(input_json)
        with open(input_json, encoding='utf-8') as f:
            self._coco = json.load(f)

        self._use_super_categories = use_super_categories
        self._out_type = out_type
        self._from_name = from_name
        self._to_name = to_name
        self._image_root_url = image_root_url
        self._output_json = f'{Path(input_json).stem}_lb.json'

    def __call__(self, ):
        # generate and save labeling config
        categories = self._get_categories(self._use_super_categories)
        images = self._get_images(categories)
        tasks = self._init_tasks(self._out_type)
        tasks, tags = self.convert(categories, images, tasks)
        label_config_file = self._output_json.replace('.json',
                                                      '') + '.label_config.xml'
        generate_label_config(categories, tags, self._to_name, self._from_name,
                              label_config_file)

        if len(tasks) > 0:
            tasks = [tasks[key] for key in sorted(tasks.keys())]
            print('Saving Label Studio JSON to', self._output_json)
            with open(self._output_json, 'w') as out:
                json.dump(tasks, out)

            print(
                '\n'
                f'Following the instructions to load data into Label Studio\n'
                f'  1. Create a new project in Label Studio\n'
                f'  2. Use Labeling Config from "{label_config_file}"\n'
                f'  3. Setup serving for test_images [e.g. you can use Local Storage (or others):\n'  # noqa
                f'     https://labelstud.io/guide/storage.html#Local-storage]\n'  # noqa
                f'  4. Import "{self._output_json}" to the project\n')
        else:
            print('No labels converted')

    def convert(self, categories, images, tasks):

        bbox = False
        bbox_once = False
        rectangles_from_name = 'RectangleLabels'
        tags = {}

        for _, annotation in enumerate(self._coco['annotations']):
            bbox |= 'bbox' in annotation  # if bbox
            if bbox and not bbox_once:
                tags.update({rectangles_from_name: 'RectangleLabels'})
                bbox_once = True

            # read image sizes
            image_id = annotation['image_id']
            image = images[image_id]
            image_width, image_height = (
                image['width'],
                image['height'],
            )
            task = tasks[image_id]

            if 'bbox' in annotation:
                item = self._create_bbox(
                    annotation,
                    categories,
                    rectangles_from_name,
                    image_height,
                    image_width,
                    self._to_name,
                )
                task[self._out_type][0]['result'].append(item)

            tasks[image_id] = task

        return tasks, tags

    def _get_images(self, categories):
        images = {item['id']: item for item in self._coco['test_images']}

        print(
            f'Found {len(categories)} categories, {len(images)} test_images' +
            f'and {len(self._coco["annotations"])} annotations')
        return images

    def _get_categories(self, use_super_categories=False):
        new_categories = {}

        categories = {
            int(category['id']): category
            for category in self._coco['categories']}

        ids = sorted(categories.keys())

        for i in ids:
            name = categories[i]['name']
            if use_super_categories and 'supercategory' in categories[i]:
                name = categories[i]['supercategory'] + ':' + name
            new_categories[i] = name

        categories = new_categories

        return categories

    def _new_task(self, out_type, image_file_name):
        return {
            'data': {
                'image': os.path.join(self._image_root_url, image_file_name)},
            # 'annotations' or 'predictions'
            out_type: [{
                'result': [],
                'ground_truth': False, }], }

    def _create_bbox(self, annotation, categories, from_name, image_height,
                     image_width, to_name):
        """create bbox labeling with Label Studio format. copy from:
        https://github.com/heartexlabs/label-studio-
        converter/blob/master/label_studio_converter/imports/coco.py.

        Args:
            annotation (dict): annotation dict with COCO format.
            categories (List): a list of categories.
            from_name (str): Name of the tag used to label the region in
                Label Studio.
            image_height (int): height of image.
            image_width (int): width of image.
            to_name (str): Name of the object tag that
                provided the region to be labeled.

        Returns:
            dict: an labeling dict with Label Studio format.
        """
        category_id = int(annotation['category_id'])
        if category_id == -1:
            label = 'None'
        else:
            label = categories[category_id]

        x, y, width, height = annotation['bbox']
        x, y, width, height = float(x), float(y), float(width), float(height)
        item = {
            'id': uuid.uuid4().hex[0:10],
            'type': 'rectanglelabels',
            'value': {
                'x': x / image_width * 100.0,
                'y': y / image_height * 100.0,
                'width': width / image_width * 100.0,
                'height': height / image_height * 100.0,
                'rotation': 0,
                'rectanglelabels': [label], },
            'to_name': to_name,
            'from_name': from_name,
            'image_rotation': 0,
            'original_width': image_width,
            'original_height': image_height, }
        return item

    def _init_tasks(self, out_type):
        tasks = {}
        for image in self._coco['test_images']:
            image_id, image_file_name = image['id'], image['file_name']
            tasks[image_id] = self._new_task(out_type, image_file_name)

        return tasks
