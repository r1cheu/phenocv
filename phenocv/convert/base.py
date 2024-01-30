import json
from pathlib import Path

from phenocv.utils import check_img_input


class YOLOto:

    def __init__(self, data_dir: Path | str):

        self.raw_data_dir = Path(data_dir)
        save_dir = Path(f'{self.raw_data_dir.name}_fromYOLO')
        self.output_dir = self.raw_data_dir.parent / save_dir

        self._check_input()
        self.output_dir.mkdir()

    def _check_input(self):
        """Check if the required path files and directories exist. This method
        checks if the 'images', 'labels', and 'classes.txt' files exist in the
        raw_data_dir. It also checks if the 'train.txt', 'val.txt', and
        'test.txt' files exist in the raw_data_dir. The existing modes are
        stored in the mode_list attribute.

        Returns:
            None
        """
        mode_list = []
        for file in ['images', 'labels', 'classes.txt']:
            check_img_input(self.raw_data_dir / file)

        for mode in ['train', 'val', 'test']:
            if (self.raw_data_dir / f'{mode}.txt').exists():
                mode_list.append(mode)
        if len(mode_list) == 0:
            raise FileNotFoundError('No train.txt, val.txt, or test.txt'
                                    'found in the path directory.')
        self.mode_list = mode_list

    @staticmethod
    def read_txt(txt_path):
        with open(str(txt_path), encoding='utf-8') as f:
            data = list(map(lambda x: x.rstrip('\n'), f))
        return data

    @staticmethod
    def write_json(json_path, content: dict):
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=4)


# TODO add COCOto as a base class for COCO to other annotation format
class COCOto:

    def __init__(self):
        pass
