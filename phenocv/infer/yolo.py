from typing import List

import numpy as np
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
from ultralytics.engine.results import Results


class YOLOinfer:

    def __init__(self, model, obj_cls=0, device='cuda:0') -> None:

        self.model = YOLO(model).to(device)
        self.obj_cls = obj_cls

    def __call__(self, source, iou=0.1):

        results = self.model.predict(source, iou=iou)
        bboxes = self._get_bbox(results)
        matrices = []
        for bbox in bboxes:
            matrices.append(self.assign_boxes(bbox))

        return matrices

    def _get_bbox(self, results: List[Results]):

        bboxes = []

        for result in results:

            index = result.boxes.cls == self.obj_cls
            boxes = result.boxes.xyxyn[index]
            bboxes.append(boxes.cpu().numpy())

        return bboxes

    # TODO: plot the boxes and row and col on the image
    def plot(self):
        pass

    def assign_boxes(self, bboxes: np.ndarray):

        x = bboxes[:, 0]
        y = bboxes[:, 1]

        rows = self._cluster(y)
        cols = self._cluster(x)

        matrix = np.zeros((rows.max() + 1, cols.max() + 1, 4))
        matrix[rows, cols] = bboxes

        return matrix

    def _cluster(self, array):

        sort_index = np.argsort(array)
        inverse_index = np.argsort(sort_index)

        sorted_array = array[sort_index].reshape(-1, 1)
        num = DBSCAN(eps=0.05, min_samples=1).fit_predict(sorted_array)

        return num[inverse_index]
