import os

from ultralytics import YOLO

if __name__ == '__main__':
    work_dir = os.environ.get('phenocv_dir')
    model = YOLO(f'{work_dir}/weights/yolov8m.pt')

    results = model.train(
        data=f'{work_dir}/configs/yolo/stubble.yaml', imgsz=640, flipud=0.5)
