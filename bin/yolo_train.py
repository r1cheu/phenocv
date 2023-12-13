import os

from ultralytics import YOLO

if __name__ == '__main__':
    work_dir = os.environ.get('phenocv_dir')
    model = YOLO('yolov8n.pt')

    results = model.train(
        data=f'{work_dir}/configs/yolo/stubble.yaml', imgsz=1280, flipud=0.5)
