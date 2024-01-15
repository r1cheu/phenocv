_base_ = [
    '../_base_/models/cascade-rcnn_r50_fpn.py',
    '../_base_/datasets/panicle_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/custom_runtime.py'
]

model = dict(
    roi_head=dict(bbox_head=[
        dict(type='Shared2FCBBoxHead', num_classes=1),
        dict(type='Shared2FCBBoxHead', num_classes=1),
        dict(type='Shared2FCBBoxHead', num_classes=1)
    ]),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300)))
env_cfg = dict(cudnn_benchmark=True, )
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_mstrain_3x_coco/cascade_mask_rcnn_r50_fpn_mstrain_3x_coco_20210628_164719-5bdc3824.pth'  # noqa
