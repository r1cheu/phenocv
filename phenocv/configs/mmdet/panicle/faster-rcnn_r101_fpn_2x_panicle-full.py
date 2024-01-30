_base_ = './faster-rcnn_r50_fpn_2x_panicle-full.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_mstrain_3x_coco/faster_rcnn_r101_fpn_mstrain_3x_coco_20210524_110822-4d4d2ca8.pth'  # noqa
