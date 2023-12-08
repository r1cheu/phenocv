_base_ = ['./cascade-rcnn_r50_fpn_2x_panicle-full.py']
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r101_fpn_mstrain_3x_coco/cascade_mask_rcnn_r101_fpn_mstrain_3x_coco_20210628_165236-51a2d363.pth'  # noqa
