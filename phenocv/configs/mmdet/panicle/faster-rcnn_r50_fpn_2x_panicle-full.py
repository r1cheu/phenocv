_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/panicle_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/custom_runtime.py']

# `lr` and `weight_decay` have been searched to be optimal.
# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=1)),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='soft_nms', iou_threshold=0.5),
            max_per_img=300)))

env_cfg = dict(cudnn_benchmark=True, )

# 使用预训练的 Mask R-CNN 模型权重来做初始化，可以提高模型性能
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'  # noqa
