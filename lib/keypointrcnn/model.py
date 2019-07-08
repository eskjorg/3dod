"""keypointrcnn_resnet50_fpn"""
from torchvision.models.detection import keypointrcnn_resnet50_fpn as kptrcnn
from lib.constants import NBR_KEYPOINTS


def Model(configs):
    return kptrcnn(pretrained=False,
                   progress=True,
                   # num_classes=1,
                   num_classes=2,
                   # num_classes=3,
                   # num_classes=1 + max(configs.data.class_map.values()),
                   num_keypoints=NBR_KEYPOINTS,
                   # rpn_post_nms_top_n_train=1,
                   rpn_post_nms_top_n_test=1,
                   pretrained_backbone=True)
