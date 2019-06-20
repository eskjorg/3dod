"""keypointrcnn_resnet50_fpn"""
from torchvision.models.detection import keypointrcnn_resnet50_fpn as kptrcnn


def Model(configs):
    return kptrcnn(pretrained=False,
                   progress=True,
                   num_classes=1 + max(configs.data.class_map.values()),
                   num_keypoints=8,
                   pretrained_backbone=True)
