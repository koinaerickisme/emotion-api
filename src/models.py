from typing import Literal
import timm


def build_model(num_classes: int = 4, backbone: str = "efficientnet_b0", pretrained: bool = False):
	"""Create a classification model using timm backbones.
	Note: default pretrained=False to avoid network downloads in container runtime.
	"""
	model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
	return model
