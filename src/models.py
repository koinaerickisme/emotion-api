from typing import Literal
import timm


def build_model(num_classes: int = 4, backbone: str = "efficientnet_b0", pretrained: bool = True):
	"""Create a classification model using timm backbones."""
	model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
	return model
