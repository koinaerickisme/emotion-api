from typing import List
import json
import os

# Centralized labels used across training and serving
LABELS: List[str] = ["angry", "happy", "sad", "neutral"]

# Allow override from artifacts/labels.json if present
def load_labels() -> List[str]:
	path = os.path.join("artifacts", "labels.json")
	if os.path.exists(path):
		try:
			with open(path, "r") as f:
				return json.load(f)
		except Exception:
			return LABELS
	return LABELS

# Image settings
IMAGE_SIZE: int = 224
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

# Model defaults
DEFAULT_BACKBONE: str = "efficientnet_b0"
PRETRAINED: bool = True

# Paths (may be overridden by CLI)
ARTIFACTS_DIR = "artifacts"
DATA_DIR = "data"
