import albumentations as A
from albumentations.pytorch import ToTensorV2
from .config import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD


def train_transforms():
	return A.Compose([
		A.Resize(IMAGE_SIZE, IMAGE_SIZE),
		A.HorizontalFlip(p=0.5),
		A.Rotate(limit=15, p=0.4),
		A.RandomBrightnessContrast(p=0.4),
		A.GaussianBlur(blur_limit=3, p=0.2),
		A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
		A.CoarseDropout(max_holes=2, max_height=32, max_width=32, p=0.3),
		A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
		ToTensorV2(),
	])


def val_transforms():
	return A.Compose([
		A.Resize(IMAGE_SIZE, IMAGE_SIZE),
		A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
		ToTensorV2(),
	])
