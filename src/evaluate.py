import argparse
import os
from typing import List

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from .models import build_model
from .config import load_labels


def parse_args():
	p = argparse.ArgumentParser()
	p.add_argument("--data", type=str, default="data/processed/val")
	p.add_argument("--weights", type=str, required=True)
	p.add_argument("--backbone", type=str, default="efficientnet_b0")
	p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	p.add_argument("--save-cm", action="store_true")
	return p.parse_args()


def save_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str):
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	fig, ax = plt.subplots(figsize=(6, 6))
	im = ax.imshow(cm, cmap="Blues")
	ax.set_xticks(range(len(labels)))
	ax.set_yticks(range(len(labels)))
	ax.set_xticklabels(labels, rotation=45, ha="right")
	ax.set_yticklabels(labels)
	for i in range(len(labels)):
		for j in range(len(labels)):
			ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
	ax.set_xlabel("Predicted")
	ax.set_ylabel("True")
	fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close(fig)


def main():
	args = parse_args()
	device = torch.device(args.device)
	labels = load_labels()

	norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
	tf = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		norm,
	])
	ds = ImageFolder(args.data, transform=tf)
	loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

	model = build_model(num_classes=len(labels), backbone=args.backbone)
	state = torch.load(args.weights, map_location=device)
	model.load_state_dict(state)
	model.eval().to(device)

	y_true, y_pred = [], []
	with torch.no_grad():
		for images, targets in loader:
			images = images.to(device)
			logits = model(images)
			preds = logits.argmax(1).cpu().tolist()
			y_pred.extend(preds)
			y_true.extend(targets.tolist())

	print(classification_report(y_true, y_pred, target_names=labels, digits=4))
	cm = confusion_matrix(y_true, y_pred)
	print("Confusion matrix:\n", cm)
	if args.save_cm:
		save_confusion_matrix(cm, labels, "reports/confusion_matrix.png")


if __name__ == "__main__":
	main()
