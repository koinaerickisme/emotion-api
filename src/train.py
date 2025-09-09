import argparse
import os
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import f1_score
import json
import numpy as np

from .models import build_model
from .config import LABELS, ARTIFACTS_DIR
from .losses import FocalLoss


def parse_args():
	p = argparse.ArgumentParser()
	p.add_argument("--data-train", type=str, default="data/processed/train")
	p.add_argument("--data-val", type=str, default="data/processed/val")
	p.add_argument("--backbone", type=str, default="efficientnet_b0")
	p.add_argument("--epochs", type=int, default=1)
	p.add_argument("--batch-size", type=int, default=32)
	p.add_argument("--lr", type=float, default=3e-4)
	p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	p.add_argument("--use-focal", action="store_true")
	p.add_argument("--alpha", type=float, default=0.25)
	p.add_argument("--gamma", type=float, default=2.0)
	p.add_argument("--class-weights", type=str, default="", help="Comma-separated weights per class")
	p.add_argument("--use-weighted-sampler", action="store_true")
	p.add_argument("--patience", type=int, default=5)
	p.add_argument("--export-torchscript", action="store_true")
	return p.parse_args()


def create_loaders(train_dir: str, val_dir: str, batch_size: int, use_weighted_sampler: bool):
	norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
	train_tf = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		norm,
	])
	val_tf = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		norm,
	])
	train_ds = ImageFolder(train_dir, transform=train_tf)
	val_ds = ImageFolder(val_dir, transform=val_tf)

	if use_weighted_sampler:
		# compute class frequencies
		class_counts = np.zeros(len(train_ds.classes), dtype=np.int64)
		for _, target in train_ds.samples:
			class_counts[target] += 1
		class_weights = 1.0 / np.maximum(class_counts, 1)
		sample_weights = [class_weights[target] for _, target in train_ds.samples]
		sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)
		train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
	else:
		train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
	return train_loader, val_loader


def main():
	args = parse_args()
	os.makedirs(ARTIFACTS_DIR, exist_ok=True)
	device = torch.device(args.device)

	model = build_model(num_classes=len(LABELS), backbone=args.backbone)
	model.to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

	if args.use_focal:
		criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma)
	else:
		if args.class_weights:
			weights = torch.tensor([float(x) for x in args.class_weights.split(",")], dtype=torch.float32)
			weights = weights.to(device)
			criterion = torch.nn.CrossEntropyLoss(weight=weights)
		else:
			criterion = torch.nn.CrossEntropyLoss()

	train_loader, val_loader = create_loaders(args.data_train, args.data_val, args.batch_size, args.use_weighted_sampler)

	best_f1 = 0.0
	pat = 0
	best_path = os.path.join(ARTIFACTS_DIR, "best.pt")

	for epoch in range(args.epochs):
		model.train()
		for images, targets in train_loader:
			images = images.to(device)
			targets = targets.to(device)
			optimizer.zero_grad()
			logits = model(images)
			loss = criterion(logits, targets)
			loss.backward()
			optimizer.step()

		# Validation macro-F1
		model.eval()
		y_true, y_pred = [], []
		with torch.no_grad():
			for images, targets in val_loader:
				images = images.to(device)
				logits = model(images)
				preds = logits.argmax(1).cpu().tolist()
				y_pred.extend(preds)
				y_true.extend(targets.tolist())
		f1 = f1_score(y_true, y_pred, average="macro")
		print(f"Epoch {epoch+1}/{args.epochs} macro-F1={f1:.4f}")
		if f1 > best_f1:
			best_f1 = f1
			pat = 0
			torch.save(model.state_dict(), best_path)
		else:
			pat += 1
			if pat >= args.patience:
				print("Early stopping.")
				break

	print("Best macro-F1=", best_f1)
	# Write labels.json for serving
	with open(os.path.join(ARTIFACTS_DIR, "labels.json"), "w") as f:
		json.dump(LABELS, f)
	if args.export_torchscript:
		model.load_state_dict(torch.load(best_path, map_location=device))
		model.eval()
		dummy = torch.randn(1, 3, 224, 224, device=device)
		scripted = torch.jit.trace(model, dummy)
		torchscript_path = os.path.join(ARTIFACTS_DIR, "model_ts.pt")
		scripted.save(torchscript_path)
		print("Saved TorchScript to", torchscript_path)


if __name__ == "__main__":
	main()
