from typing import Dict, List, Tuple
import csv
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import argparse
import random

from .config import LABELS


def load_label_to_index() -> Dict[str, int]:
	return {label: i for i, label in enumerate(LABELS)}


class ManifestImageDataset(Dataset):
	"""Dataset backed by a CSV manifest with columns: path,label"""
	def __init__(self, csv_path: str, transform=None):
		self.items: List[Tuple[str, str]] = []
		self.transform = transform
		with open(csv_path, "r") as f:
			reader = csv.DictReader(f)
			for row in reader:
				self.items.append((row["path"], row["label"]))
		self.label_to_index = load_label_to_index()

	def __len__(self):
		return len(self.items)

	def __getitem__(self, idx: int):
		path, label = self.items[idx]
		img = Image.open(path).convert("RGB")
		if self.transform is not None:
			img = self.transform(image=np.array(img))["image"]
		return img, self.label_to_index[label]


def _write_manifest(items: List[Tuple[str, str]], csv_path: str):
	os.makedirs(os.path.dirname(csv_path), exist_ok=True)
	with open(csv_path, "w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=["path", "label"])
		w.writeheader()
		for p, l in items:
			w.writerow({"path": p, "label": l})


def build_splits_from_folder(root: str, out_dir: str, seed: int = 42, ratios=(0.8, 0.1, 0.1)):
	random.seed(seed)
	train, val, test = [], [], []
	labels = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
	for label in labels:
		paths = []
		ldir = os.path.join(root, label)
		for fname in os.listdir(ldir):
			fp = os.path.join(ldir, fname)
			if os.path.isfile(fp):
				paths.append(fp)
		random.shuffle(paths)
		n = len(paths)
		n_train = int(ratios[0] * n)
		n_val = int(ratios[1] * n)
		train.extend([(p, label) for p in paths[:n_train]])
		val.extend([(p, label) for p in paths[n_train:n_train+n_val]])
		test.extend([(p, label) for p in paths[n_train+n_val:]])
	_write_manifest(train, os.path.join(out_dir, "train.csv"))
	_write_manifest(val, os.path.join(out_dir, "val.csv"))
	_write_manifest(test, os.path.join(out_dir, "test.csv"))


def _parse():
	ap = argparse.ArgumentParser(description="Build stratified splits from class folders")
	ap.add_argument("--root", required=True, help="Root folder with subfolders per class")
	ap.add_argument("--out", default="data/processed", help="Output dir for CSV manifests")
	ap.add_argument("--seed", type=int, default=42)
	ap.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1))
	return ap.parse_args()


if __name__ == "__main__":
	args = _parse()
	build_splits_from_folder(args.root, args.out, args.seed, tuple(args.ratios))
