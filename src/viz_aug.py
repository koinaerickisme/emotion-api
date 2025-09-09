import argparse
import random
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from .augment import train_transforms


def main():
	ap = argparse.ArgumentParser(description="Visualize augmentation montage")
	ap.add_argument("--image", required=True, help="Path to a sample face image")
	ap.add_argument("--n", type=int, default=12)
	ap.add_argument("--cols", type=int, default=6)
	ap.add_argument("--out", default="reports/aug_montage.png")
	args = ap.parse_args()

	img = Image.open(args.image).convert("RGB")
	tf = train_transforms()
	arr = np.array(img)
	panels = []
	for _ in range(args.n):
		aug = tf(image=arr)["image"].permute(1, 2, 0).numpy()
		aug = np.clip(aug * 0.229 + 0.485, 0, 1)  # roughly invert norm for display
		panels.append(aug)

	rows = (args.n + args.cols - 1) // args.cols
	fig, axes = plt.subplots(rows, args.cols, figsize=(args.cols * 2, rows * 2))
	axes = np.array(axes).reshape(rows, args.cols)
	for i in range(rows * args.cols):
		r = i // args.cols
		c = i % args.cols
		axes[r, c].axis("off")
		if i < len(panels):
			axes[r, c].imshow(panels[i])
	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	plt.tight_layout()
	plt.savefig(args.out)
	print("Saved", args.out)


if __name__ == "__main__":
	main()

