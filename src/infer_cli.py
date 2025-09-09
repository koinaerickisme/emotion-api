import argparse
import os
import csv
from PIL import Image
from typing import List

import torch

from .infer_service.predictor import Predictor


def collect_images(path: str) -> List[str]:
	if os.path.isdir(path):
		items = []
		for root, _, files in os.walk(path):
			for f in files:
				if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
					items.append(os.path.join(root, f))
			return items
	else:
		return [path]


def main():
	ap = argparse.ArgumentParser(description="Run offline inference and write CSV results")
	ap.add_argument("--weights", default="artifacts/best.pt")
	ap.add_argument("--input", required=True, help="Image file or directory")
	ap.add_argument("--out", default="inference_results.csv")
	ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	args = ap.parse_args()

	pred = Predictor(weights_path=args.weights, device=args.device)
	paths = collect_images(args.input)

	with open(args.out, "w", newline="") as f:
		writer = csv.writer(f)
		labels = pred.labels
		writer.writerow(["path", "emotion"] + labels)
		for p in paths:
			img = Image.open(p).convert("RGB")
			out = pred.predict(img)
			row = [p, out["emotion"]] + [out["scores"].get(lbl, 0.0) for lbl in labels]
			writer.writerow(row)
	print("Wrote", args.out)


if __name__ == "__main__":
	main()
