import argparse
import os
import shutil
import csv
from typing import Dict, Tuple, List
from PIL import Image
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def is_image(path: str) -> bool:
	ext = os.path.splitext(path)[1].lower()
	return ext in IMAGE_EXTS


def compute_ahash(img: Image.Image, size: int = 8) -> str:
	gray = img.convert("L").resize((size, size))
	arr = np.asarray(gray, dtype=np.float32)
	mean = arr.mean()
	bits = (arr >= mean).flatten()
	# pack to hex string
	val = 0
	for b in bits:
		val = (val << 1) | int(b)
	return f"{val:0{size*size//4}x}"


def walk_images(root: str) -> List[str]:
	paths = []
	for dirpath, _, filenames in os.walk(root):
		for fname in filenames:
			fp = os.path.join(dirpath, fname)
			if is_image(fp):
				paths.append(fp)
	return paths


def write_manifest(items: List[Tuple[str, str]], csv_path: str):
	os.makedirs(os.path.dirname(csv_path), exist_ok=True)
	with open(csv_path, "w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=["path", "label"])
		w.writeheader()
		for p, l in items:
			w.writerow({"path": p, "label": l})


def main():
	ap = argparse.ArgumentParser(description="Clean corrupt images and deduplicate via average hash")
	ap.add_argument("--root", required=True, help="Root folder containing class subfolders")
	ap.add_argument("--out-manifest", default="data/processed/cleaned.csv")
	ap.add_argument("--move-duplicates-to", default="", help="Directory to move duplicates to (if not deleting)")
	ap.add_argument("--delete-duplicates", action="store_true")
	ap.add_argument("--quarantine-corrupt", default="data/interim/corrupt")
	args = ap.parse_args()

	os.makedirs(args.quarantine_corrupt, exist_ok=True)
	if args.move_duplicates_to:
		os.makedirs(args.move_duplicates_to, exist_ok=True)

	hash_to_path: Dict[str, str] = {}
	kept: List[Tuple[str, str]] = []

	for fp in walk_images(args.root):
		try:
			img = Image.open(fp)
			img.verify()  # quick check
			img = Image.open(fp).convert("RGB")  # reopen to use
		except Exception:
			# quarantine corrupt
			try:
				shutil.move(fp, os.path.join(args.quarantine_corrupt, os.path.basename(fp)))
			except Exception:
				pass
			continue
		h = compute_ahash(img)
		if h in hash_to_path:
			# duplicate found
			if args.delete_duplicates:
				try:
					os.remove(fp)
				except Exception:
					pass
			elif args.move_duplicates_to:
				try:
					shutil.move(fp, os.path.join(args.move_duplicates_to, os.path.basename(fp)))
				except Exception:
					pass
			continue
		hash_to_path[h] = fp
		label = os.path.basename(os.path.dirname(fp))
		kept.append((fp, label))

	write_manifest(kept, args.out_manifest)
	print(f"Kept {len(kept)} images. Manifest -> {args.out_manifest}")


if __name__ == "__main__":
	main()
