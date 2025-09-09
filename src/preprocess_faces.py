import argparse
import os
import cv2
from PIL import Image
import numpy as np


def crop_faces_in_dir(src_root: str, dst_root: str, margin: float = 0.1):
	cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
	face_cascade = cv2.CascadeClassifier(cascade_path)
	for label in os.listdir(src_root):
		src_label_dir = os.path.join(src_root, label)
		if not os.path.isdir(src_label_dir):
			continue
		dst_label_dir = os.path.join(dst_root, label)
		os.makedirs(dst_label_dir, exist_ok=True)
		for fname in os.listdir(src_label_dir):
			src_path = os.path.join(src_label_dir, fname)
			if not os.path.isfile(src_path):
				continue
			try:
				img = Image.open(src_path).convert("RGB")
			except Exception:
				continue
			arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
			gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
			if len(faces) == 0:
				crop = img
			else:
				x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
				mh = int(margin * h)
				mw = int(margin * w)
				x0 = max(0, x - mw)
				y0 = max(0, y - mh)
				x1 = min(arr.shape[1], x + w + mw)
				y1 = min(arr.shape[0], y + h + mh)
				crop = Image.fromarray(cv2.cvtColor(arr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
			crop.save(os.path.join(dst_label_dir, fname))


def main():
	ap = argparse.ArgumentParser(description="Detect and crop faces into output dir")
	ap.add_argument("--src", required=True, help="Source root with class subfolders")
	ap.add_argument("--dst", required=True, help="Destination root for cropped faces")
	ap.add_argument("--margin", type=float, default=0.1)
	args = ap.parse_args()
	os.makedirs(args.dst, exist_ok=True)
	crop_faces_in_dir(args.src, args.dst, args.margin)
	print("Done.")


if __name__ == "__main__":
	main()
