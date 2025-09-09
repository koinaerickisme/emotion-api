import argparse
import os
import json
from collections import Counter
from PIL import Image


def main():
	ap = argparse.ArgumentParser(description="Summarize dataset into data_summary.json")
	ap.add_argument("--manifest", required=True, help="CSV with path,label")
	ap.add_argument("--out", default="data/data_summary.json")
	args = ap.parse_args()

	import csv
	paths = []
	labels = []
	with open(args.manifest, "r") as f:
		reader = csv.DictReader(f)
		for row in reader:
			p = row["path"]
			l = row["label"]
			if os.path.exists(p):
				paths.append(p)
				labels.append(l)

	w, h = [], []
	for p in paths[:2000]:  # sample for speed
		try:
			im = Image.open(p)
			w.append(im.width)
			h.append(im.height)
		except Exception:
			pass

	summary = {
		"label_counts": dict(Counter(labels)),
		"num_images": len(paths),
		"image_size_stats": {
			"mean_w": sum(w) / len(w) if w else None,
			"mean_h": sum(h) / len(h) if h else None,
		},
	}
	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	with open(args.out, "w") as f:
		json.dump(summary, f, indent=2)
	print("Wrote", args.out)


if __name__ == "__main__":
	main()
