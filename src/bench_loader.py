import argparse
import time
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch


def main():
	ap = argparse.ArgumentParser(description="Benchmark DataLoader throughput")
	ap.add_argument("--data", default="data/processed/train")
	ap.add_argument("--batch-size", type=int, default=128)
	ap.add_argument("--num-workers", type=int, default=4)
	ap.add_argument("--iters", type=int, default=200)
	args = ap.parse_args()

	norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
	tf = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		norm,
	])
	ds = ImageFolder(args.data, transform=tf)
	loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

	it = iter(loader)
	count = 0
	start = time.time()
	for i in range(args.iters):
		try:
			images, targets = next(it)
		except StopIteration:
			it = iter(loader)
			images, targets = next(it)
		count += images.size(0)
	elapsed = time.time() - start
	ips = count / max(elapsed, 1e-6)
	print(f"Images/sec: {ips:.2f} (batch={args.batch_size}, workers={args.num_workers})")


if __name__ == "__main__":
	main()

