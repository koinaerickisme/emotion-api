import os
import time
from typing import Dict, List
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

from ..models import build_model
from ..augment import val_transforms
from ..config import load_labels


class Predictor:
	def __init__(self, weights_path: str, device: str = "cpu"):
		self.device = torch.device(device)
		self.labels = load_labels()
		self.tf = val_transforms()
		self.model_version = "v1.0.0"
		# TorchScript support
		if weights_path.endswith("_ts.pt") or weights_path.endswith("model_ts.pt"):
			self.model = torch.jit.load(weights_path, map_location=self.device)
			self.model.eval().to(self.device)
		else:
			self.model = build_model(num_classes=len(self.labels))
			state = torch.load(weights_path, map_location=self.device)
			self.model.load_state_dict(state)
			self.model.eval().to(self.device)
		self.enable_face_detect = os.environ.get("FACE_DETECT", "0") == "1"
		self.conf_threshold = float(os.environ.get("EMOTION_CONF_THRESH", "0.0"))
		if self.enable_face_detect:
			cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
			self.face_cascade = cv2.CascadeClassifier(cascade_path)
		else:
			self.face_cascade = None

	def _detect_and_crop(self, arr: np.ndarray) -> np.ndarray:
		if self.face_cascade is None:
			return arr
		gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
		faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
		if len(faces) == 0:
			return arr
		x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
		mh = int(0.1 * h)
		mw = int(0.1 * w)
		x0 = max(0, x - mw)
		y0 = max(0, y - mh)
		x1 = min(arr.shape[1], x + w + mw)
		y1 = min(arr.shape[0], y + h + mh)
		crop = arr[y0:y1, x0:x1]
		return crop if crop.size else arr

	def _preprocess(self, img: Image.Image):
		arr = np.array(img)
		arr = self._detect_and_crop(arr) if self.enable_face_detect else arr
		t = self.tf(image=arr)["image"].unsqueeze(0)  # 1x3x224x224
		return t.to(self.device)

	def _preprocess_many(self, imgs: List[Image.Image]) -> torch.Tensor:
		tensors = []
		for img in imgs:
			arr = np.array(img)
			arr = self._detect_and_crop(arr) if self.enable_face_detect else arr
			t = self.tf(image=arr)["image"]
			tensors.append(t)
		return torch.stack(tensors, dim=0).to(self.device)

	def predict(self, img: Image.Image) -> Dict[str, object]:
		t0 = time.time()
		x = self._preprocess(img)
		with torch.no_grad():
			logits = self.model(x)
			probs = F.softmax(logits, dim=1).cpu().numpy()[0]
		idx = int(probs.argmax())
		maxp = float(probs[idx])
		latency_ms = (time.time() - t0) * 1000
		emotion = self.labels[idx] if maxp >= self.conf_threshold else "unknown"
		return {
			"emotion": emotion,
			"scores": {k: float(v) for k, v in zip(self.labels, probs)},
			"model_version": self.model_version,
			"latency_ms": round(latency_ms, 2),
		}

	def predict_batch(self, imgs: List[Image.Image]) -> List[Dict[str, object]]:
		t0 = time.time()
		x = self._preprocess_many(imgs)
		with torch.no_grad():
			logits = self.model(x)
			probs = F.softmax(logits, dim=1).cpu().numpy()
		outs: List[Dict[str, object]] = []
		for i in range(probs.shape[0]):
			p = probs[i]
			idx = int(p.argmax())
			maxp = float(p[idx])
			emotion = self.labels[idx] if maxp >= self.conf_threshold else "unknown"
			outs.append({
				"emotion": emotion,
				"scores": {k: float(v) for k, v in zip(self.labels, p)},
				"model_version": self.model_version,
				"latency_ms": round((time.time() - t0) * 1000, 2),
			})
		return outs
