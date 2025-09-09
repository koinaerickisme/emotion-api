from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import csv
import time
import hashlib
import base64
import json
import urllib.request
from collections import deque

from .schemas import PredictResponse, HealthResponse, ImagePayload, ExplainResponse, ImageBatchPayload, PredictBatchResponse
from .predictor import Predictor
from ..explain import gradcam_overlay_bytes
from ..config import load_labels

API_VERSION = "1.0.0"
app = FastAPI(title="Emotion API", version=API_VERSION)

# Lazy-init predictor to allow starting without weights (handled gracefully)
predictor = None

LOG_PATH = os.environ.get("REQUEST_LOG_PATH", "logs/requests.csv")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
METRICS_WINDOW = int(os.environ.get("METRICS_WINDOW", "200"))
_metrics = {"total": 0, "errors": 0, "latency_ms_sum": 0.0}
_latency_window = deque(maxlen=METRICS_WINDOW)

# CORS
origins = [o.strip() for o in CORS_ORIGINS.split(",")] if CORS_ORIGINS else ["*"]
app.add_middleware(
	CORSMiddleware,
	allow_origins=origins if origins != [""] else ["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


def _init_logging():
	os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
	if not os.path.exists(LOG_PATH):
		with open(LOG_PATH, "w", newline="") as f:
			w = csv.writer(f)
			w.writerow(["ts","request_id","model_version","top1","top1_prob","latency_ms","status","source_hash"]) 


def _alert_slack(message: str):
	if not SLACK_WEBHOOK_URL:
		return
	try:
		data = json.dumps({"text": message}).encode("utf-8")
		req = urllib.request.Request(SLACK_WEBHOOK_URL, data=data, headers={"Content-Type": "application/json"})
		urllib.request.urlopen(req, timeout=5)
	except Exception:
		pass


@app.on_event("startup")
async def startup_event():
	global predictor
	_init_logging()
	weights_path = os.environ.get("WEIGHTS_PATH", "artifacts/best.pt")
	device = os.environ.get("DEVICE", "cpu")
	if not os.path.exists(weights_path):
		url = os.environ.get("WEIGHTS_URL", "")
		if url:
			os.makedirs(os.path.dirname(weights_path), exist_ok=True)
			try:
				urllib.request.urlretrieve(url, weights_path)
			except Exception:
				pass
	try:
		predictor = Predictor(weights_path=weights_path, device=device)
	except Exception:
		predictor = None


@app.get("/health", response_model=HealthResponse)
async def health():
	status = "ok" if predictor is not None else "degraded"
	return {"status": status}


@app.get("/labels")
async def labels():
	return {"labels": getattr(predictor, "labels", load_labels())}


@app.get("/version")
async def version():
	return {
		"api_version": API_VERSION,
		"model_version": getattr(predictor, "model_version", "") if predictor else "",
	}


@app.get("/metrics")
async def metrics():
	avg_latency = (_metrics["latency_ms_sum"] / _metrics["total"]) if _metrics["total"] else 0.0
	window = list(_latency_window)
	p95 = 0.0
	if window:
		window_sorted = sorted(window)
		idx = int(0.95 * (len(window_sorted) - 1))
		p95 = window_sorted[idx]
	return {
		"total_requests": _metrics["total"],
		"error_rate": (_metrics["errors"] / _metrics["total"]) if _metrics["total"] else 0.0,
		"avg_latency_ms": round(avg_latency, 2),
		"p95_latency_ms": round(p95, 2),
		"window_size": len(window),
		"window_capacity": _latency_window.maxlen,
	}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, file: UploadFile = File(...)):
	start = time.time()
	request_id = hashlib.sha1(f"{start}-{id(request)}".encode()).hexdigest()[:12]
	source_ip = request.client.host if request.client else ""
	source_hash = hashlib.sha1(source_ip.encode()).hexdigest()[:12] if source_ip else ""
	status = "ok"
	try:
		if predictor is None:
			raise RuntimeError("Model not loaded")
		img_bytes = await file.read()
		img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
		out = predictor.predict(img)
		latency_ms = out.get("latency_ms", (time.time() - start) * 1000)
		_metrics["total"] += 1
		_metrics["latency_ms_sum"] += latency_ms
		_latency_window.append(latency_ms)
		os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
		with open(LOG_PATH, "a", newline="") as f:
			w = csv.writer(f)
			w.writerow([
				int(start), request_id, out.get("model_version", ""), out.get("emotion", ""),
				max(out.get("scores", {}).values()) if out.get("scores") else "",
				latency_ms, status, source_hash,
			])
		return out
	except Exception as e:
		_metrics["total"] += 1
		_metrics["errors"] += 1
		latency_ms = (time.time() - start) * 1000
		_latency_window.append(latency_ms)
		with open(LOG_PATH, "a", newline="") as f:
			w = csv.writer(f)
			w.writerow([int(start), request_id, getattr(predictor, "model_version", ""), "", "", latency_ms, "error", source_hash])
		_alert_slack(f":rotating_light: Emotion API error {type(e).__name__}: {e}")
		return JSONResponse(status_code=503, content={"detail": str(e)})


@app.post("/predict_json", response_model=PredictResponse)
async def predict_json(payload: ImagePayload):
	if predictor is None:
		return JSONResponse(status_code=503, content={"detail": "Model not loaded"})
	try:
		img_bytes = base64.b64decode(payload.image_b64)
		img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
		out = predictor.predict(img)
		return out
	except Exception as e:
		_alert_slack(f":rotating_light: Emotion API JSON error {type(e).__name__}: {e}")
		return JSONResponse(status_code=400, content={"detail": str(e)})


@app.post("/predict_batch", response_model=PredictBatchResponse)
async def predict_batch(payload: ImageBatchPayload):
	if predictor is None:
		return JSONResponse(status_code=503, content={"detail": "Model not loaded"})
	try:
		imgs = []
		for b64s in payload.images_b64:
			img_bytes = base64.b64decode(b64s)
			imgs.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
		outs = predictor.predict_batch(imgs)
		return {"items": outs}
	except Exception as e:
		return JSONResponse(status_code=400, content={"detail": str(e)})


@app.post("/explain", response_model=ExplainResponse)
async def explain(file: UploadFile = File(...)):
	if predictor is None:
		return JSONResponse(status_code=503, content={"detail": "Model not loaded"})
	img_bytes = await file.read()
	img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
	x = predictor._preprocess(img)
	png_bytes = gradcam_overlay_bytes(predictor.model, x)
	b64 = base64.b64encode(png_bytes).decode()
	out = predictor.predict(img)
	return {"emotion": out["emotion"], "cam_png_b64": b64, "latency_ms": out["latency_ms"]}


# Admin endpoints
@app.post("/admin/purge_logs")
async def admin_purge_logs():
	try:
		if os.path.exists(LOG_PATH):
			os.remove(LOG_PATH)
		_init_logging()
		return {"status": "ok"}
	except Exception as e:
		return JSONResponse(status_code=500, content={"detail": str(e)})


@app.post("/admin/reset_metrics")
async def admin_reset_metrics():
	global _metrics, _latency_window
	_metrics = {"total": 0, "errors": 0, "latency_ms_sum": 0.0}
	_latency_window = deque(maxlen=METRICS_WINDOW)
	return {"status": "ok"}


@app.get("/admin/config")
async def admin_config():
	return {
		"log_path": LOG_PATH,
		"cors_origins": CORS_ORIGINS,
		"metrics_window": METRICS_WINDOW,
	}


@app.delete("/admin/request/{request_id}")
async def admin_delete_request(request_id: str):
	"""Delete a specific request row from the CSV log by request_id."""
	if not os.path.exists(LOG_PATH):
		return {"status": "ok", "deleted": 0}
	kept = []
	deleted = 0
	with open(LOG_PATH, "r") as f:
		reader = csv.reader(f)
		rows = list(reader)
	if rows:
		header = rows[0]
		for row in rows[1:]:
			if len(row) >= 2 and row[1] == request_id:
				deleted += 1
				continue
			kept.append(row)
		with open(LOG_PATH, "w", newline="") as f:
			w = csv.writer(f)
			w.writerow(header)
			w.writerows(kept)
	return {"status": "ok", "deleted": deleted}
