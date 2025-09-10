import io
import os
import base64
import requests
import streamlit as st
from PIL import Image
import numpy as np

import pandas as pd

st.set_page_config(page_title="Emotion Demo", layout="centered")
st.title("Facial Emotion Recognition")

with st.expander("Privacy & Consent", expanded=True):
	st.write("This demo performs on-device or API inference on your uploaded image to predict an emotion class. Images are not stored by default. Do not upload sensitive images. By proceeding, you consent to this processing.")
	consent = st.checkbox("I consent to process the uploaded image for this demo.")

API_URL_DEFAULT = os.environ.get("EMOTION_API_URL", "https://emotion-api.fly.dev")
api_url = st.text_input("API URL", value=API_URL_DEFAULT)
# Track API URL changes to invalidate caches
if st.session_state.get("api_url") != api_url:
	st.session_state.api_url = api_url
	st.session_state.pop("labels_cache", None)
health_col1, health_col2 = st.columns([1,3])
with health_col1:
	if st.button("Check API"):
		try:
			resp_h = requests.get(f"{api_url}/health", timeout=5)
			status = resp_h.json().get("status", "unknown") if resp_h.status_code == 200 else "down"
			st.session_state.api_health = status
		except Exception:
			st.session_state.api_health = "down"
with health_col2:
	status = st.session_state.get("api_health", None)
	if status is not None:
		color = "green" if status == "ok" else ("orange" if status == "degraded" else "red")
		st.markdown(f"**API Health:** <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)

with st.expander("API Metrics & Admin", expanded=False):
	met_col1, met_col2 = st.columns([1,1])
	with met_col1:
		if st.button("Load Metrics"):
			try:
				resp_m = requests.get(f"{api_url}/metrics", timeout=5)
				if resp_m.status_code == 200:
					st.session_state.api_metrics = resp_m.json()
				else:
					st.session_state.api_metrics = {"error": "unavailable"}
			except Exception:
				st.session_state.api_metrics = {"error": "unavailable"}
	with met_col2:
		if st.button("Reset Metrics"):
			try:
				resp_r = requests.post(f"{api_url}/admin/reset_metrics", timeout=5)
				st.success("Metrics reset" if resp_r.status_code == 200 else "Reset failed")
			except Exception:
				st.warning("Reset failed")
	metrics = st.session_state.get("api_metrics", None)
	if metrics:
		st.json(metrics)

col1, col2 = st.columns(2)
with col1:
	mode = st.radio("Inference mode", ["API", "Local"], horizontal=True)
with col2:
	show_cam = st.checkbox("Show Grad-CAM overlay", value=False)

if mode == "Local":
	st.info("Local mode requires a weights file at artifacts/best.pt")
	st.session_state.weights_path = st.text_input("Weights path", value=st.session_state.get("weights_path", "artifacts/best.pt"))
	device_options = ["cpu"]
	try:
		import torch
		if torch.cuda.is_available():
			device_options.append("cuda")
	except Exception:
		pass
	current_index = device_options.index(st.session_state.get("device", "cpu")) if st.session_state.get("device", "cpu") in device_options else 0
	st.session_state.device = st.selectbox("Device", device_options, index=current_index)

# Fetch labels from API with simple cache
labels = st.session_state.get("labels_cache", None)
if labels is None:
	try:
		resp_labels = requests.get(f"{api_url}/labels", timeout=5)
		if resp_labels.status_code == 200:
			labels = resp_labels.json().get("labels", None)
			if labels:
				st.session_state.labels_cache = labels
	except Exception:
		labels = None
if labels:
	st.caption(f"Labels: {', '.join(labels)}")

uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"]) 

# Webcam input
st.subheader("Webcam (optional)")
webcam = st.camera_input("Take a picture")

with st.expander("Batch Predict (API)", expanded=False):
	files_batch = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
	show_cam_batch = st.checkbox("Show Grad-CAM overlays for batch", value=False)
	batch_disabled = (not consent) or (not files_batch) or (mode != "API")
	if st.button("Predict Batch", disabled=batch_disabled):
		if not consent:
			st.warning("Please provide consent to proceed.")
		elif mode != "API":
			st.info("Batch predict is available only in API mode.")
		else:
			try:
				# Read files once and keep bytes for optional Grad-CAM
				file_bytes = []  # list[(name, bytes)]
				images_b64 = []
				for f in files_batch:
					b = f.read()
					file_bytes.append((getattr(f, "name", ""), b))
					images_b64.append(base64.b64encode(b).decode())
				payload = {"images_b64": images_b64}
				resp_b = requests.post(f"{api_url}/predict_batch", json=payload, timeout=60)
				resp_b.raise_for_status()
				items = resp_b.json().get("items", [])
				if items:
					rows = []
					for (name, _), it in zip(file_bytes, items):
						max_prob = max(it.get("scores", {}).values()) if it.get("scores") else None
						rows.append({
							"file": name,
							"emotion": it.get("emotion", ""),
							"max_prob": max_prob,
							"latency_ms": it.get("latency_ms", ""),
						})
					df = pd.DataFrame(rows)
					st.dataframe(df)
					csv_bytes = df.to_csv(index=False).encode()
					st.download_button("Download CSV", data=csv_bytes, file_name="batch_results.csv", mime="text/csv")

					if show_cam_batch:
						with st.spinner("Generating Grad-CAM overlays..."):
							for (name, b), it in zip(file_bytes, items):
								try:
									files = {"file": (name or "image.png", b, "image/png")}
									resp_cam = requests.post(f"{api_url}/explain", files=files, timeout=60)
									if resp_cam.status_code == 200:
										cam_b64 = resp_cam.json().get("cam_png_b64", None)
										if cam_b64:
											st.image(Image.open(io.BytesIO(base64.b64decode(cam_b64))), caption=f"{name}: {it.get('emotion','')} (Grad-CAM)", use_column_width=True)
									else:
										st.warning(f"Grad-CAM unavailable for {name}")
								except Exception:
									st.warning(f"Grad-CAM failed for {name}")
				else:
					st.info("No results returned.")
			except Exception as e:
				st.error(f"Batch API error: {e}")

# EMA smoothing state
if "ema_prob" not in st.session_state:
	st.session_state.ema_prob = None
if "ema_alpha" not in st.session_state:
	st.session_state.ema_alpha = 0.5

if "weights_path" not in st.session_state:
	st.session_state.weights_path = "artifacts/best.pt"
if "device" not in st.session_state:
	st.session_state.device = "cpu"

ema_alpha = st.slider("EMA smoothing (alpha)", 0.0, 1.0, st.session_state.ema_alpha, 0.05)
st.session_state.ema_alpha = ema_alpha

if st.button("Reset EMA"):
	st.session_state.ema_prob = None

# Client-side threshold for 'unknown'
thresh = st.slider("Unknown threshold (client)", 0.0, 1.0, 0.45, 0.01)


def apply_ema(probs: dict) -> dict:
	arr = np.array(list(probs.values()), dtype=np.float32)
	if st.session_state.ema_prob is None:
		st.session_state.ema_prob = arr
	else:
		st.session_state.ema_prob = ema_alpha * arr + (1 - ema_alpha) * st.session_state.ema_prob
	return {k: float(v) for k, v in zip(probs.keys(), st.session_state.ema_prob)}


def predict_via_api(pil_img: Image.Image):
	buf = io.BytesIO()
	pil_img.save(buf, format="PNG")
	files = {"file": ("capture.png", buf.getvalue(), "image/png")}
	resp = requests.post(f"{api_url}/predict", files=files, timeout=60)
	resp.raise_for_status()
	return resp.json()

@st.cache_resource(show_spinner=False)
def load_predictor(weights_path: str, device: str):
	from src.infer_service.predictor import Predictor
	return Predictor(weights_path=weights_path, device=device)

def render_probability_chart(scores: dict):
	try:
		df = pd.DataFrame({
			"label": list(scores.keys()),
			"probability": list(scores.values()),
		}).sort_values("probability", ascending=False)
		st.bar_chart(df.set_index("label"), use_container_width=True)
	except Exception:
		pass


img_source = None
if uploaded is not None:
	img_source = Image.open(uploaded).convert("RGB")
elif webcam is not None:
	img_source = Image.open(webcam).convert("RGB")

if img_source is not None:
	st.image(img_source, caption="Input", use_column_width=True)
	predict_disabled = (not consent) or (img_source is None) or (mode == "Local" and not os.path.exists(st.session_state.weights_path))
	if not consent:
		st.info("Please provide consent to enable prediction.")
	if mode == "Local" and not os.path.exists(st.session_state.weights_path):
		st.warning(f"Weights not found at {st.session_state.weights_path}")
	if st.button("Predict", disabled=predict_disabled):
		if not consent:
			st.warning("Please provide consent to proceed.")
		else:
			if mode == "API":
				with st.spinner("Calling API..."):
					try:
						out = predict_via_api(img_source)
						if ema_alpha > 0:
							out["scores"] = apply_ema(out["scores"])
							out["emotion"] = max(out["scores"], key=out["scores"].get)
						# Client-side unknown thresholding
						if max(out["scores"].values()) < thresh:
							out["emotion"] = "unknown"
						st.success(f"Prediction: {out['emotion']} ({out['latency_ms']} ms)")
						st.json(out["scores"])
						render_probability_chart(out["scores"])
						if show_cam:
							buf = io.BytesIO()
							img_source.save(buf, format="PNG")
							files = {"file": ("capture.png", buf.getvalue(), "image/png")}
							resp_cam = requests.post(f"{api_url}/explain", files=files, timeout=60)
							if resp_cam.status_code == 200:
								cam_b64 = resp_cam.json()["cam_png_b64"]
								st.image(Image.open(io.BytesIO(base64.b64decode(cam_b64))), caption="Grad-CAM", use_column_width=True)
							else:
								st.warning("Grad-CAM unavailable.")
					except Exception as e:
						st.error(f"API error: {e}")
			else:
				try:
					pred = load_predictor(st.session_state.weights_path, st.session_state.device)
					out = pred.predict(img_source)
					st.success(f"Prediction: {out['emotion']} ({out['latency_ms']} ms)")
					st.json(out["scores"])
					render_probability_chart(out["scores"])
				except Exception as e:
					st.error(f"Local inference failed: {e}")
