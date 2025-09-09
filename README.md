# Facial Emotion ML

End-to-end facial emotion recognition project with training, evaluation, explainability (Grad-CAM), and serving via FastAPI with an optional Streamlit UI.

## Quickstart

- Create and activate a Python 3.11 environment
- Install deps:

```bash
pip install -r requirements.txt
```

### Data prep

```bash
# Face crop/alignment (optional)
python -m src.preprocess_faces --src data/raw/<dataset> --dst data/interim/<dataset> --margin 0.1

python -m src.clean_data --root data/raw/<dataset> --out-manifest data/processed/cleaned.csv --delete-duplicates
python -m src.summarize_data --manifest data/processed/cleaned.csv --out data/data_summary.json
python -m src.data_utils --root /path/to/class_folders --out data/processed
```

### Train (example)

```bash
python -m src.train --backbone efficientnet_b0 --epochs 20 --batch-size 128 --use-focal --use-weighted-sampler --export-torchscript
```

### Evaluate

```bash
python -m src.evaluate --weights artifacts/best.pt --save-cm
```

### Benchmarks & sanity checks

```bash
# Loader throughput
python -m src.bench_loader --data data/processed/train --batch-size 128 --num-workers 4 --iters 500

# Augmentation montage
python -m src.viz_aug --image path/to/sample.jpg --n 12 --cols 6 --out reports/aug_montage.png
```

### Serve locally

```bash
# Optional envs
env FACE_DETECT=1 EMOTION_CONF_THRESH=0.45 CORS_ORIGINS=* METRICS_WINDOW=200 uvicorn src.infer_service.app:app --host 0.0.0.0 --port 8000 --reload
```

- Endpoints:
	- GET `/health`, `/version`, `/metrics`, `/labels`
	- POST `/predict`, `/predict_json`, `/predict_batch`, `/explain`
	- Admin: POST `/admin/purge_logs`, `/admin/reset_metrics`; GET `/admin/config`; DELETE `/admin/request/{request_id}`

### Streamlit demo

```bash
streamlit run ui/app_streamlit.py
```

## Deploy

### Render (API)

1) Fork or push this repo
2) Create a Web Service from repo
3) Set env vars (examples):

```
FACE_DETECT=1
EMOTION_CONF_THRESH=0.45
CORS_ORIGINS=*
REQUEST_LOG_PATH=logs/requests.csv
WEIGHTS_PATH=artifacts/best.pt
DEVICE=cpu
# Optional if weights are not committed:
WEIGHTS_URL=https://your-bucket/best.pt
```

Start command is already set by `render.yaml`.

### Hugging Face Spaces (UI)

1) Create a Space (Streamlit)
2) Add this repo or just the `ui/` folder and a minimal `app.py` that imports `ui/app_streamlit.py`
3) Set a Space secret `EMOTION_API_URL` to your Render API URL (e.g., `https://emotion-api.onrender.com`)
4) The UI allows overriding the API URL at runtime

### Offline inference CLI

```bash
python -m src.infer_cli --weights artifacts/best.pt --input path/to/images_or_dir --out inference_results.csv
```

### Docker

```bash
docker build -t emotion-api .
docker run -p 8000:8000 -e FACE_DETECT=1 -e EMOTION_CONF_THRESH=0.45 -e CORS_ORIGINS=* emotion-api
```

## Ethics & Privacy

- Consent required in the UI
- No images are stored by default
- Logs contain hashed IDs only
- Admin deletion endpoint is provided for removing specific request log entries.

## License

MIT
