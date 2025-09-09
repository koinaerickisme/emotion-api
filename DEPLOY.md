# Deploy Guide

## Render (FastAPI)

1. Push repo to GitHub
2. Create new Web Service on Render, connect repo
3. Use:
	- Build Command: `pip install -r requirements.txt`
	- Start Command: `uvicorn src.infer_service.app:app --host 0.0.0.0 --port $PORT`
4. Set env vars as desired: `FACE_DETECT=1`, `EMOTION_CONF_THRESH=0.45`, `CORS_ORIGINS=*`
5. Upload `artifacts/best.pt` and `artifacts/labels.json` (e.g., via Render Disk or bake into image)

## Hugging Face Spaces (Streamlit)

1. Create a Space (Streamlit)
2. Add `ui/app_streamlit.py` as the entrypoint
3. Include `requirements.txt`
4. If calling API, set `EMOTION_API_URL` in Space secrets to your API URL

## Docker

```bash
docker build -t emotion-api .
docker run -p 8000:8000 -e FACE_DETECT=1 -e EMOTION_CONF_THRESH=0.45 emotion-api
```
