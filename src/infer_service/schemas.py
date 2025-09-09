from pydantic import BaseModel
from typing import Dict, List


class PredictResponse(BaseModel):
	emotion: str
	scores: Dict[str, float]
	model_version: str
	latency_ms: float


class HealthResponse(BaseModel):
	status: str


class ImagePayload(BaseModel):
	image_b64: str


class ExplainResponse(BaseModel):
	emotion: str
	cam_png_b64: str
	latency_ms: float


class ImageBatchPayload(BaseModel):
	images_b64: List[str]


class PredictBatchResponse(BaseModel):
	items: List[PredictResponse]
