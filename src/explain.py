from typing import List, Optional
import io
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image


def get_last_conv_layer(model: torch.nn.Module):
	# Heuristic: grab last module with 'conv' in name or Conv layer
	last = None
	for name, module in model.named_modules():
		if "conv" in name.lower():
			last = module
	return last


def gradcam_overlay_bytes(model: torch.nn.Module, input_tensor: torch.Tensor, target_category: Optional[int] = None) -> bytes:
	model.eval()
	target_layer = get_last_conv_layer(model)
	if target_layer is None:
		raise RuntimeError("No convolutional layer found for Grad-CAM")
	# use_cuda argument removed for compatibility with latest pytorch-grad-cam
	cam = GradCAM(model=model, target_layers=[target_layer])
	grayscale_cam = cam(input_tensor=input_tensor, targets=None)
	grayscale_cam = grayscale_cam[0, :]
	img = input_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
	img = np.clip(img, 0, 1)
	vis = show_cam_on_image(img, grayscale_cam, use_rgb=True)
	pil = Image.fromarray(vis)
	buf = io.BytesIO()
	pil.save(buf, format="PNG")
	return buf.getvalue()
