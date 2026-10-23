"""Lazily imported inference runtime. Shared hooks and telemetry are serialized."""
from __future__ import annotations
import os
from threading import RLock
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as functional
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from medfusionnet import MedFusionNet

_INFERENCE_LOCK = RLock()

@st.cache_resource
def load_medfusionnet_model(checkpoint_path: str) -> tuple[MedFusionNet, dict, torch.device]:
    torch.set_num_threads(max(1, int(os.environ.get("SKINGUARD_CPU_THREADS", "2"))))
    device = torch.device("cuda" if torch.cuda.device_count() > 0 else "cpu")
    model, checkpoint = MedFusionNet.load_from_checkpoint(checkpoint_path, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, checkpoint, device


def predict_with_multiscale_gradcam(*args, **kwargs):
    # cache_resource shares a model across sessions; hooks and last_fusion_weights
    # must belong to one request for the entire forward/backward operation.
    with _INFERENCE_LOCK:
        return _predict(*args, **kwargs)


def _predict(
    model: MedFusionNet,
    image: Image.Image,
    image_size: int,
    device: torch.device,
    use_tta: bool = False,
    confidence_threshold: float = 0.90,
    include_heatmap: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, int]:
    """Execute memory-safe inference and high-precision Multi-Scale Layer-CAM."""
    import gc

    # Direct bilinear resize ensures 1-to-1 coordinate alignment with display view (no cropping distortion)
    direct_eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_tensor = direct_eval_transform(image).unsqueeze(0).to(device)

    # 1. Run inference with adaptive learned attention
    with torch.inference_mode():
        p1 = functional.softmax(model(image_tensor), dim=1)
        if use_tta:
            p2 = functional.softmax(model(torch.flip(image_tensor, dims=[-1])), dim=1)
            p3 = functional.softmax(model(torch.flip(image_tensor, dims=[-2])), dim=1)
            p4 = functional.softmax(model(torch.flip(image_tensor, dims=[-2, -1])), dim=1)
            probabilities = (p1 + p2 + p3 + p4) / 4.0
        else:
            probabilities = p1
        predicted_index = int(probabilities.argmax(dim=1).item())

    fusion_weights = model.last_fusion_weights[0].detach().cpu().numpy()
    confidence = float(probabilities[0, predicted_index].item())

    if not include_heatmap or confidence < confidence_threshold:
        return probabilities[0].cpu().numpy(), None, fusion_weights, predicted_index

    # 2. Multi-Scale Layer-CAM (Element-Wise Gradient Weighting)
    # Hooks Stage 5 (14x14 micro-textures & pigment networks) and Stage 7 (7x7 global context)
    acts_5, grads_5 = [], []
    acts_7, grads_7 = [], []

    def hook_5(_module, _inputs, output):
        acts_5.append(output)
        if output.requires_grad:
            output.register_hook(lambda gradient: grads_5.append(gradient))

    def hook_7(_module, _inputs, output):
        acts_7.append(output)
        if output.requires_grad:
            output.register_hook(lambda gradient: grads_7.append(gradient))

    h5 = model.convnext_features[5].register_forward_hook(hook_5)
    h7 = model.convnext_features[7].register_forward_hook(hook_7)

    try:
        with torch.enable_grad():
            cam_input = image_tensor.clone().detach().requires_grad_(True)
            cam_logits = model(cam_input)
            cam_logits[0, predicted_index].backward()

        act5, grad5 = acts_5[0], grads_5[0]
        act7, grad7 = acts_7[0], grads_7[0]

        # Stage 5 Layer-CAM (Fine Pigment Reticulations, Borders, and Micro-margins)
        cam5 = functional.relu(act5 * functional.relu(grad5)).sum(dim=1, keepdim=True)
        cam5 = functional.interpolate(cam5, size=(image_size, image_size), mode="bilinear", align_corners=False)
        cam5_norm = (cam5 - cam5.min()) / (cam5.max() - cam5.min() + 1e-8)

        # Stage 7 Layer-CAM (Global Lesion Geometry and Semantic Core)
        cam7 = functional.relu(act7 * functional.relu(grad7)).sum(dim=1, keepdim=True)
        cam7 = functional.interpolate(cam7, size=(image_size, image_size), mode="bilinear", align_corners=False)
        cam7_norm = (cam7 - cam7.min()) / (cam7.max() - cam7.min() + 1e-8)

        # Multi-scale harmonic blending: 60% Stage 5 (borders & lesion structure) + 40% Stage 7 (semantic context)
        fused_cam = 0.60 * cam5_norm + 0.40 * cam7_norm
        fused_cam = (fused_cam - fused_cam.min()) / (fused_cam.max() - fused_cam.min() + 1e-8)
        cam_np = fused_cam[0, 0].detach().cpu().numpy()

        # Adaptive background skin noise suppression
        cutoff = np.percentile(cam_np, 20) * 0.5
        cam_np = np.where(cam_np < cutoff, 0.0, (cam_np - cutoff) / (1.0 - cutoff + 1e-8))
        cam_np = np.clip(cam_np, 0.0, 1.0)
        # Smooth boundary falloff for natural clinical appearance
        cam_np = np.power(cam_np, 1.1)

    finally:
        h5.remove()
        h7.remove()
        del acts_5, acts_7, grads_5, grads_7
        # Function-local tensors are released on return or exception. Do not
        # reference partially initialized locals here and mask the real error.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fusion_weights = model.last_fusion_weights[0].detach().cpu().numpy()
    return probabilities[0].cpu().numpy(), cam_np, fusion_weights, predicted_index


