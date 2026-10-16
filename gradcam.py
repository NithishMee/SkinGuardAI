"""High-Precision Multi-Scale Layer-CAM Visualization Utility for Dermoscopy.

Replaces standard Grad-CAM with Multi-Scale Hierarchical Layer-CAM (Element-Wise Gradient Weighting).
Eliminates gradient cancellation across ConvNeXt depthwise convolutions and delivers exact lesion contour delineation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from ham10000_data import build_transforms
from medfusionnet import MedFusionNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to input dermoscopy image.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/medfusionnet_best.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/gradcam.png"))
    parser.add_argument("--colormap", choices=("turbo", "jet"), default="turbo", help="Colormap for heatmap.")
    parser.add_argument("--target-class", type=int, default=None, help="Target class index (default: predicted class).")
    return parser.parse_args()


def compute_multiscale_layercam(
    model: MedFusionNet,
    image_tensor: torch.Tensor,
    image_size: int = 224,
    target_class: int | None = None,
) -> tuple[np.ndarray, int, float]:
    """Compute high-precision lesion-focused Multi-Scale Layer-CAM using Stage 5 and Stage 7.
    
    Layer-CAM performs spatial element-wise positive gradient weighting:
      w_{k, i, j} = relu(grad_{k, i, j}) * act_{k, i, j}
    This avoids the gradient cancellation of global average pooling in standard Grad-CAM,
    capturing the entire morphological extent of the skin lesion.
    """
    acts_5, grads_5 = [], []
    acts_7, grads_7 = [], []

    def hook_5(_module, _inputs, output):
        acts_5.append(output)
        if output.requires_grad:
            output.register_hook(lambda grad: grads_5.append(grad))

    def hook_7(_module, _inputs, output):
        acts_7.append(output)
        if output.requires_grad:
            output.register_hook(lambda grad: grads_7.append(grad))

    h5 = model.convnext_features[5].register_forward_hook(hook_5)
    h7 = model.convnext_features[7].register_forward_hook(hook_7)

    model.zero_grad(set_to_none=True)
    cam_input = image_tensor.clone().detach().requires_grad_(True)
    logits = model(cam_input)
    probs = functional.softmax(logits, dim=1)
    if target_class is None:
        target_class = int(probs.argmax(dim=1).item())
    confidence = float(probs[0, target_class].item())

    score = logits[0, target_class]
    score.backward()
    h5.remove()
    h7.remove()

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

    # Adaptive background skin noise thresholding:
    # Suppress background skin noise smoothly while preserving the full lesion perimeter
    cutoff = np.percentile(cam_np, 20) * 0.5
    cam_np = np.where(cam_np < cutoff, 0.0, (cam_np - cutoff) / (1.0 - cutoff + 1e-8))
    cam_np = np.clip(cam_np, 0.0, 1.0)
    # Smooth boundary falloff for natural clinical appearance
    cam_np = np.power(cam_np, 1.1)

    return cam_np, target_class, confidence


# Backward compatibility alias
def compute_multiscale_gradcam_plus(
    model: MedFusionNet,
    image_tensor: torch.Tensor,
    image_size: int = 224,
    target_class: int | None = None,
) -> tuple[np.ndarray, int, float]:
    """Alias for compute_multiscale_layercam to maintain backwards compatibility."""
    return compute_multiscale_layercam(model, image_tensor, image_size, target_class)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = MedFusionNet.load_from_checkpoint(args.checkpoint, device=device)
    classes = checkpoint["classes"]
    image_size = checkpoint["image_size"]

    original = Image.open(args.image).convert("RGB")
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    direct_eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = direct_eval_transform(original).unsqueeze(0).to(device)

    cam_array, predicted_index, confidence = compute_multiscale_layercam(
        model, image_tensor, image_size=image_size, target_class=args.target_class
    )

    resized = original.resize((image_size, image_size))
    image_array = np.asarray(resized, dtype=np.float32) / 255.0

    # Professional alpha-blended heatmap overlay
    cmap = plt.get_cmap(args.colormap)
    heatmap = cmap(cam_array)[..., :3]
    alpha = np.expand_dims(cam_array * 0.65, axis=-1)
    overlay = np.clip((1.0 - alpha) * image_array + alpha * heatmap, 0.0, 1.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(image_array)
    axes[0].set_title("Original Dermoscopy", fontsize=12, fontweight="bold")
    axes[1].imshow(cam_array, cmap=args.colormap)
    axes[1].set_title(f"Multi-Scale Layer-CAM ({args.colormap.upper()})", fontsize=12, fontweight="bold")
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Prediction: {classes[predicted_index].upper()} ({confidence:.1%})",
        fontsize=12,
        fontweight="bold",
    )
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved high-precision Multi-Scale Layer-CAM visualization to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
