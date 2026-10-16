"""SkinGuardAI: Deep Multi-Modal Dermoscopy Diagnostic Workstation.

Architecture: MedFusionNet (ConvNeXt-Tiny + Vision Transformer)
Clinical Protocol: Fixed >=90.0% Confidence Safety Gate & Multi-Scale Grad-CAM++
Clinical Features: Patient Demographics Intake, Centered Upload, & Physician PDF/EMR Export
"""

from __future__ import annotations

import base64
import io
import re
from datetime import datetime
from pathlib import Path

import streamlit as st

# ==============================================================================
# 1. Page Configuration & Cohesive Medical Workstation Theme
# ==============================================================================
st.set_page_config(
    page_title="SkinGuardAI | Clinical Diagnostic Workstation",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    "<style>" + (Path(__file__).parent / "static" / "workstation.css").read_text(encoding="utf-8") + "</style>",
    unsafe_allow_html=True,
)

# Show a real loading state before heavy imports and model initialization.
startup_view = st.empty()
if not st.session_state.get("workspace_ready", False):
    startup_view.markdown(
        """<div class="sg-startup" role="status" aria-live="polite">
            <div class="sg-startup-content">
                <div class="sg-loading-ring" aria-hidden="true"></div>
                <div class="sg-wordmark">SkinGuard<span>AI</span></div>
                <p>Preparing your workspace</p>
                <small>Loading image analysis tools…</small>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as functional
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    
    from ham10000_data import build_transforms
    from medfusionnet import MedFusionNet
except Exception:
    startup_view.empty()
    raise

# ==============================================================================
# 2. Clinical Classification Metadata & Fixed System Configuration
# ==============================================================================
CHECKPOINT_PATH = Path("checkpoints/medfusionnet_best.pt")
FIXED_THRESHOLD = 0.90
threshold = 0.90
threshold_percent = 90
use_tta = False

CLASS_METADATA = {
    "mel": {
        "name": "Melanoma",
        "category": "Malignant Neoplasm",
        "severity": "CRITICAL",
        "icd10": "C43.9",
        "is_cancer": True,
        "color": "#E11D48",
        "description": "Malignant neoplasm of melanocytes with high invasive and metastatic potential. Characterized by prominent structural asymmetry, atypical pigment reticulation, and multi-hue coloration.",
        "protocol": "Immediate urgent dermatological oncology referral and complete surgical excision with wide safety margins.",
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "category": "Malignant Neoplasm",
        "severity": "HIGH",
        "icd10": "C44.91",
        "is_cancer": True,
        "color": "#E11D48",
        "description": "Slowly expanding malignant cutaneous neoplasm derived from follicular germinative cells. Characterized by arborizing telangiectasia, blue-gray ovoid nests, and ulceration.",
        "protocol": "Mohs micrographic surgery, standard surgical excision, or targeted photodynamic therapy.",
    },
    "akiec": {
        "name": "Actinic Keratosis / Bowen's Disease",
        "category": "Pre-Malignant Neoplasm",
        "severity": "MODERATE",
        "icd10": "L57.0",
        "is_cancer": True,
        "color": "#D97706",
        "description": "Dysplastic epidermal lesion induced by chronic ultraviolet radiation exposure. Exhibits strawberry pattern, erythema, and superficial hyperkeratosis.",
        "protocol": "Cryotherapy with liquid nitrogen, topical 5-Fluorouracil, or photodynamic field therapy.",
    },
    "bkl": {
        "name": "Benign Keratosis-like Lesion",
        "category": "Benign",
        "severity": "LOW",
        "icd10": "L82.1",
        "is_cancer": False,
        "color": "#059669",
        "description": "Benign epidermal proliferation encompassing seborrheic keratosis, solar lentigo, and lichen planus-like keratosis. Exhibits milia-like cysts and comedo-like openings.",
        "protocol": "Harmless benign lesion. Reassurance indicated; intervention only if symptomatic or cosmetically requested.",
    },
    "nv": {
        "name": "Melanocytic Nevus (Common Mole)",
        "category": "Benign",
        "severity": "LOW",
        "icd10": "D22.9",
        "is_cancer": False,
        "color": "#059669",
        "description": "Benign architectural aggregation of normal melanocytes showing homogeneous, uniform pigmentation, regular reticular meshwork, and symmetry.",
        "protocol": "Benign mole. Routine annual self-examination using standardized ABCDE criteria.",
    },
    "vasc": {
        "name": "Vascular Lesion (Hemangioma / Angioma)",
        "category": "Benign",
        "severity": "LOW",
        "icd10": "D18.01",
        "is_cancer": False,
        "color": "#059669",
        "description": "Benign proliferation of cutaneous blood vessels with distinct red-purple lacunae, lagoons, and well-circumscribed vascular spaces.",
        "protocol": "Completely benign. Reassurance provided; pulse dye laser ablation only if recurrent bleeding occurs.",
    },
    "df": {
        "name": "Dermatofibroma",
        "category": "Benign",
        "severity": "LOW",
        "icd10": "D23.9",
        "is_cancer": False,
        "color": "#059669",
        "description": "Benign fibrous dermal histiocytoma presenting with central white scar-like patch surrounded by delicate faint pigment network and positive dimple sign.",
        "protocol": "Benign lesion. Conservative clinical management; no surgical intervention required.",
    },
}

ANATOMIC_SITES = [
    "Back / Torso",
    "Lower Extremity (Leg / Foot)",
    "Upper Extremity (Arm / Hand)",
    "Head / Scalp / Neck",
    "Face / Facial Region",
    "Anterior Torso (Chest / Abdomen)",
    "Acral / Palmar / Plantar",
    "Genital / Perianal",
    "Other / Unspecified Anatomical Site",
]

# ==============================================================================
# 3. Model Loading & Inference Engine
# ==============================================================================
@st.cache_resource
def load_medfusionnet_model(checkpoint_path: str) -> tuple[MedFusionNet, dict, torch.device]:
    device = torch.device("cuda" if torch.cuda.device_count() > 0 else "cpu")
    model, checkpoint = MedFusionNet.load_from_checkpoint(checkpoint_path, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, checkpoint, device


def validate_dermoscopy_domain(image: Image.Image) -> tuple[bool, str, dict]:
    """Verify that an uploaded image conforms to dermatoscopic/cutaneous domain characteristics.
    
    Detects Out-of-Distribution (OOD) non-skin subjects such as plants, flowers, foliage,
    household objects, synthetic text, or outdoor scenery.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # 1. Vegetation / plant foliage detection (dominant green: G > R and G > B)
    green_dominant = (g > r * 1.05) & (g > 0.20)
    green_ratio = float(green_dominant.mean())

    # 2. Unnatural blue/cyan detection (sky, water, synthetic blue)
    blue_dominant = (b > r * 1.15) & (b > g * 1.10) & (b > 0.20)
    blue_ratio = float(blue_dominant.mean())

    # 3. Human skin chromaticity spectrum (R >= G*0.85 and R >= B*0.85 for non-black pixels)
    valid_pixels = (r > 0.08) | (g > 0.08) | (b > 0.08)  # exclude black dermatoscope border
    if valid_pixels.sum() > 0:
        skin_hue_ratio = float(((r >= g * 0.85) & (r >= b * 0.85) & valid_pixels).sum() / valid_pixels.sum())
    else:
        skin_hue_ratio = 0.0

    stats = {
        "green_ratio": green_ratio,
        "blue_ratio": blue_ratio,
        "skin_hue_ratio": skin_hue_ratio,
    }

    if green_ratio > 0.06:
        return (
            False,
            f"Significant green vegetation/foliage detected ({green_ratio:.1%} of image area). Clinical dermoscopy scans do not contain plant foliage.",
            stats,
        )
    if blue_ratio > 0.12:
        return (
            False,
            f"Unnatural blue/cyan chromaticity detected ({blue_ratio:.1%} of image area). Skin lesion scans do not exhibit dominant blue backgrounds.",
            stats,
        )
    if skin_hue_ratio < 0.60:
        return (
            False,
            f"Image does not conform to human cutaneous skin-tone spectrum (only {skin_hue_ratio:.1%} matches epidermal/dermal tones).",
            stats,
        )

    return True, "Valid dermatoscopic scan.", stats


def predict_with_multiscale_gradcam(
    model: MedFusionNet,
    image: Image.Image,
    image_size: int,
    device: torch.device,
    use_tta: bool = False,
    confidence_threshold: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
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

    # Clean any lingering forward hooks from previous executions
    model.convnext_features[5]._forward_hooks.clear()
    model.convnext_features[7]._forward_hooks.clear()

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
        model.convnext_features[5]._forward_hooks.clear()
        model.convnext_features[7]._forward_hooks.clear()
        del acts_5, acts_7, grads_5, grads_7
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fusion_weights = model.last_fusion_weights[0].detach().cpu().numpy()
    return probabilities[0].cpu().numpy(), cam_np, fusion_weights, predicted_index


# ==============================================================================
# 4. Clinical Report Generation Helpers (PDF & EMR Format)
# ==============================================================================
def generate_clinical_pdf_report(
    patient_id: str,
    patient_name: str,
    patient_age: str,
    anatomic_site: str,
    clinical_notes: str,
    image_name: str,
    active_image: Image.Image,
    cam: np.ndarray,
    overlay_display: np.ndarray,
    predicted_code: str,
    confidence: float,
    is_confident: bool,
    meta: dict,
    probabilities: np.ndarray,
    classes: list[str],
    fusion_weights: np.ndarray,
    threshold_percent: int = 90,
) -> bytes:
    """Render a single A4 summary; full notes remain available in the text export."""
    import textwrap
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.font_manager import FontProperties

    buf = io.BytesIO()
    ink, muted, blue = "#233C50", "#627482", "#2B638C"
    fig = plt.figure(figsize=(8.27, 11.69), dpi=160)
    fig.patch.set_facecolor("white")
    renderer = fig.canvas.get_renderer()

    def text(x, y, value, size=8, color=ink, bold=False, **kwargs):
        fig.text(x, y, str(value), fontsize=size, color=color,
                 weight="bold" if bold else "normal", va="top", **kwargs)

    def wrapped(x, y, value, width, max_lines, size=8, bold=False, overflow="..."):
        # Measure glyph widths so long filenames and wide letters cannot spill out.
        font = FontProperties(size=size, weight="bold" if bold else "normal")
        max_px = width * fig.get_figwidth() * fig.dpi
        def fits(line):
            return renderer.get_text_width_height_descent(line, font, False)[0] <= max_px
        lines = []
        for block in str(value).splitlines() or [""]:
            line = ""
            for word in block.split():
                if line and not fits(line + " " + word):
                    lines.append(line)
                    line = ""
                for char in word:
                    if not fits(line + char):
                        lines.append(line)
                        line = ""
                    line += char
                line += " "
            lines.append(line.rstrip())
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            if overflow != "...":
                lines[-1] = overflow
            else:
                while lines[-1] and not fits(lines[-1] + "..."):
                    lines[-1] = lines[-1][:-1]
                lines[-1] += "..."
        for i, line in enumerate(lines):
            text(x, y - i * size * 1.4 / (72 * 11.69), line, size=size, bold=bold)

    def heading(x, y, title):
        text(x, y, title.upper(), size=8, color=blue, bold=True)

    now = datetime.now()
    text(.065, .955, "SkinGuardAI", size=22, bold=True)
    text(.065, .922, "Clinical dermoscopy report", size=10, color=muted)
    text(.935, .952, f"SKG-{now.strftime('%Y%m%d-%H%M%S')}", size=8, ha="right")
    text(.935, .931, now.strftime("%d %b %Y  |  %H:%M"), size=8, color=muted, ha="right")
    fig.lines.append(plt.Line2D([.065,.935],[.902,.902],transform=fig.transFigure,color=blue,lw=1.4))

    heading(.065,.886,"Patient")
    wrapped(.065,.866,patient_name or "Unspecified",.42,1,9,bold=True)
    wrapped(.065,.846,f"ID: {patient_id or 'PT-UNASSIGNED'}  |  Age: {patient_age or 'N/A'}",.42,1)
    heading(.525,.886,"Scan details")
    wrapped(.525,.866,f"Site: {anatomic_site or 'Unspecified'}",.41,1,9,bold=True)
    wrapped(.525,.846,f"Source: {image_name}",.41,1)

    accent = ("#A33C51" if meta.get("is_cancer",False) else "#28765D") if is_confident else "#936A28"
    fig.patches.append(FancyBboxPatch((.065,.637),.87,.184,boxstyle="round,pad=0.008,rounding_size=0.012",transform=fig.transFigure,facecolor="#F4F7FA",edgecolor="#DDE5EC",zorder=-1))
    status = ("MALIGNANT NEOPLASM CONFIRMED" if meta.get("is_cancer",False) else "BENIGN LESION CONFIRMED") if is_confident else "UNKNOWN CATEGORY / INCONCLUSIVE"
    text(.083,.805,status,size=8,color=accent,bold=True)
    diagnosis = meta.get("name",predicted_code) if is_confident else "Clinical review required"
    wrapped(.083,.782,diagnosis,.83,2,size=12,bold=True)
    if is_confident:
        text(.083,.741,f"Certainty: {confidence:.1%}  |  ICD-10: {meta.get('icd10','N/A')}  |  Severity: {meta.get('severity','MODERATE')}  |  Gate: >= {threshold_percent}.0%",size=8,color=accent)
        wrapped(.083,.719,"Pathology: " + meta.get("description",""),.83,3,size=8)
        wrapped(.083,.675,"Protocol: " + meta.get("protocol",""),.83,2,size=8,bold=True)
    else:
        text(.083,.741,f"Certainty suppressed by safety gate (< {threshold_percent}.0%)  |  Clinical risk: UNRESOLVED",size=8,color=accent)
        wrapped(.083,.719,"Clinical Advisory: Automated prediction certainty does not satisfy the mandatory 90.0% threshold.",.83,3,size=8)
        wrapped(.083,.675,"Mandatory Action: Direct in-person clinical dermatological evaluation and histopathological biopsy mandated.",.83,2,size=8,bold=True)

    heading(.065,.611,"Image attribution")
    for x, data, label, cmap in [
        (.065,active_image.convert("RGB"),"01  Dermoscopy scan",None),
        (.365,np.nan_to_num(np.clip(cam,0,1)),"02  Layer-CAM saliency","turbo"),
        (.665,np.nan_to_num(np.clip(overlay_display,0,1)),"03  Pathological overlay",None),
    ]:
        ax=fig.add_axes([x,.414,.27,.177])
        ax.imshow(data,cmap=cmap)
        ax.axis("off")
        text(x,.404,label,size=8,bold=True)
    text(.065,.382,f"ConvNeXt: {float(fusion_weights[0])*100:.1f}%   |   ViT: {float(fusion_weights[1])*100:.1f}%   |   " + ("Safety Gate Passed" if is_confident else "Borderline Attention Focus"),size=7.5,color=muted)

    heading(.065,.350,"Differential probabilities")
    text(.065,.330,f"Red marker: {threshold_percent}.0% clinical threshold",size=7,color=muted)
    for row,i in enumerate(np.argsort(probabilities)[::-1]):
        y=.308-row*.027
        code=classes[i]
        label=f"{code.upper()} / {CLASS_METADATA.get(code,{}).get('name',code)}"
        wrapped(.065,y,label,.355,1,size=7)
        text(.485,y,f"{float(probabilities[i]):.1%}",size=7,bold=True,ha="right")
        fig.patches.append(plt.Rectangle((.065,y-.017),.42,.0035,transform=fig.transFigure,facecolor="#E3EBF0",edgecolor="none"))
        fig.patches.append(plt.Rectangle((.065,y-.017),.42*float(probabilities[i]),.0035,transform=fig.transFigure,facecolor=blue,edgecolor="none"))
        tx=.065+.42*threshold_percent/100
        fig.lines.append(plt.Line2D([tx,tx],[y-.019,y-.012],transform=fig.transFigure,color="#A33C51",lw=.7))

    heading(.535,.350,"Physician review")
    text(.535,.328,"[ ] Confirmed    [ ] Urgent biopsy",size=7.5)
    text(.535,.310,"[ ] Surgical excision    [ ] Observation",size=7.5)
    text(.535,.284,"Clinical notes",size=8,bold=True)
    wrapped(.535,.264,clinical_notes.strip() if clinical_notes else "None recorded.",.40,8,size=8,overflow="[Excerpt - full notes in text export]")
    fig.lines.append(plt.Line2D([.065,.935],[.120,.120],transform=fig.transFigure,color="#DDE5EC",lw=.8))
    text(.065,.106,"Impression: ___________________________",size=8)
    text(.535,.106,"Physician: __________________________",size=8)
    text(.065,.083,"License / Facility: ______________________",size=8)
    text(.535,.083,"Signature / Date: _____________________",size=8)
    text(.065,.049,"AI diagnostic decision support. All outputs must be corroborated by histopathological biopsy.",size=7,color=muted)
    text(.935,.030,"1 / 1",size=7,color=muted,ha="right")
    with PdfPages(buf) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    return buf.getvalue()


def generate_clinical_text_report(
    patient_id: str,
    patient_name: str,
    patient_age: str,
    anatomic_site: str,
    clinical_notes: str,
    image_name: str,
    predicted_code: str,
    confidence: float,
    is_confident: bool,
    meta: dict,
    probabilities: np.ndarray,
    classes: list[str],
    fusion_weights: np.ndarray,
    threshold_percent: int = 90,
) -> str:
    """Generate structured Clinical Progress Note text formatted for hospital EMR / EHR export."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_id = f"SKG-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    lines = [
        "=" * 78,
        "SKINGUARDAI CLINICAL DERMOSCOPY CONSULTATION REPORT",
        "Deep Multi-Modal Lesion Assessment & Explainable AI (XAI) Synthesis",
        "=" * 78,
        f"Report Reference : {report_id}",
        f"Date & Time      : {now_str}",
        f"Patient ID / MRN : {patient_id or 'PT-UNASSIGNED'}",
        f"Patient Name     : {patient_name or 'Unspecified'}",
        f"Patient Age      : {patient_age or 'N/A'}",
        f"Anatomic Site    : {anatomic_site or 'Unspecified'}",
        f"Source Image     : {image_name}",
        f"Safety Protocol  : Mandatory >={threshold_percent}.0% Clinical Confidence Gate",
        f"Neural Model     : MedFusionNet Dual Backbone (ConvNeXt-Tiny + ViT)",
        "-" * 78,
        "PRIMARY CLINICAL FINDINGS",
        "-" * 78,
    ]
    if is_confident:
        category_str = meta.get("category", "Confirmed").upper()
        lines.extend([
            f"Diagnostic Status: CONFIRMED {category_str} (Passed >={threshold_percent}.0% Gate)",
            f"Diagnosis Code   : {predicted_code.upper()} - {meta.get('name', predicted_code)}",
            f"Certainty Metric : {confidence:.1%}",
            f"ICD-10 Code      : {meta.get('icd10', 'N/A')}",
            f"Clinical Severity: {meta.get('severity', 'HIGH')}",
            "",
            "Pathological Characteristics:",
            f"  {meta.get('description', '')}",
            "",
            "Recommended Clinical Protocol:",
            f"  {meta.get('protocol', '')}",
        ])
    else:
        lines.extend([
            f"Diagnostic Status: SAFETY DECISION GATE INTERCEPTION (< {threshold_percent}.0%)",
            "Classification   : Unknown Category (Inconclusive)",
            f"Certainty Metric : {confidence:.1%} (Below {threshold_percent}.0% Gate)",
            "Clinical Risk    : Unresolved / Equivocal",
            "",
            "Safety Advisory:",
            f"  Model confidence did not reach the mandatory {threshold_percent}.0% decision cutoff.",
            f"  Primary neural pattern indicates {meta.get('name', predicted_code)} ({confidence:.1%}), but requires biopsy confirmation.",
            "",
            "Mandated Clinical Action:",
            "  Direct in-person clinical dermatological evaluation, high-magnification",
            "  dermatoscope review, and histopathological biopsy are required.",
        ])

    lines.extend([
        "-" * 78,
        "EXPLAINABLE AI (XAI) SALIENCY & ATTENTION SYNTHESIS",
        "-" * 78,
        f"ConvNeXt-Tiny Branch (Local Texture & Edge Networks): {fusion_weights[0] * 100:.1f}%",
        f"Vision Transformer Branch (Global Context Networks) : {fusion_weights[1] * 100:.1f}%",
        "",
        "-" * 78,
        "7-CLASS DIFFERENTIAL DIAGNOSIS DISTRIBUTION",
        "-" * 78,
    ])

    sorted_idx = np.argsort(probabilities)[::-1]
    for i in sorted_idx:
        c_code = classes[i]
        c_p = float(probabilities[i])
        c_meta = CLASS_METADATA.get(c_code, {})
        c_name = c_meta.get("name", c_code)
        lines.append(f"  [{c_code.upper():5s}] {c_name:38s}: {c_p:6.1%}")
    if not is_confident:
        lines.append(f"  [SAFETY ADVISORY] Case flagged under Unknown Category (<{threshold_percent}.0% Confirmation Cutoff).")

    lines.extend([
        "",
        "-" * 78,
        "PHYSICIAN CONSULTATION NOTES & SIGN-OFF",
        "-" * 78,
        f"Clinical Notes   : {clinical_notes.strip() if clinical_notes else 'None recorded.'}",
        "",
        "Clinical Determination: [ ] Confirmed  [ ] Biopsy Ordered  [ ] Follow-up Required",
        "Physician Sign-Off   : ___________________________   Date: _______________",
        "=" * 78,
        "DISCLAIMER: Research decision-support system. Histopathological biopsy required.",
        "=" * 78,
    ])
    return "\n".join(lines)


# ==============================================================================
# 5. Workflow State Initialization & Navigation
# ==============================================================================
if "current_page" not in st.session_state:
    st.session_state.current_page = "intake"
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "patient_id" not in st.session_state:
    st.session_state.patient_id = "PID-2026-084"
if "patient_name" not in st.session_state:
    st.session_state.patient_name = "Eleanor Vance"
if "patient_age" not in st.session_state:
    st.session_state.patient_age = "48"
if "anatomic_site" not in st.session_state:
    st.session_state.anatomic_site = ANATOMIC_SITES[0]
if "clinical_notes" not in st.session_state:
    st.session_state.clinical_notes = ""

# Guard: redirect to intake if results view has no payload
if st.session_state.current_page == "results" and st.session_state.analysis_results is None:
    st.session_state.current_page = "intake"
    st.rerun()

# ==============================================================================
# 6. Workstation Header Component (Compact Single-Bar Viewport Fit)
# ==============================================================================
is_intake = st.session_state.current_page == "intake"
st.markdown(
    f"""<div class="sg-workstation-header">
        <div class="sg-brand"><span class="sg-wordmark">SkinGuard<span>AI</span></span><span class="sg-product">Dermoscopy</span></div>
    </div>
    <div class="sg-page-heading">
        <h1>{'New assessment' if is_intake else 'Assessment results'}</h1>
        <p>{'Enter patient details and select a dermoscopy image.' if is_intake else 'Classification, image attribution and clinical report.'}</p>
    </div>""",
    unsafe_allow_html=True,
)

if not CHECKPOINT_PATH.is_file():
    startup_view.empty()
    st.error(
        f"Checkpoint file not found: `{CHECKPOINT_PATH.resolve()}`. "
        "Please verify the model checkpoint location."
    )
    st.stop()

try:
    model, checkpoint, device = load_medfusionnet_model(str(CHECKPOINT_PATH))
except Exception as error:
    startup_view.empty()
    st.error(f"Failed to load MedFusionNet checkpoint: {error}")
    st.stop()

startup_view.empty()
st.session_state.workspace_ready = True

classes = checkpoint["classes"]
image_size = checkpoint["image_size"]

# ==============================================================================
# VIEW 1: PATIENT INTAKE & LESION ACQUISITION (PAGE 1 - ZERO SCROLL)
# ==============================================================================
if st.session_state.current_page == "intake":
    intake_details, intake_scan = st.columns([1, 1], gap="medium")
    with intake_details:
        with st.container(border=True, key="intake_patient_card"):
            st.markdown('<div class="sg-panel-heading"><h2>Patient details</h2><p>Included in the clinical report.</p></div>', unsafe_allow_html=True)
            patient_name_input = st.text_input("Patient Full Name", value=st.session_state.patient_name, key="input_pname")
            st.session_state.patient_name = patient_name_input
            col_pid, col_age = st.columns([2, 1])
            with col_pid:
                patient_id_input = st.text_input("Patient ID / MRN", value=st.session_state.patient_id, key="input_pid")
                st.session_state.patient_id = patient_id_input
            with col_age:
                patient_age_input = st.text_input("Age", value=st.session_state.patient_age, key="input_page")
                st.session_state.patient_age = patient_age_input
            site_index = ANATOMIC_SITES.index(st.session_state.anatomic_site) if st.session_state.anatomic_site in ANATOMIC_SITES else 0
            anatomic_site_input = st.selectbox("Anatomic Site", ANATOMIC_SITES, index=site_index, key="input_psite")
            st.session_state.anatomic_site = anatomic_site_input

    with intake_scan:
        scan_card = st.container(border=True, key="intake_scan_card")
        with scan_card:
            st.markdown('<div class="sg-panel-heading"><h2>Dermoscopy image</h2><p>Select a JPG or PNG scan for analysis.</p></div>', unsafe_allow_html=True)
            with st.container(key="intake_upload"):
                uploaded_file = st.file_uploader(
                    "Upload Dermoscopy Image",
                    type=("jpg", "jpeg", "png"),
                    label_visibility="collapsed",
                    key="file_uploader_intake",
                )

            if uploaded_file is None:
                st.button("Analyze image", type="primary", use_container_width=True, key="btn_run_analysis", disabled=True)
            else:
                active_image = Image.open(uploaded_file).convert("RGB")
                image_source_label = uploaded_file.name

                # Display-only copy: keep the original image untouched for inference.
                preview_image = active_image.copy()
                preview_image.thumbnail((1600, 1600))
                preview_buffer = io.BytesIO()
                preview_image.save(preview_buffer, format="PNG")
                preview_base64 = base64.b64encode(preview_buffer.getvalue()).decode("ascii")
                st.html(
                    f"""<style>
                    .st-key-intake_upload [data-testid="stFileUploaderDropzone"] {{
                        background-image: url("data:image/png;base64,{preview_base64}");
                        background-size: contain;
                        background-repeat: no-repeat;
                        background-position: center;
                        height: 210px;
                        min-height: 210px;
                        box-sizing: border-box;
                        padding: 12px;
                        justify-content: flex-end;
                        align-items: flex-end;
                    }}
                    .st-key-intake_upload [data-testid="stFileChip"] > :not([data-testid="stFileChipDeleteBtn"]) {{ display: none; }}
                    .st-key-intake_upload [data-testid="stFileChip"] {{ background: transparent; padding: 0; min-width: 0; }}
                    .st-key-intake_upload [data-testid="stFileChips"] {{ width: auto; flex: 0 0 auto; }}
                    .st-key-intake_upload [data-testid="stFileUploaderDropzoneInstructions"] {{ display: none; }}
                    </style>""",
                )

                # Domain & Out-Of-Distribution Quality Gate
                is_dermoscopy, ood_reason, ood_stats = validate_dermoscopy_domain(active_image)
                if st.button("Analyze image", type="primary", use_container_width=True, key="btn_run_analysis"):
                    if not is_dermoscopy:
                        # Rejected input gets a result status without running inference.
                        st.session_state.analysis_results = {
                            "active_image": active_image,
                            "image_source_label": image_source_label,
                            "patient_id": patient_id_input,
                            "patient_name": patient_name_input,
                            "patient_age": patient_age_input,
                            "anatomic_site": anatomic_site_input,
                            "probabilities": None,
                            "cam": None,
                            "fusion_weights": None,
                            "predicted_index": None,
                            "predicted_code": None,
                            "confidence": None,
                            "meta": {},
                            "is_confident": False,
                            "is_ood": True,
                            "ood_reason": ood_reason,
                            "is_cancer": False,
                        }
                        st.session_state.current_page = "results"
                        st.rerun()

                    with st.spinner("Executing MedFusionNet Diagnostic Inference & Multi-Scale Layer-CAM..."):
                        probabilities, cam, fusion_weights, predicted_index = predict_with_multiscale_gradcam(
                            model=model,
                            image=active_image,
                            image_size=image_size,
                            device=device,
                            use_tta=use_tta,
                            confidence_threshold=threshold,
                        )
                        predicted_code = classes[predicted_index]
                        confidence = float(probabilities[predicted_index])
                        meta = CLASS_METADATA.get(predicted_code, {})
                        is_confident = confidence >= threshold
                        is_cancer = meta.get("is_cancer", False)

                        st.session_state.analysis_results = {
                            "active_image": active_image,
                            "image_source_label": image_source_label,
                            "patient_id": patient_id_input,
                            "patient_name": patient_name_input,
                            "patient_age": patient_age_input,
                            "anatomic_site": anatomic_site_input,
                            "probabilities": probabilities,
                            "cam": cam,
                            "fusion_weights": fusion_weights,
                            "predicted_index": predicted_index,
                            "predicted_code": predicted_code,
                            "confidence": confidence,
                            "meta": meta,
                            "is_confident": is_confident,
                            "is_ood": not is_dermoscopy,
                            "is_cancer": is_cancer,
                        }
                        st.session_state.current_page = "results"
                        st.rerun()

# ==============================================================================
# VIEW 2: DIAGNOSTIC RESULTS & CLINICAL CONSULTATION (PAGE 2)
# ==============================================================================
else:
    results = st.session_state.analysis_results
    active_image = results["active_image"]
    image_source_label = results["image_source_label"]
    patient_id = results["patient_id"]
    patient_name = results["patient_name"]
    patient_age = results["patient_age"]
    anatomic_site = results["anatomic_site"]
    probabilities = results["probabilities"]
    cam = results["cam"]
    fusion_weights = results["fusion_weights"]
    predicted_index = results["predicted_index"]
    predicted_code = results["predicted_code"]
    confidence = results["confidence"]
    meta = results["meta"]
    is_confident = results["is_confident"]
    is_ood = results.get("is_ood", False)
    show_assessment = is_confident and not is_ood
    is_cancer = results["is_cancer"]

    # Top Navigation Bar & Action
    if st.button("← Back to patient intake", type="secondary", key="btn_nav_back_top"):
        st.session_state.current_page = "intake"
        st.session_state.analysis_results = None
        st.rerun()

    # Read-Only Telemetry Banner (NO INPUT FIELDS)
    st.markdown(
        f"""
        <div class="sg-card" style="padding: 16px 24px; margin-bottom: 22px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px; font-size: 0.88rem; color: #52606D;">
                <div><strong style="color: #65717B;">Patient:</strong> <span style="color: #253442; font-weight: 600;">{patient_name}</span> (ID: {patient_id}, Age: {patient_age})</div>
                <div><strong style="color: #65717B;">Anatomic Site:</strong> <span style="color: #275D89; font-weight: 600;">{anatomic_site}</span></div>
                <div><strong style="color: #65717B;">Source Scan:</strong> <span style="color: #253442; font-weight: 600;">{image_source_label}</span> ({active_image.width} &times; {active_image.height} px)</div>
                <div><strong style="color: #65717B;">Clinical Gate:</strong> <span style="color: #275D89; font-weight: 700;">Fixed &ge; 90.0% Cutoff</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==============================================================================
    # 8. Primary Clinical Assessment Card (Fixed 90.0% Safety Gate)
    # ==============================================================================
    if not show_assessment:
        status_title = "Out of distribution" if is_ood else "Unknown category"
        status_reason = (
            results.get("ood_reason", "This image did not pass dermoscopy validation. No diagnostic classification is available.")
            if is_ood else
            "The model could not classify this image with sufficient confidence. The result is inconclusive."
        )
        st.markdown(
            f"""<div class="sg-notice" role="status">
                <div class="sg-notice-label">Assessment unavailable</div>
                <div class="sg-notice-title">{status_title}</div>
                <p>{status_reason}</p>
                <p class="sg-notice-action">Review the image or upload a different dermoscopy scan. Seek an in-person dermatological assessment for clinical evaluation.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        if is_cancer:
            st.markdown(
                f"""
                <div class="sg-result-card sg-result-malignant">
                    <span class="sg-result-badge sg-badge-red">{meta.get('category', 'Malignant Neoplasm').upper()} CONFIRMED (&ge; {threshold_percent}.0%)</span>
                    <div class="sg-result-title">Confirmed Diagnosis: {meta.get('name', predicted_code)}</div>
                    <div style="font-size: 1.02rem; line-height: 1.6; margin-top: 10px; color: #703D46;">
                        <strong>Certainty Metric:</strong> <span style="font-size: 1.35rem; font-weight: 800; color: #A32E43;">{confidence:.1%}</span> (Exceeds {threshold_percent}.0% Safety Gate)<br>
                        <strong>ICD-10 Code:</strong> {meta.get('icd10', 'N/A')} &nbsp;|&nbsp; <strong>Clinical Severity:</strong> {meta.get('severity', 'HIGH')}<br><br>
                        <strong>Pathological Characteristics:</strong> {meta.get('description', '')}<br><br>
                        <strong>Recommended Clinical Protocol:</strong> {meta.get('protocol', '')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="sg-result-card sg-result-benign">
                    <span class="sg-result-badge sg-badge-green">BENIGN LESION CONFIRMED (&ge; {threshold_percent}.0%)</span>
                    <div class="sg-result-title">Confirmed Diagnosis: {meta.get('name', predicted_code)}</div>
                    <div style="font-size: 1.02rem; line-height: 1.6; margin-top: 10px; color: #315D4D;">
                        <strong>Certainty Metric:</strong> <span style="font-size: 1.35rem; font-weight: 800; color: #246B53;">{confidence:.1%}</span> (Exceeds {threshold_percent}.0% Safety Gate)<br>
                        <strong>ICD-10 Code:</strong> {meta.get('icd10', 'N/A')} &nbsp;|&nbsp; <strong>Clinical Status:</strong> Harmless / Non-Malignant<br><br>
                        <strong>Pathological Characteristics:</strong> {meta.get('description', '')}<br><br>
                        <strong>Clinical Recommendation:</strong> {meta.get('protocol', '')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if show_assessment:
        # ==============================================================================
        # 9. Explainable AI (XAI) Studio: Multi-Scale Layer-CAM
        # ==============================================================================
        resized_display = active_image.resize((image_size, image_size))
        image_array = np.asarray(resized_display, dtype=np.float32) / 255.0

        col_hdr_left, col_hdr_right = st.columns([3.2, 1.3])
        with col_hdr_left:
            st.markdown(
                """
                <div class="sg-card" style="margin-bottom: 0px; padding-bottom: 14px;">
                    <div class="sg-card-title">Image attribution</div>
                    <div class="sg-card-caption" style="margin-bottom: 0px;">
                        High-precision element-wise gradient attribution combining Stage 5 (fine boundaries & reticular pigment mesh) and Stage 7 (macro-lesion context). Eliminates gradient cancellation artifacts and delineates exact pathological margins.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_hdr_right:
            selected_cmap = st.selectbox(
                "Heatmap Palette",
                ["turbo (High Precision)", "jet (Classic)", "magma (Thermal)"],
                index=0,
                key="results_cmap_palette",
            )
            cmap_name = selected_cmap.split()[0].lower()

        heatmap_colored = plt.get_cmap(cmap_name)(cam)[..., :3]
        alpha_mask = np.expand_dims(cam * 0.65, axis=-1)
        overlay_display = np.clip((1.0 - alpha_mask) * image_array + alpha_mask * heatmap_colored, 0.0, 1.0)

        if not is_confident:
            st.markdown(
                """
                <div style="background: rgba(217, 119, 6, 0.12); border: 1px solid #D97706; border-radius: 12px; padding: 12px 18px; color: #70511E; font-size: 0.88rem; line-height: 1.5; margin: 12px 0 16px 0;">
                    <strong>Exploratory Saliency Active (Borderline Case &lt; 90.0%):</strong> While automated confirmation requires &ge;90.0% certainty, the high-resolution Multi-Scale Layer-CAM below highlights the atypical textural margins and pigment network that drew neural attention to assist with biopsy planning.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        col_xai_1, col_xai_2, col_xai_3 = st.columns(3)
        with col_xai_1:
            st.image(resized_display, caption="1. Normalized Dermoscopy View", width="stretch")
        with col_xai_2:
            st.image(heatmap_colored, caption=f"2. Multi-Scale Layer-CAM Saliency ({cmap_name.upper()})", width="stretch")
        with col_xai_3:
            st.image(overlay_display, caption="3. Pathological Heatmap Overlay", width="stretch")

        # ==============================================================================
        # 10. Differential Diagnosis: 7-Class Probability Distribution
        # ==============================================================================
        st.markdown(
            f"""
            <div class="sg-card">
                <div class="sg-card-title">Differential probabilities</div>
                <div class="sg-card-caption">
                    Full posterior probability distribution across all 7 diagnostic classes. The vertical red indicator marks the <strong>{threshold_percent}.0% Clinical Safety Threshold</strong>.
                </div>
            """,
            unsafe_allow_html=True,
        )

        if not is_confident:
            st.markdown(
                """
                <div style="background: rgba(217, 119, 6, 0.10); border: 1px solid rgba(217, 119, 6, 0.4); border-radius: 10px; padding: 10px 16px; margin-bottom: 14px; color: #70511E; font-size: 0.86rem;">
                    <strong>Safety Gate Status:</strong> Peak confidence is below the 90.0% threshold. The differential breakdown below provides full clinical transparency across competing diagnoses.
                </div>
                """,
                unsafe_allow_html=True,
            )

        sorted_indices = np.argsort(probabilities)[::-1]

        for idx in sorted_indices:
            code = classes[idx]
            p_val = float(probabilities[idx])
            c_meta = CLASS_METADATA.get(code, {})
            c_name = c_meta.get("name", code)
            c_color = c_meta.get("color", "#0284C7")
            c_type = c_meta.get("category", "Benign")

            st.markdown(
                f"""
                <div class="sg-prob-row">
                    <div class="sg-prob-label">
                        <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: {c_color}; margin-right: 6px;"></span>
                        {c_name}
                        <span style="font-size: 0.72rem; color: #65717B; font-weight: normal; margin-left: 4px;">({c_type})</span>
                    </div>
                    <div class="sg-prob-bar-container">
                        <div class="sg-prob-bar-fill" style="width: {p_val * 100:.1f}%; background: {c_color};"></div>
                        <div style="position: absolute; left: {threshold_percent}%; top: 0; bottom: 0; width: 2px; background: #EF4444; z-index: 2;"></div>
                    </div>
                    <div class="sg-prob-value" style="color: {c_color};">{p_val:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # ==============================================================================
        # 11. Clinical Assessment Report & Physician Export Hub
        # ==============================================================================
        st.markdown(
            """
            <div class="sg-card">
                <div class="sg-card-title">Clinical report</div>
                <div class="sg-card-caption">
                    Generate and export an official clinical diagnostic consultation report for electronic health record (EMR/EHR) archiving, physician sign-off, or multidisciplinary oncology review.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        clinical_notes_input = st.text_area(
            "Physician Clinical Observations & Case Notes (Optional)",
            placeholder="Enter clinical examination notes, patient lesion history, anatomical progression, or recommended surgical timeline...",
            height=80,
            key="results_clinical_notes",
        )

        # Generate PDF & Text report payloads
        pdf_bytes = generate_clinical_pdf_report(
            patient_id=patient_id,
            patient_name=patient_name,
            patient_age=patient_age,
            anatomic_site=anatomic_site,
            clinical_notes=clinical_notes_input,
            image_name=image_source_label,
            active_image=resized_display,
            cam=cam,
            overlay_display=overlay_display,
            predicted_code=predicted_code,
            confidence=confidence,
            is_confident=is_confident,
            meta=meta,
            probabilities=probabilities,
            classes=classes,
            fusion_weights=fusion_weights,
            threshold_percent=threshold_percent,
        )

        txt_report = generate_clinical_text_report(
            patient_id=patient_id,
            patient_name=patient_name,
            patient_age=patient_age,
            anatomic_site=anatomic_site,
            clinical_notes=clinical_notes_input,
            image_name=image_source_label,
            predicted_code=predicted_code,
            confidence=confidence,
            is_confident=is_confident,
            meta=meta,
            probabilities=probabilities,
            classes=classes,
            fusion_weights=fusion_weights,
            threshold_percent=threshold_percent,
        )

        clean_pid = re.sub(r"[^\w\-]", "_", patient_id.strip()) or "PT-UNASSIGNED"
        report_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename_pdf = f"SkinGuardAI_Report_{clean_pid}_{report_timestamp}.pdf"
        filename_txt = f"SkinGuardAI_Report_{clean_pid}_{report_timestamp}.txt"

        # Archive locally to reports/ directory on disk
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = reports_dir / filename_pdf
        pdf_path.write_bytes(pdf_bytes)
        txt_path = reports_dir / filename_txt
        txt_path.write_text(txt_report, encoding="utf-8")

        # Also save directly to static/reports/ for rock-solid HTTP downloads with exact filename
        static_reports_dir = Path("static/reports")
        static_reports_dir.mkdir(parents=True, exist_ok=True)
        static_pdf_path = static_reports_dir / filename_pdf
        static_pdf_path.write_bytes(pdf_bytes)
        static_txt_path = static_reports_dir / filename_txt
        static_txt_path.write_text(txt_report, encoding="utf-8")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.markdown(
                f"""
                <a href="/app/static/reports/{filename_pdf}" download="{filename_pdf}" class="sg-btn-download" target="_blank">
                    Download PDF report
                </a>
                """,
                unsafe_allow_html=True,
            )
        with col_btn2:
            st.markdown(
                f"""
                <a href="/app/static/reports/{filename_txt}" download="{filename_txt}" class="sg-btn-download sg-btn-secondary" target="_blank">
                    Download text summary
                </a>
                """,
                unsafe_allow_html=True,
            )

    # Bottom navigation to return to intake
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    if st.button("◀ Return to Patient Intake / Upload New Scan", type="secondary", key="btn_nav_back_bottom", use_container_width=True):
        st.session_state.current_page = "intake"
        st.session_state.analysis_results = None
        st.rerun()



if __name__ == "__main__":
    import sys
    try:
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", __file__, "--server.port", "8501", "--server.headless", "true"]
        sys.exit(stcli.main())
    except Exception:
        pass

