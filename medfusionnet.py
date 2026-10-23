"""MedFusionNet: ConvNeXt-Tiny and Vision Transformer with adaptive attention fusion."""

from __future__ import annotations

from typing import Literal
from pathlib import Path

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    Swin_T_Weights,
    convnext_tiny,
    swin_t,
)


class CompactVisionTransformer(nn.Module):
    """A 768-dim ViT branch sized to keep the full model near the paper's 45M parameters."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 2,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        self.patch_embedding = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        patch_count = (image_size // patch_size) ** 2
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, patch_count + 1, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.patch_embedding.weight, std=0.02)
        if self.patch_embedding.bias is not None:
            nn.init.zeros_(self.patch_embedding.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embedding(images).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((class_token, tokens), dim=1)
        tokens = self.dropout(tokens + self.position_embedding)
        tokens = self.encoder(tokens)
        return self.norm(tokens[:, 0])


class MedFusionNet(nn.Module):
    """Parallel ConvNeXt and Vision Transformer with dynamic attention fusion."""

    def __init__(
        self,
        num_classes: int = 7,
        image_size: int = 224,
        vit_type: Literal["swin", "compact"] = "swin",
        vit_depth: int = 2,
        fusion_type: Literal["gate", "attention"] = "gate",
        pretrained: bool = True,
        pretrained_convnext: bool | None = None,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.vit_type = vit_type
        self.fusion_type = fusion_type
        if pretrained_convnext is not None:
            pretrained = pretrained_convnext

        # Branch 1: ConvNeXt-Tiny (extracts fine local borders and textures)
        convnext_weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = convnext_tiny(weights=convnext_weights)
        self.convnext_features = convnext.features
        self.convnext_pool = convnext.avgpool
        self.convnext_norm = convnext.classifier[0]
        self.convnext_flatten = convnext.classifier[1]

        # Branch 2: Vision Transformer (captures global lesion context)
        if vit_type == "swin":
            swin_weights = Swin_T_Weights.DEFAULT if pretrained else None
            swin = swin_t(weights=swin_weights)
            self.swin_features = swin.features
            self.swin_norm = swin.norm
            self.swin_permute = swin.permute
            self.swin_avgpool = swin.avgpool
            self.swin_flatten = swin.flatten
            self.vit = None
        else:
            self.vit = CompactVisionTransformer(image_size=image_size, depth=vit_depth)
            self.swin_features = None

        feature_dim = 768
        if fusion_type == "attention":
            self.fusion_attention = nn.Sequential(
                nn.LayerNorm(feature_dim * 2),
                nn.Linear(feature_dim * 2, 256),
                nn.GELU(),
                nn.Linear(256, 2),
            )
            self.fusion_gate = None
            self.fusion_proj = None
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )
        else:
            self.fusion_attention = None
            self.fusion_gate = nn.Sequential(
                nn.LayerNorm(feature_dim * 2),
                nn.Linear(feature_dim * 2, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, feature_dim * 2),
                nn.Sigmoid(),
            )
            self.fusion_proj = nn.Sequential(
                nn.LayerNorm(feature_dim * 2),
                nn.Linear(feature_dim * 2, feature_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(256, num_classes),
            )
        self.last_fusion_weights: torch.Tensor | None = None

    def extract_convnext(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.convnext_features(images)
        features = self.convnext_pool(feature_map)
        features = self.convnext_norm(features)
        return self.convnext_flatten(features), feature_map

    def extract_vit(self, images: torch.Tensor) -> torch.Tensor:
        if self.vit_type == "swin":
            features = self.swin_features(images)
            features = self.swin_norm(features)
            features = self.swin_permute(features)
            features = self.swin_avgpool(features)
            return self.swin_flatten(features)
        return self.vit(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        cnn_features, _ = self.extract_convnext(images)
        vit_features = self.extract_vit(images)
        concat = torch.cat((cnn_features, vit_features), dim=1)

        if self.fusion_type == "attention":
            attention_logits = self.fusion_attention(concat)
            weights = torch.softmax(attention_logits, dim=1)
            self.last_fusion_weights = weights.detach()
            fused = weights[:, 0:1] * cnn_features + weights[:, 1:2] * vit_features
            return self.classifier(fused)
        else:
            gate = self.fusion_gate(concat)
            gated = concat * gate
            fused = self.fusion_proj(gated) + cnn_features
            self.last_fusion_weights = torch.stack(
                [gate[:, :768].mean(dim=1), gate[:, 768:].mean(dim=1)], dim=1
            ).detach()
            return self.classifier(fused)

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: str | Path, device: torch.device | str = "cpu"
    ) -> tuple[MedFusionNet, dict]:
        # Map the checkpoint on CPU; load_state_dict copies/casts into the model.
        # Avoid a second full float32 state dictionary during startup.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
        state_dict = checkpoint["model_state_dict"]
        num_classes = len(checkpoint.get("classes", ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]))
        image_size = checkpoint.get("image_size", 224)

        if "fusion_attention.0.weight" in state_dict:
            fusion_type = "attention"
            vit_type = "compact"
        else:
            fusion_type = "gate"
            vit_type = checkpoint.get("vit_type", "swin")

        vit_depth = checkpoint.get("vit_depth", 2)

        model = cls(
            num_classes=num_classes,
            image_size=image_size,
            vit_type=vit_type,
            vit_depth=vit_depth,
            fusion_type=fusion_type,
            pretrained=False,
        ).to(device)

        model.load_state_dict(state_dict)
        model.eval()

        # Release checkpoint tensors; actual RSS also includes runtime/activations.
        del state_dict
        import gc
        gc.collect()

        metadata = {
            "classes": checkpoint.get("classes", ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]),
            "image_size": image_size,
            "vit_depth": vit_depth,
            "vit_type": vit_type,
            "fusion_type": fusion_type,
            "best_val_macro_f1": checkpoint.get("best_val_macro_f1", None),
            "epoch": checkpoint.get("epoch", None),
        }
        del checkpoint
        gc.collect()

        return model, metadata

    def get_parameter_groups(
        self, backbone_lr: float, head_lr: float, weight_decay: float
    ) -> list[dict]:
        backbone_params = []
        head_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "convnext_features" in name or "swin_features" in name or "vit" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        return [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay * 0.5},
        ]


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

