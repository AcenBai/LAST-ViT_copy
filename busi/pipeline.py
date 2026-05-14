import math
import numpy as np
import os
import torch

from dataclasses import dataclass
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.models import ResNet50_Weights, resnet50, vit_b_16
from torchvision.transforms import functional as TF
from typing import Dict, Sequence, Tuple

from detectron2.data.samplers import InferenceSampler, TrainingSampler
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm

from busi.data import load_manifest


def build_data_loader(dataset, batch_size, num_workers, training=True):
    return torch.utils.data.DataLoader(
        dataset,
        sampler=(TrainingSampler if training else InferenceSampler)(len(dataset)),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=training,
    )


def _merge_mask(mask_paths: Sequence[str], size: Tuple[int, int]) -> Image.Image:
    merged = np.zeros((size[1], size[0]), dtype=np.uint8)
    for mp in mask_paths:
        if not os.path.exists(mp):
            continue
        m = Image.open(mp).convert("L")
        arr = np.array(m, dtype=np.uint8)
        merged = np.maximum(merged, (arr > 0).astype(np.uint8) * 255)
    return Image.fromarray(merged, mode="L")


@dataclass
class ResizeNormCfg:
    image_size: int = 224
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class BUSIManifestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest_path: str,
        image_size: int = 224,
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.rows = load_manifest(manifest_path)
        self.cfg = ResizeNormCfg(
            image_size=image_size,
            mean=normalize_mean,
            std=normalize_std,
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        orig_w, orig_h = image.size
        mask = _merge_mask(row["mask_paths"], image.size)

        image = TF.resize(image, [self.cfg.image_size, self.cfg.image_size], antialias=True)
        mask = TF.resize(mask, [self.cfg.image_size, self.cfg.image_size], interpolation=Image.NEAREST)

        image_t = TF.to_tensor(image)
        image_t = TF.normalize(image_t, self.cfg.mean, self.cfg.std)
        mask_t = (TF.to_tensor(mask) > 0).float().squeeze(0)

        bbox = torch.zeros(4, dtype=torch.long)
        ys, xs = torch.where(mask_t > 0)
        if len(xs) > 0 and len(ys) > 0:
            bbox[0] = xs.min().long()
            bbox[1] = ys.min().long()
            bbox[2] = xs.max().long()
            bbox[3] = ys.max().long()

        return {
            "image": image_t,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "mask": mask_t,
            "bbox": bbox,
            "id": row["id"],
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
        }


class ViTWithPatchScores(VisionTransformer):
    def forward(self, x: torch.Tensor):
        x = self._process_input(x)
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        cls_token = x[:, 0]
        tokens = x[:, 1:]
        patch_scores = F.cosine_similarity(
            tokens,
            cls_token.unsqueeze(1).expand_as(tokens),
            dim=-1,
        )
        logits = self.heads(cls_token)
        return logits, patch_scores


class DenseViTWithPatchScores(VisionTransformer):
    def __init__(self, *args, score_mode: str = "original", **kwargs):
        super().__init__(*args, **kwargs)
        if score_mode not in {"original", "inverse"}:
            raise ValueError(f"Unsupported score_mode={score_mode!r}")
        self.score_mode = score_mode
        self.cached_kernel = None

    def gaussian_kernel_1d(self, kernel_size, sigma):
        kernel = torch.exp(
            -0.5 * (torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1).float() / sigma) ** 2
        )
        kernel = kernel / torch.max(kernel)
        return kernel

    def forward(self, x: torch.Tensor):
        x = self._process_input(x)
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        cls_token = x[:, 0]
        x_detach = x[:, 1:]

        if self.cached_kernel is None:
            self.cached_kernel = (
                self.gaussian_kernel_1d(x_detach.shape[-1], x_detach.shape[-1] ** 0.5)
                .to(x.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
        x_fft = torch.fft.fft(x_detach, dim=-1)
        x_fft = torch.fft.fftshift(x_fft, dim=-1)
        x_fft = x_fft * self.cached_kernel
        x_fft = torch.fft.ifftshift(x_fft, dim=-1)
        x_recon = torch.fft.ifft(x_fft, dim=-1).real

        diff = x_detach / (torch.abs(x_recon - x_detach) + 1e-6)
        if self.score_mode == "inverse":
            token_select_scores = 1.0 / (torch.abs(diff) + 1e-6)
        else:
            token_select_scores = diff

        _, top_patch = torch.topk(token_select_scores, k=1, dim=1, largest=True)
        selected = torch.gather(x_detach, 1, top_patch)
        pooled_token = torch.mean(selected, dim=1)
        logits = self.heads(pooled_token)
        cosine_patch_scores = F.cosine_similarity(
            x_detach,
            pooled_token.unsqueeze(1).expand_as(x_detach),
            dim=-1,
        )
        return logits, cosine_patch_scores


class ResNet50WithPatchScores(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        feature_layer: str = "layer4",
    ):
        super().__init__()
        if feature_layer not in {"layer3", "layer4"}:
            raise ValueError(f"Unsupported feature_layer={feature_layer!r}")
        self.feature_layer = feature_layer

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        feat_l3 = self.backbone.layer3(x)
        feat_l4 = self.backbone.layer4(feat_l3)

        feat = feat_l4 if self.feature_layer == "layer4" else feat_l3

        pooled = self.backbone.avgpool(feat_l4)
        pooled = torch.flatten(pooled, 1)
        logits = self.backbone.fc(pooled)

        q_gap = feat.mean(dim=(2, 3))
        patch_tokens = feat.flatten(2).transpose(1, 2)
        patch_scores = F.cosine_similarity(
            patch_tokens,
            q_gap.unsqueeze(1).expand_as(patch_tokens),
            dim=-1,
        )
        return logits, patch_scores


def _load_vit_weights_with_pos_resize(model: VisionTransformer, state_dict: Dict[str, torch.Tensor]) -> None:
    pos_key = "encoder.pos_embedding"
    if pos_key in state_dict and pos_key in model.state_dict():
        src_pos = state_dict[pos_key]
        dst_pos = model.state_dict()[pos_key]
        if src_pos.shape != dst_pos.shape:
            cls_src = src_pos[:, :1, :]
            patch_src = src_pos[:, 1:, :]
            num_src = patch_src.shape[1]
            dim = patch_src.shape[2]
            src_hw = int(math.sqrt(num_src))
            dst_hw = int(math.sqrt(dst_pos.shape[1] - 1))
            if src_hw * src_hw == num_src and dst_hw * dst_hw == (dst_pos.shape[1] - 1):
                patch_src = patch_src.reshape(1, src_hw, src_hw, dim).permute(0, 3, 1, 2)
                patch_src = F.interpolate(
                    patch_src,
                    size=(dst_hw, dst_hw),
                    mode="bicubic",
                    align_corners=False,
                )
                patch_src = patch_src.permute(0, 2, 3, 1).reshape(1, dst_hw * dst_hw, dim)
                state_dict[pos_key] = torch.cat([cls_src, patch_src], dim=1)
            else:
                state_dict.pop(pos_key, None)
    model.load_state_dict(state_dict, strict=False)


def build_busi_vit(
    num_classes=3,
    pretrained=True,
    use_dense=False,
    image_size=224,
    backbone_kind: str | None = None,
    dense_score_mode: str = "original",
    resnet_feature_layer: str = "layer4",
):
    if backbone_kind is None:
        backbone_kind = "dense" if use_dense else "vit"

    if backbone_kind == "resnet50":
        return ResNet50WithPatchScores(
            num_classes=num_classes,
            pretrained=pretrained,
            feature_layer=resnet_feature_layer,
        )

    from torchvision.models import ViT_B_16_Weights

    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    if backbone_kind in {"dense", "dense_inv"}:
        model = DenseViTWithPatchScores(
            image_size=image_size,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=1000,
            score_mode="inverse" if backbone_kind == "dense_inv" else dense_score_mode,
        )
        if weights is not None:
            base_model = vit_b_16(weights=weights)
            _load_vit_weights_with_pos_resize(model, base_model.state_dict())
    elif backbone_kind == "vit":
        model = ViTWithPatchScores(
            image_size=image_size,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=1000,
        )
        if weights is not None:
            base_model = vit_b_16(weights=weights)
            _load_vit_weights_with_pos_resize(model, base_model.state_dict())
    else:
        raise ValueError(f"Unsupported backbone_kind={backbone_kind!r}")

    in_dim = model.heads.head.in_features
    model.heads.head = nn.Linear(in_dim, num_classes)
    return model


class BUSIClassificationNet(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @property
    def device(self):
        return list(self.model.parameters())[0].device

    def forward(self, inputs):
        image = inputs["image"].to(self.device)
        label = inputs["label"].to(self.device)
        logits, patch_scores = self.model(image)
        if self.training:
            return F.cross_entropy(logits, label)
        return {"logits": logits, "patch_scores": patch_scores}


def _top_patch_center(
    top_idx: torch.Tensor,
    image_size: int,
    patches_per_side: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if patches_per_side <= 0:
        raise ValueError(f"Invalid patches_per_side={patches_per_side}")
    patch_size = image_size // patches_per_side
    y_idx = top_idx // patches_per_side
    x_idx = top_idx % patches_per_side
    cx = x_idx * patch_size + (patch_size // 2)
    cy = y_idx * patch_size + (patch_size // 2)
    return cx, cy


class BUSIMetrics(DatasetEvaluator):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        metric_backbone_kind: str = "vit",
        resnet_fg_threshold: float = 0.05,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.metric_backbone_kind = metric_backbone_kind
        self.resnet_fg_threshold = float(resnet_fg_threshold)

    def reset(self):
        self.probs = []
        self.preds = []
        self.labels = []
        self.pim_hits = []
        self.pib_hits = []

    def process(self, inputs, outputs):
        logits = outputs["logits"]
        patch_scores = outputs["patch_scores"]
        probs = torch.softmax(logits, dim=1).detach().cpu()
        pred = probs.argmax(dim=1)
        label = inputs["label"].detach().cpu().long()
        mask = inputs["mask"].detach().cpu()
        bbox = inputs["bbox"].detach().cpu()

        top_idx = patch_scores.detach().cpu().argmax(dim=1)
        num_patches = int(patch_scores.shape[1])
        patches_per_side = int(round(math.sqrt(num_patches)))
        if patches_per_side * patches_per_side != num_patches:
            raise ValueError(f"Expected square patch grid, got num_patches={num_patches}")
        cx, cy = _top_patch_center(top_idx, self.image_size, patches_per_side)

        for i in range(mask.shape[0]):
            if int(label[i].item()) == 2:
                continue
            cur_mask = mask[i]
            has_mask = bool((cur_mask > 0).any().item())
            if has_mask:
                if self.metric_backbone_kind == "resnet50":
                    num_patches = int(patch_scores.shape[1])
                    grid_side = int(round(math.sqrt(num_patches)))
                    if grid_side * grid_side != num_patches:
                        raise ValueError(f"Expected square patch grid, got num_patches={num_patches}")
                    mask_ratio = F.interpolate(
                        cur_mask.float().unsqueeze(0).unsqueeze(0),
                        size=(grid_side, grid_side),
                        mode="area",
                    ).squeeze(0).squeeze(0)
                    fg_mask = mask_ratio > self.resnet_fg_threshold

                    top_flat = int(top_idx[i].item())
                    top_y = top_flat // grid_side
                    top_x = top_flat % grid_side
                    in_mask = float(fg_mask[top_y, top_x].item())
                    self.pim_hits.append(in_mask)

                    ys, xs = torch.where(fg_mask)
                    if ys.numel() > 0:
                        x0 = int(xs.min().item())
                        y0 = int(ys.min().item())
                        x1 = int(xs.max().item())
                        y1 = int(ys.max().item())
                        in_bbox = float((top_x >= x0) and (top_x <= x1) and (top_y >= y0) and (top_y <= y1))
                        self.pib_hits.append(in_bbox)
                else:
                    x = int(torch.clamp(cx[i], 0, self.image_size - 1).item())
                    y = int(torch.clamp(cy[i], 0, self.image_size - 1).item())
                    in_mask = float(cur_mask[y, x] > 0)
                    self.pim_hits.append(in_mask)

                    x0, y0, x1, y1 = [int(v.item()) for v in bbox[i]]
                    in_bbox = float((x >= x0) and (x <= x1) and (y >= y0) and (y <= y1))
                    self.pib_hits.append(in_bbox)

        self.probs.append(probs)
        self.preds.append(pred)
        self.labels.append(label)

    def evaluate(self):
        if len(self.labels) == 0:
            return {
                "acc": 0.0,
                "auc": float("nan"),
                "pim": float("nan"),
                "pib": float("nan"),
                "pim_count": 0,
            }

        probs = torch.cat(self.probs, dim=0).numpy()
        preds = torch.cat(self.preds, dim=0).numpy()
        labels = torch.cat(self.labels, dim=0).numpy()
        pim_hits = np.array(self.pim_hits, dtype=np.float32)
        pib_hits = np.array(self.pib_hits, dtype=np.float32)

        gathered = comm.gather(
            {"probs": probs, "preds": preds, "labels": labels, "pim_hits": pim_hits, "pib_hits": pib_hits},
            dst=0,
        )
        if not comm.is_main_process():
            return {}

        probs = np.concatenate([x["probs"] for x in gathered], axis=0)
        preds = np.concatenate([x["preds"] for x in gathered], axis=0)
        labels = np.concatenate([x["labels"] for x in gathered], axis=0)
        pim_hits = np.concatenate([x["pim_hits"] for x in gathered], axis=0)
        pib_hits = np.concatenate([x["pib_hits"] for x in gathered], axis=0)

        acc = float((preds == labels).mean())
        try:
            from sklearn.metrics import roc_auc_score

            num_classes = probs.shape[1]
            labels_one_hot = np.eye(num_classes, dtype=np.float32)[labels]
            auc = float(
                roc_auc_score(labels_one_hot, probs, average="macro", multi_class="ovr")
            )
        except Exception:
            auc = float("nan")

        pim = float(pim_hits.mean()) if pim_hits.size > 0 else float("nan")
        pib = float(pib_hits.mean()) if pib_hits.size > 0 else float("nan")
        return {
            "acc": acc,
            "auc": auc,
            "pim": pim,
            "pib": pib,
            "pim_count": int(pim_hits.size),
        }

