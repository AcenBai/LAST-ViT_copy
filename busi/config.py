import os
import sys
import torch

from detectron2.config import LazyCall as L
from detectron2.model_zoo import get_config
from detectron2.solver import WarmupParamScheduler
from detectron2.solver.build import get_default_optimizer_params
from fvcore.common.param_scheduler import CosineParamScheduler
from omegaconf import OmegaConf

_CUR_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_CUR_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from busi.pipeline import (
    BUSIClassificationNet,
    BUSIManifestDataset,
    BUSIMetrics,
    build_busi_vit,
    build_data_loader,
)


dataloader = OmegaConf.create()
dataloader.train = L(build_data_loader)(
    dataset=L(BUSIManifestDataset)(
        manifest_path="./datasets/busi_splits/train.json",
        image_size=224,
    ),
    batch_size=8,
    num_workers=8,
    training=True,
)

dataloader.test = L(build_data_loader)(
    dataset=L(BUSIManifestDataset)(
        manifest_path="./datasets/busi_splits/val.json",
        image_size=224,
    ),
    batch_size=32,
    num_workers=8,
    training=False,
)

dataloader.evaluator = L(BUSIMetrics)(
    image_size=224,
    patch_size=16,
    metric_backbone_kind="vit",
    resnet_fg_threshold=0.05,
)

model = L(BUSIClassificationNet)(
    model=L(build_busi_vit)(
        num_classes=3,
        pretrained=True,
        use_dense=False,
        image_size=224,
        backbone_kind="vit",
        dense_score_mode="original",
        resnet_feature_layer="layer4",
    )
)

optimizer = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(),
    lr=5e-5,
    weight_decay=0.05,
    betas=(0.9, 0.999),
)

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(CosineParamScheduler)(start_value=1.0, end_value=0.01),
    warmup_length=0.05,
    warmup_factor=0.1,
)

train = get_config("common/train.py").train
train.max_iter = 5000
train.eval_period = 200
train.log_period = 50
train.output_dir = "output/busi_vit_b16_512"
train.init_checkpoint = ""
train.checkpointer["period"] = 200

