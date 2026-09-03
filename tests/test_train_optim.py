import numpy as np
import pytest
import torch

from aion_hats.train.config import OptimConfig, load_config
from aion_hats.train.optim import (
    Schedules,
    cosine_schedule,
    create_optimizer,
    epochs_from_tokens,
    parameter_groups,
    scaled_lr,
    steps_from_tokens,
)


def test_cosine_schedule_shape_and_endpoints():
    s = cosine_schedule(1.0, 0.1, total_steps=10, warmup_steps=3)
    assert len(s) == 10 and s[0] == 0.0 and s[2] == pytest.approx(1.0)
    assert s[3] == pytest.approx(1.0) and s[-1] > 0.1 and np.all(np.diff(s[3:]) <= 0)
    assert len(cosine_schedule(1.0, 0.0, 0)) == 0
    assert cosine_schedule(1.0, 0.0, 4, warmup_steps=10).tolist() == pytest.approx([0, 1 / 3, 2 / 3, 1.0])
    sched = Schedules.build(OptimConfig(weight_decay=0.05, weight_decay_end=0.01), 1e-3, 1e-5, 8, 2)
    assert len(sched.lr) == len(sched.wd) == 8 and sched.wd[0] == 0.05 and sched.wd[-1] > 0.01


def test_token_arithmetic():
    assert scaled_lr(1e-4, 256) == 1e-4 and scaled_lr(1e-4, 512) == 2e-4
    assert steps_from_tokens(1e-3, 384, 256) == int(np.ceil(1e6 / (384 * 256)))
    assert epochs_from_tokens(1e-3, 384, steps_per_epoch=5, global_batch_size=256) == 3


def test_parameter_groups_and_optimizer():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    model[1].weight.requires_grad_(False)
    groups = parameter_groups(model, weight_decay=0.1)
    by_wd = {g["weight_decay"]: g for g in groups}
    assert set(by_wd) == {0.1, 0.0} and all(g["lr_scale"] == 1.0 for g in groups)
    assert len(by_wd[0.1]["params"]) == 1 and len(by_wd[0.0]["params"]) == 2  # linear bias + norm bias
    opt = create_optimizer(model, OptimConfig(blr=1.0, betas=(0.9, 0.99)), lr=1e-3)
    assert isinstance(opt, torch.optim.AdamW) and opt.param_groups[0]["betas"] == (0.9, 0.99)


def test_config_loading_and_overrides(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "model: {preset: small}\n"
        "data:\n  datasets:\n    - {name: ls, catalog: /x, modalities: [tok_image, tok_flux_g], out_domains: [tok_image]}\n"
        "schedule: {total_tokens_b: 2, warmup_tokens_b: 0.1}\n"
        "run: {batch_size: 8, dtype: bf16, wandb: {project: p}}\n"
    )
    cfg = load_config(path, ["optim.blr=3e-4", "run.output_dir=/tmp/o", "model.overrides={dim: 32}"])
    assert cfg.model.preset == "small" and cfg.model.overrides == {"dim": 32}
    assert cfg.model.domains_in == ["tok_flux_g", "tok_image"] and cfg.model.domains_out == ["tok_image"]
    assert cfg.optim.blr == 3e-4 and cfg.run.dtype == "bfloat16" and cfg.run.wandb.project == "p"
    assert cfg.data.min_input_tokens == 256 and cfg.all_domains == ["tok_flux_g", "tok_image"]
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path, ["run.nope=1"])
    with pytest.raises(ValueError, match="exactly one"):
        load_config(path, ["schedule.epochs=1"])
    with pytest.raises(ValueError, match="not in modalities"):
        load_config(None, ["data.datasets=[{catalog: /x, modalities: [tok_image], in_domains: [tok_z]}]"])
    with pytest.raises(NotImplementedError):
        load_config(None, ["data.datasets=[{name: a, catalog: /x}, {name: b, catalog: /y}]"])
