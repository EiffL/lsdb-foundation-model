import json

import pytest
import torch
from conftest import TINY_MODEL, tiny_train_config

pytest.importorskip("lsdb")

from aion_hats.train import (
    Trainer,
    build_model,
    export_pretrained,
    model_config,
    train,
)
from aion_hats.train.checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from aion_hats.train.config import ModelConfig


def test_model_config_matches_aion_base_layout():
    cfg = model_config("base", ["tok_image"], ["tok_image", "tok_z"])
    assert cfg["dim"] == 768 and cfg["act_layer"] == "SiLU" and cfg["norm_bias"] is False
    assert cfg["domains_in"] == ["tok_image"] and cfg["domains_out"] == ["tok_image", "tok_z"]
    assert "num_register_tokens" not in cfg
    assert model_config("tiny", ["tok_image"], ["tok_image"], {"dim": 64})["dim"] == 64
    with pytest.raises(ValueError, match="unknown preset"):
        model_config("huge", [], [])


def test_build_model_and_export_roundtrip(tmp_path):
    model = build_model(ModelConfig(preset="tiny", overrides=TINY_MODEL), ["tok_image"], ["tok_image"])
    out = export_pretrained(model, tmp_path / "final")
    assert (out / "config.json").exists() and (out / "model.safetensors").exists()
    from aion import AION
    from aion.fourm.fm import FM

    reloaded = FM.from_pretrained(str(out))
    for (name, a), (_, b) in zip(model.state_dict().items(), reloaded.state_dict().items()):
        assert torch.equal(a, b), name
    assert set(AION.from_pretrained(str(out)).encoder_embeddings) == {"tok_image"}
    # the exported directory is a valid init_from
    again = build_model(ModelConfig(init_from=str(out)), ["tok_image"], ["tok_image"])
    assert torch.equal(again.mask_token, model.mask_token)
    with pytest.raises(ValueError, match="no embeddings"):
        build_model(ModelConfig(init_from=str(out)), ["tok_image", "tok_z"], ["tok_image"])


def test_checkpoint_roundtrip(tmp_path):
    model = build_model(ModelConfig(preset="tiny", overrides=TINY_MODEL), ["tok_image"], ["tok_image"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.mask_token.grad = torch.ones_like(model.mask_token)
    opt.step()
    path = save_checkpoint(tmp_path / "checkpoint-3.pth", model, opt, 3, {"a": 1})
    assert latest_checkpoint(tmp_path) == path
    fresh = build_model(ModelConfig(preset="tiny", overrides=TINY_MODEL), ["tok_image"], ["tok_image"])
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    state = load_checkpoint(path, fresh, fresh_opt)
    assert state == {"epoch": 3, "args": {"a": 1}}
    assert torch.equal(fresh.mask_token, model.mask_token)
    assert fresh_opt.state_dict()["state"]  # optimizer moments restored
    assert latest_checkpoint(tmp_path / "nothing") is None


def test_fit_resume_and_export(tokenized_catalog, tmp_path):
    cfg = tiny_train_config(tokenized_catalog, tmp_path / "run")
    out = train(cfg)
    names = sorted(p.name for p in out.iterdir())
    assert names == ["checkpoint-0.pth", "checkpoint-1.pth", "final", "log.txt"]
    records = [json.loads(line) for line in (out / "log.txt").read_text().splitlines()]
    assert [r["epoch"] for r in records] == [0, 1]
    assert all(r["[Epoch] loss"] < 20 and r["n_parameters"] > 0 for r in records)
    assert records[1]["total_tokens_seen_b"] == pytest.approx(2 * 2 * 4 * 96 / 1e9)

    # auto-resume from the last checkpoint: nothing left to train
    resumed = Trainer(cfg).setup()
    assert resumed.start_epoch == 2 and resumed.epochs == 2 and resumed.train_dataset.epoch == 2
    # explicit resume from epoch 0 continues at epoch 1
    cfg_epoch1 = tiny_train_config(tokenized_catalog, tmp_path / "run", **{"run.resume": str(out / "checkpoint-0.pth")})
    trainer = Trainer(cfg_epoch1).setup()
    assert trainer.start_epoch == 1
    stats = trainer.train_one_epoch(1)
    assert "[Epoch] loss" in stats and stats["[Epoch] grad_norm"] > 0

    from aion.fourm.fm import FM

    model = FM.from_pretrained(str(out / "final"))
    assert model.encoder_embeddings["tok_image"].num_patches == 576


def test_max_steps_and_eval_and_catalog_override(tokenized_catalog, tmp_path):
    import lsdb

    catalog = lsdb.open_catalog(tokenized_catalog, columns=["tok_image"])
    cfg = tiny_train_config(
        tokenized_catalog,
        tmp_path / "run",
        **{
            "run.max_steps": 3,
            "schedule.epochs": 5,
            "data.eval_datasets": "[{name: val, catalog: unused, modalities: [tok_image]}]",
            "data.eval_steps": 1,
        },
    )
    trainer = Trainer(cfg, catalogs={"train": catalog, "val": catalog})
    out = trainer.fit()
    assert trainer.epochs == 2  # ceil(3 / 2)
    records = [json.loads(line) for line in (out / "log.txt").read_text().splitlines()]
    assert len(records) == 2 and "[Eval (val)] loss" in records[0]
    assert (out / "checkpoint-1.pth").exists() and (out / "final" / "model.safetensors").exists()
