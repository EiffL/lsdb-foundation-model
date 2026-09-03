import json

import pytest

pytest.importorskip("lsdb")

from aion_hats.cli import build_parser, main


def test_train_help():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["train", "--help"])
    assert exc.value.code == 0


def test_train_smoke_via_cli(tokenized_catalog, tmp_path, capsys):
    out = tmp_path / "run"
    code = main(
        [
            "train",
            "--catalog", str(tokenized_catalog),
            "--output-dir", str(out),
            "--preset", "tiny",
            "--set", "model.overrides={dim: 64, encoder_depth: 1, decoder_depth: 1, num_heads: 2}",
            "--set", "data.num_input_tokens=64",
            "--set", "data.num_target_tokens=32",
            "--batch-size", "4",
            "--epochs", "1",
            "--steps-per-epoch", "2",
            "--num-workers", "0",
            "--device", "cpu",
            "--dtype", "float32",
            "--no-auto-resume",
        ]
    )
    assert code == 0
    records = [json.loads(line) for line in (out / "log.txt").read_text().splitlines()]
    assert len(records) == 1 and (out / "final" / "config.json").exists()
    assert str(out) in capsys.readouterr().out


def test_train_requires_a_catalog(tmp_path):
    with pytest.raises(SystemExit):
        main(["train", "--output-dir", str(tmp_path)])
