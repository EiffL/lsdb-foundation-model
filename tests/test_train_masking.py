import random

import pytest
import torch

from aion_hats.train import UnifiedMasking, empty_mod_dict, resolve_modality_info


@pytest.fixture(autouse=True)
def _seed():
    random.seed(0)
    torch.manual_seed(0)


def test_resolve_modality_info_fills_tokens_and_alphas():
    info = resolve_modality_info(["tok_image", "tok_flux_g"], ["tok_image"], input_alphas=2.0)
    assert list(info) == ["tok_flux_g", "tok_image"]
    assert info["tok_image"]["max_tokens"] == 576 and info["tok_flux_g"]["max_tokens"] == 1
    assert info["tok_image"]["input_alphas"] == [2.0] and info["tok_image"]["target_alphas"] == [1.0]
    assert info["tok_flux_g"]["target_alphas"] == [0.0]  # not an output modality
    # the global registry is untouched
    from aion.fourm.modality_info import MODALITY_INFO

    assert MODALITY_INFO["tok_image"]["max_tokens"] is None
    with pytest.raises(ValueError, match="unknown AION modality"):
        resolve_modality_info(["tok_nope"], [])
    with pytest.raises(ValueError, match="only"):
        resolve_modality_info(["catalog"], ["catalog"])


def test_image_mask_budgets_are_disjoint():
    tensor = torch.arange(576)
    out = UnifiedMasking.image_mask(tensor, 576, input_budget=100, target_budget=50)
    assert out["tensor"] is tensor
    assert int((~out["input_mask"]).sum()) == 100 and int((~out["target_mask"]).sum()) == 50
    assert not (~out["input_mask"] & ~out["target_mask"]).any()
    dam = out["decoder_attention_mask"]
    assert dam.dtype == torch.int and int((dam != 0).sum()) == 1
    first_target = int(torch.nonzero(~out["target_mask"])[0])
    assert int(dam[first_target]) == 50
    everything = UnifiedMasking.image_mask(tensor, 576, 100, None)
    assert torch.equal(everything["target_mask"], ~everything["input_mask"])


def test_single_modality_budgets_and_ranges():
    info = resolve_modality_info(["tok_image"], ["tok_image"])
    masker = UnifiedMasking(info, (32, 128), (16, 64))
    for _ in range(50):
        out = masker({"tok_image": torch.zeros(576, dtype=torch.long)})["tok_image"]
        n_in = int((~out["input_mask"]).sum())
        n_tgt = int((~out["target_mask"]).sum())
        assert 1 <= n_in <= 128 and 1 <= n_tgt <= 64
        assert not (~out["input_mask"] & ~out["target_mask"]).any()
        assert int(out["decoder_attention_mask"].max()) == n_tgt
    fixed = UnifiedMasking(info, 256, 128)
    out = fixed({"tok_image": torch.zeros(576, dtype=torch.long)})["tok_image"]
    assert int((~out["input_mask"]).sum()) <= 256 and int((~out["target_mask"]).sum()) <= 128
    with pytest.raises(ValueError, match="expected 576 tokens"):
        fixed({"tok_image": torch.zeros(10, dtype=torch.long)})


def test_two_modalities_budgets_sum_and_absent_modality_is_masked():
    info = resolve_modality_info(["tok_image", "tok_flux_g"], ["tok_image", "tok_flux_g"])
    masker = UnifiedMasking(info, 128, 64)
    for _ in range(30):
        in_budget = masker.input_token_budget(128)
        assert sum(in_budget) <= 128 and in_budget[list(info).index("tok_flux_g")] <= 1
        assert max(in_budget) >= 1
        tgt_budget = masker.target_token_budget(in_budget, 64)
        assert sum(tgt_budget) <= 64 and max(tgt_budget) >= 1
        for i, mod in enumerate(info):
            assert in_budget[i] + tgt_budget[i] <= info[mod]["max_tokens"]
    sample = masker({"tok_image": torch.zeros(576, dtype=torch.long)})
    assert set(sample) == {"tok_flux_g", "tok_image"}
    absent = sample["tok_flux_g"]
    assert absent["input_mask"].all() and absent["target_mask"].all()
    assert absent["tensor"].shape == (1,) and absent["tensor"].dtype == torch.long


def test_inactive_modality_never_gets_budget():
    info = resolve_modality_info(["tok_image"], ["tok_image", "tok_flux_g"])  # flux_g: target only
    masker = UnifiedMasking(info, 128, 64)
    idx = list(info).index("tok_flux_g")
    for _ in range(30):
        assert masker.input_token_budget(128)[idx] == 0
    empty = empty_mod_dict(info)
    assert empty["tok_image"]["tensor"].shape == (576,) and empty["tok_flux_g"]["input_mask"].all()


def test_mixture_weights_validation():
    info = resolve_modality_info(["tok_image"], ["tok_image"])
    with pytest.raises(ValueError, match="sampling_weights"):
        UnifiedMasking(info, 128, 64, sampling_weights=[0.5, 0.5])
    UnifiedMasking(info, 128, 64, sampling_weights=[1.0])
