import os

import torch

from doppel.checkpoint import load_conditioner, remap_state_dict


def _inputs(B=1, T=5, H=8, W=8):
    g = torch.Generator().manual_seed(1)
    return dict(hidden_states=torch.randn(B, 36, T, H, W, generator=g),
                timestep=torch.tensor([500.0] * B),
                encoder_hidden_states=torch.randn(B, 7, 32, generator=g),
                render_latents=torch.randn(B, 16, T, H, W, generator=g),
                actions=torch.randn(B, 17, 159, generator=g))


def test_forward_shapes(tiny_redar):
    x = _inputs()
    with torch.no_grad():
        out = tiny_redar(**x, return_dict=False)[0]
        plain = tiny_redar(**{**x, "render_latents": None}, return_dict=False)[0]
    assert out.shape == (1, 16, 5, 8, 8) and torch.isfinite(out).all()
    assert plain.shape == out.shape and not torch.allclose(plain, out)


def test_actions_change_output(tiny_redar):
    x = _inputs()
    with torch.no_grad():
        a = tiny_redar(**x, return_dict=False)[0]
        x["actions"] = x["actions"] + 1.0
        b = tiny_redar(**x, return_dict=False)[0]
    assert not torch.allclose(a, b)


def test_padded_slots_are_inert(tiny_redar):
    """Zero-padded slots only see zeros, so widening an 8-d action by zeros is exact."""
    x = _inputs()
    x["actions"][..., 8:] = 0
    with torch.no_grad():
        a = tiny_redar(**x, return_dict=False)[0]
        w = tiny_redar.render_action_adapter.action_encoder.net[0].weight
        w[:, 8:] += 1.0                                        # weights of always-zero slots
        b = tiny_redar(**x, return_dict=False)[0]
    assert torch.equal(a, b)


def test_legacy_checkpoint_roundtrip(tiny_redar, tmp_path):
    from tests.conftest import TINY
    from doppel.model_redar import RenderActionWanTransformer
    sd = {k.replace("render_action_adapter.", "action_adapter_v4."): v.clone()
          for k, v in tiny_redar.state_dict().items() if not k.startswith("blocks.0.")}
    sd["tracks_head.mlp.0.weight"] = torch.zeros(2, 2)          # research-code auxiliary head: dropped
    p = os.path.join(tmp_path, "conditioner.pt")
    torch.save(sd, p)
    torch.manual_seed(123)
    fresh = RenderActionWanTransformer(**TINY, redar_adapter_kwargs=dict(action_dim=159, hidden_dim=32, num_heads=2))
    info = load_conditioner(fresh, p)
    assert info["renamed"] > 0 and info["dropped"] == 1 and info["n_trained_blocks"] == 1
    ref = tiny_redar.state_dict()
    for k, v in fresh.state_dict().items():
        if not k.startswith("blocks.0."):
            assert torch.equal(v, ref[k]), k
    remapped, _ = remap_state_dict({"action_adapter_v4.x": 1, "blocks.0.y": 2})
    assert remapped == {"render_action_adapter.x": 1, "blocks.0.y": 2}


def test_safetensors_overlay_roundtrip(tiny_redar, tmp_path):
    """A released overlay (sharded safetensors + index, research-code key names) loads like
    conditioner.pt, given as the release dir, the overlay dir or the index file."""
    import json
    from safetensors.torch import save_file
    from tests.conftest import TINY
    from doppel.model_redar import RenderActionWanTransformer
    sd = {k.replace("render_action_adapter.", "action_adapter_v4."): v.clone().contiguous()
          for k, v in tiny_redar.state_dict().items() if not k.startswith("patch_embedding.")}
    keys = sorted(sd)
    ov = tmp_path / "rel" / "transformer_overlay"
    ov.mkdir(parents=True)
    halves = {"model-00001-of-00002.safetensors": keys[::2], "model-00002-of-00002.safetensors": keys[1::2]}
    weight_map = {}
    for fn, ks in halves.items():
        save_file({k: sd[k] for k in ks}, str(ov / fn))
        weight_map.update({k: fn for k in ks})
    (ov / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    ref = tiny_redar.state_dict()
    for p in (tmp_path / "rel", ov, ov / "model.safetensors.index.json"):
        torch.manual_seed(321)
        fresh = RenderActionWanTransformer(**TINY, redar_adapter_kwargs=dict(action_dim=159, hidden_dim=32, num_heads=2))
        info = load_conditioner(fresh, str(p))
        assert info["renamed"] > 0 and info["n_tensors"] == len(sd)
        for k, v in fresh.state_dict().items():
            if not k.startswith("patch_embedding."):
                assert torch.equal(v, ref[k]), k
