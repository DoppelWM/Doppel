import json
import os

import numpy as np
import pytest
import torch

from doppel import data_multi as dm

STATS_DIR = os.path.join(dm.ASSETS_DIR, "action_stats")


def all_stats():
    return dm.load_stats_map({rt: os.path.join(STATS_DIR, f"{rt}.json") for rt in dm.EMBODIMENTS})


def test_registry_widths():
    w = {rt: dm.action_dim(rt) for rt in dm.EMBODIMENTS}
    assert w == {"panda": 8, "ur5e_2f85": 8, "widowx": 8, "smplx_human": 159, "yam_bimanual": 14,
                 "mano_hands": 108, "smplh_human": 159, "vega1_sharpa": 58}
    assert dm.MAX_ACTION_DIM == 159
    assert sum(1 - x for x in dm.embodiment("smplx_human")["dof_mask"]) == 42
    assert sorted(dm.DATASET_IDS.values()) == list(range(8))


def test_shipped_stats_match_registry():
    st = all_stats()
    assert set(st) == set(dm.EMBODIMENTS)
    layouts = json.load(open(os.path.join(dm.ASSETS_DIR, "embodiments.json")))["embodiments"]
    for rt, s in st.items():
        assert s["mean"].shape == (dm.action_dim(rt),)
        assert layouts[rt]["D"] == dm.action_dim(rt)
        blob = json.load(open(s["path"]))
        if blob["root_unwrap"]:
            assert "root_rotvec" in dm.embodiment(rt)
    assert json.load(open(st["smplh_human"]["path"]))["root_unwrap"]


@pytest.mark.parametrize("lib", ["numpy", "torch"])
def test_pad_actions(lib):
    a = np.random.randn(17, 14).astype(np.float32)
    m = np.ones(14, np.float32)
    if lib == "torch":
        a, m = torch.from_numpy(a), torch.from_numpy(m)
    pa, pm = dm.pad_actions(a, m, 159)
    assert tuple(pa.shape) == (17, 159) and tuple(pm.shape) == (159,)
    assert float(abs(pa[:, 14:]).sum()) == 0 and float(pm[14:].sum()) == 0
    assert np.array_equal(np.asarray(pa[:, :14]), np.asarray(a))
    same = dm.pad_actions(a, m, 14)
    assert same[0] is a
    with pytest.raises(ValueError):
        dm.pad_actions(a, m, 8)


def test_unwrap_rotvec():
    assert dm.embodiment("smplx_human")["root_rotvec"] == dm.embodiment("smplh_human")["root_rotvec"] == (0, 3)
    rng = np.random.default_rng(0)
    smooth = np.cumsum(rng.normal(0, 0.01, (17, 3)), 0) + np.array([0.3, 0.2, 0.1])
    assert dm.unwrap_rotvec(smooth) is smooth                      # no flip: unchanged
    th = np.linspace(2.9, 3.3, 17)                                 # crosses pi about one axis
    u = np.array([1.0, 0, 0])
    exported = np.stack([u * t if t <= np.pi else -u * (2 * np.pi - t) for t in th])
    fixed = dm.unwrap_rotvec(exported)
    assert np.allclose(fixed, np.stack([u * t for t in th]), atol=1e-6)


def test_normalize_actions_all_embodiments():
    st = all_stats()
    for rt, s in st.items():
        D = dm.action_dim(rt)
        raw = (s["mean"] + s["std"] * 0.5)[None].repeat(17, 0)
        raw[:, s["dof_mask"] == 0] = 0
        z, m = dm.normalize_actions(raw, rt, st)
        assert z.shape == (17, D) and z.dtype == np.float32
        assert np.allclose(z[:, m == 1], 0.5, atol=1e-4) and np.all(z[:, m == 0] == 0)


def test_load_actions_npz(tmp_path):
    st = all_stats()
    raw = np.random.randn(17, 8).astype(np.float32)
    raw[:, 6] = 0
    p = tmp_path / "a.npz"
    np.savez(p, state=raw, robot_type=np.array(["widowx"]))
    z, m = dm.load_actions(str(p), "widowx", st)
    assert np.allclose(z, dm.normalize_actions(raw, "widowx", st)[0])
    with pytest.raises(ValueError):
        dm.load_actions(str(p), "ur5e_2f85", st)                   # npz says widowx


class _FakeStore:
    def get(self, row, with_render):
        out = {"clean_latents": torch.zeros(16, 5, 30, 54), "cond_latents": torch.zeros(16, 5, 30, 54),
               "image_embeds": torch.zeros(257, 1280)}
        if with_render:
            out["render_latents"] = torch.zeros(16, 5, 30, 54)
        return out

    def get_prompt(self, text):
        return torch.zeros(1, 4096)


def test_cache_pads_mixed_widths(tmp_path):
    st = all_stats()
    rows = []
    for rt in ("panda", "smplh_human", "vega1_sharpa"):
        raw = np.zeros((17, dm.action_dim(rt)), np.float32)
        np.savez(tmp_path / f"{rt}.npz", state=raw)
        rows.append(dict(dataset=dm.embodiment(rt)["dataset"], video=f"{rt}.mp4", robot_type=rt, actions=f"{rt}.npz"))
    with pytest.raises(ValueError):
        dm.MultiEmbodimentCache(rows, str(tmp_path), _FakeStore(), st, with_render=True)
    cache = dm.MultiEmbodimentCache(rows, str(tmp_path), _FakeStore(), st, with_render=True, action_pad_dim=159)
    for i in range(3):
        s = cache[i]
        assert tuple(s["actions"].shape) == (17, 159)
        assert s["render_latents"].shape == (1, 16, 5, 30, 54)
