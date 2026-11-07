import glob
import json
import os
import sys

import pytest

from doppel import data_multi as dm
from doppel.checkpoint import model_config

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = sorted(glob.glob(os.path.join(REPO, "configs", "*.json")))


def _parse(argv, monkeypatch):
    from doppel import train
    monkeypatch.setattr(sys, "argv", ["train"] + argv)
    return train.parse_args()


@pytest.mark.parametrize("path", CONFIGS, ids=os.path.basename)
def test_config_roundtrip(path, monkeypatch, tmp_path):
    """Every config parses (unknown keys are rejected), and dumping the parsed options and
    parsing them again gives the same options."""
    args = vars(_parse(["--config", path], monkeypatch))
    args.pop("config")
    dumped = tmp_path / "args.json"
    dumped.write_text(json.dumps(args))
    again = vars(_parse(["--config", str(dumped)], monkeypatch))
    again.pop("config")
    assert again == args
    cfg = json.load(open(path))
    for k, v in cfg.items():
        assert args[k] == v, k
    stats = dm.load_stats_map(cfg["action_stats_path"], cfg["dataset_base_path"])  # shipped stats resolve
    mc = model_config(cfg)
    w = mc["action_pad_dim"] or max(dm.action_dim(rt) for rt in stats)
    kw = mc["redar_adapter_kwargs"] or mc["action_only_adapter_kwargs"]
    assert kw["action_dim"] == w


def test_emb8_config():
    cfg = json.load(open(os.path.join(REPO, "configs", "emb8_render_action_4node.json")))
    assert cfg["action_pad_dim"] == cfg["redar_adapter_kwargs"]["action_dim"] == 159
    assert set(cfg["action_stats_path"]) == set(dm.EMBODIMENTS)
    w = cfg["sampler_dataset_weights"]
    assert set(w) == set(dm.DATASET_IDS) and abs(sum(w.values()) - 1) < 1e-4
    assert cfg["batch_size"] == 8 and cfg["total_epochs"] == 2490


def test_legacy_meta_is_accepted():
    meta = {"legacy_render_variant": "v4", "v4_adapter_kwargs": {"action_dim": 159, "hidden_dim": 512},
            "action_pad_dim": 159, "action_stats_path": {"panda": "x", "smplh_human": "y"}}
    mc = model_config(meta)
    assert mc["conditioning"] == "redar" and mc["redar_adapter_kwargs"]["action_dim"] == 159
    assert mc["robot_types"] == ["panda", "smplh_human"]
    with pytest.raises(ValueError):
        model_config({**meta, "action_pad_dim": 64})


def test_legacy_action_only_meta():
    mc = model_config({"legacy_render_variant": "action_only",
                       "action_only_adapter_kwargs": {"mode": "egowm", "action_dim": 8, "hidden_dim": 512}})
    assert mc["conditioning"] == "action_only" and "mode" not in mc["action_only_adapter_kwargs"]


def test_release_config_json():
    """The released model's config.json (nested layout) gives the same model config as the
    training config of that model."""
    rel = {"architecture": {"legacy_render_variant": "v4",
                            "v4_adapter_kwargs": json.load(open(os.path.join(
                                REPO, "configs", "emb8_render_action_4node.json")))["redar_adapter_kwargs"]},
           "inputs": {"num_frames": 17, "height": 240, "width": 432, "action_field": "state",
                      "action_pad_dim": 159},
           "action_stats": {rt: {"file": f"action_stats/{rt}.json"} for rt in dm.EMBODIMENTS}}
    a = model_config(rel)
    b = model_config(json.load(open(os.path.join(REPO, "configs", "emb8_render_action_4node.json"))))
    assert a == b
