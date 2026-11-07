import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


TINY = dict(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=16, in_channels=36, out_channels=16,
            text_dim=32, freq_dim=32, ffn_dim=64, num_layers=2, image_dim=None, rope_max_seq_len=64)


@pytest.fixture
def tiny_redar():
    """A 2-block RenderActionWanTransformer with the 8-embodiment adapter width (159)."""
    from doppel.model_redar import RenderActionWanTransformer
    torch.manual_seed(0)
    m = RenderActionWanTransformer(**TINY, redar_adapter_kwargs=dict(action_dim=159, hidden_dim=32, num_heads=2))
    m.render_action_adapter.init_weights()
    return m.eval()
