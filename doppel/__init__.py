"""Doppel: one video world model across embodiments (Wan 2.1 I2V conditioned on renderings and actions).

The heavy submodules (diffusers models, training) are imported lazily, so ``doppel.data_multi``,
``doppel.checkpoint`` and ``doppel.metrics`` can be used without loading diffusers.
"""

__version__ = "0.1.0"

_LAZY = {
    "RenderConditionedWanI2VPipeline": "doppel.model",
    "WanTransformerConditioned": "doppel.model",
    "WanVAEChunkedEncode": "doppel.model",
    "ActionOnlyWanTransformer": "doppel.model_action_only",
    "RenderActionWanTransformer": "doppel.model_redar",
    "DoppelWorldModel": "doppel.inference",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'doppel' has no attribute {name!r}")
