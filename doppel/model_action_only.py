"""Action-only baseline transformer: ``ActionOnlyWanTransformer``.

A subclass of ``WanTransformerConditioned`` (model.py) conditioned on
the numerical action trajectory only, following the conditioning mechanism of
EgoWM: the action adapter (action_only_adapter.py) produces a per-token
contribution that is added to the flow-timestep embedding before ``time_proj``
(per-token timestep modulation). There is no render encoder and no
render-action fusion.

The render is never used, so ``actions is None`` is the unconditional branch
(unmodified Wan I2V forward), and ``render_latents`` is accepted and discarded
so that training and sampling code can pass the same arguments to every model.
On a conditioning-dropout step the training loop omits ``actions``, which
lands in that unconditional branch.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers

from doppel.model import WanTransformerConditioned
from doppel.action_only_adapter import EgoWMActionAdapter


class ActionOnlyWanTransformer(WanTransformerConditioned):
    """Wan DiT conditioned on the action trajectory only (no rendering)."""

    _keep_in_fp32_modules = [
        "time_embedder",
        "time_proj",
        "action_adapter_only",
        "scale_shift_table",
        "norm_out",
    ]

    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        action_only_adapter_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            patch_size=patch_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            qk_norm=qk_norm,
            eps=eps,
            image_dim=image_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=rope_max_seq_len,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        inner_dim = num_attention_heads * attention_head_dim

        kw = dict(action_only_adapter_kwargs or {})
        if "action_dim" not in kw:
            raise ValueError(
                "ActionOnlyWanTransformer requires "
                "action_only_adapter_kwargs.action_dim (e.g. 8 for joint+gripper)."
            )
        kw.setdefault("inner_dim", inner_dim)
        kw.setdefault("hidden_dim", 512)
        self.action_adapter_only = EgoWMActionAdapter(**kw)
        self.action_adapter_only.to(torch.float32)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Load the Wan checkpoint, then initialise the adapter if the checkpoint has no
        adapter weights (see ``_AdapterInitMixin``).
        """
        want_info = kwargs.pop("output_loading_info", False)
        try:
            model, info = super().from_pretrained(*args, output_loading_info=True, **kwargs)
        except TypeError:                       # diffusers versions without the argument
            model, info = super().from_pretrained(*args, **kwargs), None
        missing = list((info or {}).get("missing_keys", ["action_adapter_only."]))
        if any(str(k).startswith("action_adapter_only") for k in missing):
            model.action_adapter_only.init_weights()
        return (model, info) if want_info else model

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        render_latents: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # `render_latents` is accepted and ignored: this is the render-free
        # baseline. `actions is None` is the unconditional branch.
        del render_latents

        if actions is None:
            return WanTransformerConditioned.__bases__[0].forward(
                self,
                hidden_states,
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
            )

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        tokens_per_frame = post_patch_height * post_patch_width
        num_tokens = post_patch_num_frames * tokens_per_frame

        rotary_emb = self.rope(hidden_states)
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if timestep.ndim == 2:
            timestep = timestep[:, 0]

        temb_base, _unused_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep, encoder_hidden_states, encoder_hidden_states_image,
                timestep_seq_len=None,
            )
        )  # temb_base: (B, inner_dim)

        ce = self.condition_embedder

        # ---- action-only conditioning ---------------------------------------
        aa_dtype = next(self.action_adapter_only.parameters()).dtype
        contribution = self.action_adapter_only(
            actions.to(device=hidden_states.device, dtype=aa_dtype),
            num_post_patch_frames=post_patch_num_frames,
            post_patch_height=post_patch_height,
            post_patch_width=post_patch_width,
        )  # (B, num_tokens, inner_dim)
        if contribution.shape[1] != num_tokens:
            raise RuntimeError(
                f"The action-only adapter produced {contribution.shape[1]} tokens but "
                f"the DiT expects {num_tokens}."
            )
        contribution = contribution.to(dtype=temb_base.dtype)
        temb_per_token = self.action_adapter_only.combine_with_temb(
            temb_base, contribution
        )  # (B, num_tokens, inner_dim)

        # Per-token timestep modulation: (B, num_tokens, 6, inner_dim).
        tp_in = ce.act_fn(temb_per_token.to(ce.time_proj.weight.dtype))
        timestep_proj = ce.time_proj(tp_in).to(dtype=temb_per_token.dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
        temb_for_norm_out = temb_per_token
        # ---------------------------------------------------------------------

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )

        shift, scale = (
            self.scale_shift_table.unsqueeze(0) + temb_for_norm_out.unsqueeze(2)
        ).chunk(2, dim=2)
        shift = shift.squeeze(2).to(hidden_states.device)
        scale = scale.squeeze(2).to(hidden_states.device)

        hidden_states = (
            self.norm_out(hidden_states.float()) * (1 + scale) + shift
        ).type_as(hidden_states)

        diffusion_hidden = self.proj_out(hidden_states)
        diffusion_hidden = diffusion_hidden.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width,
            p_t, p_h, p_w, -1,
        )
        diffusion_hidden = diffusion_hidden.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = diffusion_hidden.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


__all__ = ["ActionOnlyWanTransformer"]
