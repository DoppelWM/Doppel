"""
The render-action adapter g_φ (paper Sec. 3 and appendix): one shared adapter for
all embodiments that fuses rendering and action trajectory into a dense,
per-token conditioning signal.

Maps the VAE latents of the robot rendering sequence and the numerical
configuration trajectory to one conditioning vector per video token:

1. Render encoder (:class:`RenderTokenEncoderWithSelfAttn`): per-channel
   centering of the render latents, three Conv3d layers (the middle one mixes
   time), factorized learned 3D positional embeddings and pre-norm
   self-attention blocks, on the backbone's post-patch token grid.
2. Action encoder (:class:`ActionTokenEncoder`): a shared per-frame MLP plus a
   learned embedding of the frame index.
3. Render-action fusion (:class:`CrossAttnTransformerBlock`): render tokens
   (queries) cross-attend to the action tokens (keys/values).
4. Projection to the backbone width; the per-token result is added to the
   flow-timestep embedding before ``time_proj`` (per-token timestep
   modulation, see :meth:`RenderActionAdapter.combine_with_temb`).

Initialization: PyTorch default initialization (MultiheadAttention input
projections Xavier), positional tables N(0, 0.02); nothing is
zero-initialized.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _default_init_(module: nn.Module) -> None:
    """Re-apply PyTorch's own default initialisation to one module."""
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv3d)):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(module.bias, -bound, bound)
    elif isinstance(module, nn.LayerNorm):
        if module.elementwise_affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        if module.in_proj_weight is not None:
            nn.init.xavier_uniform_(module.in_proj_weight)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
        if module.out_proj.bias is not None:
            nn.init.zeros_(module.out_proj.bias)


class _AdapterInitMixin:
    """Initialisation for adapters built inside diffusers' ``from_pretrained``.

    ``from_pretrained`` constructs the model under ``no_init_weights()`` and leaves
    parameters missing from the checkpoint on the meta device, so the init in
    ``__init__`` never runs. ``init_weights()`` materialises them and applies the
    default init (bare parameters such as positional tables: N(0, BARE_STD)).
    """

    #: bare nn.Parameters get normal(0, BARE_STD); subclasses may override.
    BARE_STD = 0.02

    def init_weights(self) -> None:
        # 1. Materialise anything the checkpoint did not provide.
        for mod in self.modules():
            for name, p in list(mod.named_parameters(recurse=False)):
                if p.is_meta:
                    setattr(mod, name, nn.Parameter(
                        torch.empty(p.shape, dtype=p.dtype, device="cpu"),
                        requires_grad=p.requires_grad,
                    ))
            for name, b in list(mod.named_buffers(recurse=False)):
                if b is not None and b.is_meta:
                    setattr(mod, name, torch.empty(b.shape, dtype=b.dtype, device="cpu"))
        # 2. Standard init for the standard layers.
        typed = (nn.Linear, nn.Conv1d, nn.Conv3d, nn.LayerNorm, nn.MultiheadAttention)
        for mod in self.modules():
            if isinstance(mod, typed):
                _default_init_(mod)
        # 3. Bare parameters, which own no reset_parameters().
        for mod in self.modules():
            if isinstance(mod, typed):
                continue
            for _name, p in mod.named_parameters(recurse=False):
                nn.init.normal_(p, std=self.BARE_STD)


class ActionTokenEncoder(nn.Module):
    """(B, T_act, action_dim) -> (B, T_act, hidden): Linear, GELU, Linear."""

    def __init__(self, action_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.net(actions)


class CrossAttnTransformerBlock(nn.Module):
    """Pre-norm transformer block: residual cross-attention + residual FFN.

    Query stream = render tokens; key/value = action tokens.
    """

    def __init__(self, dim: int = 512, num_heads: int = 8, ffn_mult: int = 4) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        ctx = self.norm_kv(context)
        attn_out, _ = self.cross_attn(
            self.norm_q(x), ctx, ctx, need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class SelfAttnTransformerBlock(nn.Module):
    """Pre-norm transformer block: self-attention + residual + FFN.

    Same structure as CrossAttnTransformerBlock but Q, K, V all come from
    the same input. Used inside RenderTokenEncoderWithSelfAttn to let
    spatially distant render tokens exchange information: most tokens are
    background and only a small fraction carry robot-pose information.
    """

    def __init__(self, dim: int = 512, num_heads: int = 8, ffn_mult: int = 4) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = self.norm_attn(x)
        attn_out, _ = self.self_attn(x_n, x_n, x_n, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class PositionalEncoding3D(nn.Module):
    """Factorized 3D learned positional encoding.

    For a grid of (T, H, W) tokens with feature dim D, returns a
    (T*H*W, D) embedding such that ``pos[t, h, w] = pos_t[t] + pos_h[h] +
    pos_w[w]``. Factorization keeps the parameter cost at
    O(max_t + max_h + max_w) instead of O(max_t · max_h · max_w).
    """

    def __init__(self, max_t: int, max_h: int, max_w: int, dim: int) -> None:
        super().__init__()
        self.pos_t = nn.Parameter(torch.zeros(max_t, dim))
        self.pos_h = nn.Parameter(torch.zeros(max_h, dim))
        self.pos_w = nn.Parameter(torch.zeros(max_w, dim))
        nn.init.normal_(self.pos_t, std=0.02)
        nn.init.normal_(self.pos_h, std=0.02)
        nn.init.normal_(self.pos_w, std=0.02)

    def forward(self, T: int, H: int, W: int) -> torch.Tensor:
        """Returns (T*H*W, dim)."""
        t = self.pos_t[:T].unsqueeze(1).unsqueeze(2)    # (T, 1, 1, dim)
        h = self.pos_h[:H].unsqueeze(0).unsqueeze(2)    # (1, H, 1, dim)
        w = self.pos_w[:W].unsqueeze(0).unsqueeze(1)    # (1, 1, W, dim)
        return (t + h + w).reshape(T * H * W, -1)


class RenderTokenEncoderWithSelfAttn(nn.Module):
    """Render encoder: (B, C, T, H, W) render latents -> (B, T*h*w, hidden).

    ``h = H // spatial_downsample``, ``w = W // spatial_downsample``;
    spatial_downsample=2 matches Wan's patch_size=(1, 2, 2), so the output
    tokens line up 1:1 with the DiT post-patch grid.

    1. **Latent centering.** Subtract the per-sample, per-channel
       spatiotemporal mean from the render latents before the first Conv3d.
       Robot renderings are mostly empty background, so this mean is close to
       the VAE response of an empty canvas; after subtraction, background
       positions are near zero and robot positions keep their deviation.

    2. **Conv3d stack.** (1, 3, 3) spatial conv -> (3, 3, 3) conv that mixes
       adjacent latent frames (replicate-padded in time) -> (1, 3, 3) conv
       with stride (1, s, s) that maps to the token grid. GELU after each.

    3. **Positional encoding + self-attention.** Factorized learned 3D
       positional embeddings, then N pre-norm self-attention blocks so that
       information from the sparse robot region reaches tokens outside the
       convolutions' receptive field.
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_dim: int = 512,
        spatial_downsample: int = 2,
        num_self_blocks: int = 2,
        num_heads: int = 8,
        ffn_mult: int = 4,
        max_t: int = 32,
        max_h: int = 32,
        max_w: int = 64,
    ) -> None:
        super().__init__()
        s = spatial_downsample
        mid = hidden_dim // 2
        # Individual Conv3d modules (not Sequential) so the temporal dim of the
        # middle conv can be padded with replicate mode instead of zeros: the
        # boundary frame extends, rather than an empty frame before/after the clip.
        self.conv1 = nn.Conv3d(in_channels, mid, (1, 3, 3), padding=(0, 1, 1))
        # Middle conv: kernel (3,3,3). Spatial zero padding is baked in (the
        # area outside a rendering is background); temporal padding is done
        # manually with replicate mode in forward.
        self.conv2 = nn.Conv3d(mid, hidden_dim, (3, 3, 3), padding=(0, 1, 1))
        self.conv3 = nn.Conv3d(
            hidden_dim, hidden_dim,
            kernel_size=(1, 3, 3),
            stride=(1, s, s),
            padding=(0, 1, 1),
        )
        # Learned (t, h, w) embedding, factored as pos_t + pos_h + pos_w, added
        # after flattening to the per-token sequence.
        self.pos_enc = PositionalEncoding3D(max_t, max_h, max_w, hidden_dim)
        self.self_attn_blocks = nn.ModuleList([
            SelfAttnTransformerBlock(hidden_dim, num_heads, ffn_mult)
            for _ in range(num_self_blocks)
        ])

    def forward(self, render_latents: torch.Tensor) -> torch.Tensor:
        # Per-sample, per-channel spatiotemporal mean subtraction on the VAE
        # latents. Computed in fp32: under FSDP the adapter may run in bf16,
        # and the robot-vs-background deviation is a small difference of large
        # values that bf16's 7-bit mantissa would cancel.
        in_dtype = render_latents.dtype
        rl32 = render_latents.float()
        render_latents = (rl32 - rl32.mean(dim=(2, 3, 4), keepdim=True)).to(in_dtype)

        # Conv1: spatial-only kernel.
        x = F.gelu(self.conv1(render_latents))

        # Conv2: temporal mixing. Replicate-pad T by 1 on each side. F.pad arg
        # order is reverse-dim: (W_left, W_right, H_left, H_right, T_left, T_right).
        x = F.pad(x, (0, 0, 0, 0, 1, 1), mode="replicate")
        x = F.gelu(self.conv2(x))

        # Conv3: strided spatial downsample.
        x = F.gelu(self.conv3(x))

        B, C, T, H, W = x.shape                                # T=T_lat, H=h, W=w
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)  # (B, T·h·w, hidden)

        # Add 3D positional encoding. Broadcast over batch dim.
        x = x + self.pos_enc(T, H, W).unsqueeze(0)

        for block in self.self_attn_blocks:
            x = block(x)
        return x


class RenderActionAdapter(_AdapterInitMixin, nn.Module):
    """Render-action adapter g_φ(render latents, actions) -> (B, N, inner_dim).

    Render tokens from :class:`RenderTokenEncoderWithSelfAttn` (queries)
    cross-attend to the positionally embedded action tokens (keys/values)
    through ``num_cross_blocks`` pre-norm blocks; a LayerNorm -> Linear ->
    GELU -> Linear head lifts each fused token to the backbone width. The
    result is added to the flow-timestep embedding before ``time_proj`` via
    :meth:`combine_with_temb`.
    """

    def __init__(
        self,
        action_dim: int,
        inner_dim: int = 5120,
        render_in_channels: int = 16,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_self_blocks: int = 2,
        num_cross_blocks: int = 2,
        spatial_downsample: int = 2,
        ffn_mult: int = 4,
        max_action_frames: int = 128,
        max_t: int = 32,
        max_h: int = 32,
        max_w: int = 64,
    ) -> None:
        super().__init__()
        self.inner_dim = inner_dim
        self.hidden_dim = hidden_dim

        self.render_encoder = RenderTokenEncoderWithSelfAttn(
            in_channels=render_in_channels,
            hidden_dim=hidden_dim,
            spatial_downsample=spatial_downsample,
            num_self_blocks=num_self_blocks,
            num_heads=num_heads,
            ffn_mult=ffn_mult,
            max_t=max_t,
            max_h=max_h,
            max_w=max_w,
        )
        self.action_encoder = ActionTokenEncoder(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

        # Learned embedding of the action frame index, so the cross-attention
        # can align render and action tokens in time instead of treating the
        # actions as an unordered set.
        self.action_pos_emb = nn.Parameter(torch.zeros(max_action_frames, hidden_dim))
        nn.init.normal_(self.action_pos_emb, std=0.02)

        # Cross-attention blocks (render Q, action K/V).
        self.cross_blocks = nn.ModuleList([
            CrossAttnTransformerBlock(hidden_dim, num_heads, ffn_mult)
            for _ in range(num_cross_blocks)
        ])

        # 2-layer projection from the fused hidden space up to temb space.
        self.to_temb = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * ffn_mult),
            nn.GELU(),
            nn.Linear(hidden_dim * ffn_mult, inner_dim),
        )

    def forward(
        self,
        render_latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            contribution: (B, num_tokens, inner_dim), where
            ``num_tokens = T * h * w`` matches the DiT post-patch grid.
            Caller adds it to ``temb_base`` via ``combine_with_temb``.
        """
        r = self.render_encoder(render_latents)        # (B, T·h·w, hidden)
        a = self.action_encoder(actions)               # (B, T_act, hidden)

        # Per-frame positional embedding on the action tokens.
        T_act = a.shape[1]
        a = a + self.action_pos_emb[:T_act].unsqueeze(0)

        for block in self.cross_blocks:
            r = block(r, a)                            # render attends over actions

        contribution = self.to_temb(r)                 # (B, T·h·w, inner_dim)
        return contribution

    @staticmethod
    def combine_with_temb(
        temb_base: torch.Tensor,
        contribution: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pre-add the per-token contribution to temb. Broadcasts ``temb_base``
        up to per-token from any of:
          - (B, D) or (B, 1, D): broadcast to all tokens
          - (B, num_tokens, D): already per-token

        ``contribution`` is (B, num_tokens, D).
        """
        num_tokens = contribution.shape[1]
        if temb_base.ndim == 2:
            temb_base = temb_base.unsqueeze(1).expand(-1, num_tokens, -1)
        elif temb_base.shape[1] == 1 and num_tokens != 1:
            temb_base = temb_base.expand(-1, num_tokens, -1)
        return temb_base + contribution


__all__ = [
    "RenderTokenEncoderWithSelfAttn",
    "ActionTokenEncoder",
    "CrossAttnTransformerBlock",
    "SelfAttnTransformerBlock",
    "PositionalEncoding3D",
    "RenderActionAdapter",
]
