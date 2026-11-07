"""Render-free action conditioning for the action-only baseline.

``EgoWMActionAdapter`` follows the conditioning mechanism of EgoWM
("Walk through Paintings: Egocentric World Models from Internet Priors",
Bagchi et al., arXiv:2601.15284): a pretrained video diffusion model is
conditioned on the initial frame plus an action sequence, with no rendered
robot, and the action embedding is injected at every timestep-modulation site::

    P_i^{scale}, P_i^{shift}, P_i^{gate} = F_i(Z_ts + Z_a)          (Eq. 5)
    P_i^{scale}, P_i^{shift}, P_i^{gate} = F_i(Z_ts + Z_a + Z_s)    (Eq. 6, optional)

Z_a is produced by lightweight MLPs and is **per video-latent frame**,
downsampled to the latent frame rate by a 1D convolution. Z_s is an embedding of
the initial agent state, which EgoWM adds only for embodiments whose body leaves
the camera view; it is off by default here (``use_initial_state=False``).

Initialisation: diffusers' ``from_pretrained`` runs model construction inside
``no_init_weights()``, which replaces every ``torch.nn.init`` function with a
no-op, and leaves parameters absent from the checkpoint on the meta device. The
adapter does not appear in the Wan checkpoint, so the init written in __init__
never runs under that path. The adapter therefore exposes :meth:`init_weights`,
which materialises meta parameters and applies the intended initialisation; the
model wrapper calls it after loading.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from doppel.redar_adapter import RenderActionAdapter, _AdapterInitMixin


class EgoWMActionAdapter(_AdapterInitMixin, nn.Module):
    """EgoWM Eq. 5 conditioning on actions alone.

    ``actions (B, T_act, action_dim)``
      -> per-frame MLP                      -> ``(B, T_act, hidden)``
      -> 1D conv downsample to latent rate  -> ``(B, T_lat, hidden)``
      -> MLP                                -> ``(B, T_lat, inner_dim)``  = Z_a
      -> broadcast across each frame's tokens -> ``(B, T_lat*H*W, inner_dim)``

    The temporal downsample follows the Wan VAE's own layout rather than a generic
    stride: latent frame 0 comes from video frame 0, and latent frame t from video
    frames ``4t-3 .. 4t``. So video frame 0 gets its own projection and the
    remaining ``4*(T_lat-1)`` frames go through a stride-4 convolution. A generic
    resample would smear frame 0, which is the one frame the model is *given*.
    """

    def __init__(
        self,
        action_dim: int,
        inner_dim: int = 5120,
        hidden_dim: int = 512,
        use_initial_state: bool = False,
    ) -> None:
        super().__init__()
        self.inner_dim = inner_dim
        self.hidden_dim = hidden_dim
        self.use_initial_state = use_initial_state

        self.encoder = nn.Sequential(                 # "lightweight MLPs", per frame
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head_proj = nn.Linear(hidden_dim, hidden_dim)          # video frame 0
        self.temporal = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=4)
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, inner_dim))
        if use_initial_state:                                        # Eq. 6, Z_s
            self.state_encoder = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, inner_dim),
            )

    def encode_per_frame(self, actions: torch.Tensor, num_latent_frames: int) -> torch.Tensor:
        """``(B, T_act, action_dim)`` -> ``(B, num_latent_frames, inner_dim)`` = Z_a."""
        x = self.encoder(actions)
        head, rest = x[:, :1], x[:, 1:]
        expected = 4 * (num_latent_frames - 1)
        if rest.shape[1] == expected and expected > 0:
            z = self.temporal(rest.transpose(1, 2)).transpose(1, 2)
        elif expected == 0:
            z = rest[:, :0]
        else:
            # Action stream does not match the VAE grid (e.g. a resampled clip):
            # linearly resample to the expected length before the convolution.
            z = torch.nn.functional.interpolate(
                rest.transpose(1, 2), size=expected, mode="linear", align_corners=False
            )
            z = self.temporal(z).transpose(1, 2)
        z = torch.cat([self.head_proj(head), z], dim=1)             # (B, T_lat, hidden)
        z_a = self.out(z)                                            # (B, T_lat, inner_dim)
        if self.use_initial_state:
            z_a = z_a + self.state_encoder(actions[:, :1])           # broadcast Z_s
        return z_a

    def forward(
        self,
        actions: torch.Tensor,
        num_post_patch_frames: int,
        post_patch_height: int,
        post_patch_width: int,
    ) -> torch.Tensor:
        """Returns the per-token contribution ``(B, T*H*W, inner_dim)``."""
        z_a = self.encode_per_frame(actions, num_post_patch_frames)
        if z_a.shape[1] != num_post_patch_frames:
            raise RuntimeError(
                f"action embedding has {z_a.shape[1]} latent frames but the DiT "
                f"expects {num_post_patch_frames}"
            )
        tokens_per_frame = post_patch_height * post_patch_width
        contribution = z_a.repeat_interleave(tokens_per_frame, dim=1)
        return contribution

    @staticmethod
    def combine_with_temb(temb_base: torch.Tensor, contribution: torch.Tensor) -> torch.Tensor:
        return RenderActionAdapter.combine_with_temb(temb_base, contribution)


__all__ = ["EgoWMActionAdapter"]
