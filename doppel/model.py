"""
Wan 2.1 image-to-video components shared by the conditioned models (Diffusers).

* :class:`WanVAEChunkedEncode`: Wan VAE with temporally chunked encoding, used
  for both the target videos and the robot renderings.
* :class:`WanTransformerConditioned`: base class of the conditioned DiTs,
  the render-action adapter model (``model_redar.py``) and the action-only
  baseline (``model_action_only.py``).
* :class:`RenderConditionedWanI2VPipeline`: I2V sampling with a conditioned DiT.

The target video the DiT denoises is the real camera video; the conditioning
(robot renderings and/or numerical actions) enters through the timestep
modulation.

Weights layout: Hugging Face Diffusers Wan I2V repo with ``transformer/``,
``vae/``, ``text_encoder/``, ``tokenizer/``, ``image_encoder/``,
``image_processor/``, ``scheduler/``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.image_processor import PipelineImageInput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.utils.import_utils import is_torch_xla_available

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class WanVAEChunkedEncode(AutoencoderKLWan):
    """Wan VAE with chunked causal encoding, used to build the latent store.

    ``_encode`` runs the causal encoder on the first frame alone and then on
    consecutive 4-frame chunks, carrying the causal feature cache from chunk
    to chunk, and concatenates the chunk latents along time: ``T`` frames give
    ``1 + (T - 1) // 4`` latent frames. The target videos, the robot
    renderings and the I2V condition videos in the latent store were encoded
    with it (``latent_store.WanStoreEncoders``).
    """

    def encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _encode(self, x: torch.Tensor):
        _, _, num_frame, height, width = x.shape
        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)

        self.clear_cache()
        iter_ = 1 + (num_frame - 1) // 4
        out = None
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                chunk = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                chunk = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            out = chunk if out is None else torch.cat([out, chunk], dim=2)

        enc = self.quant_conv(out)
        self.clear_cache()
        return enc


class WanTransformerConditioned(WanTransformer3DModel):
    """Base class of the conditioned Wan 2.1 I2V DiTs.

    Subclasses add a conditioning adapter whose per-token output ``g_j`` is
    added to the flow-timestep embedding ``e(s)`` before the backbone's
    ``time_proj`` (per-token timestep modulation), so each token receives its
    own shift/scale/gate in every block and in the final ``norm_out``:

      * ``RenderActionWanTransformer`` (``model_redar.py``): the
        render-action adapter over robot renderings and numerical actions.
      * ``ActionOnlyWanTransformer`` (``model_action_only.py``): the
        action-only baseline following EgoWM.

    Without conditioning inputs, the subclasses run this class's unmodified
    Wan I2V forward.
    """

    _keep_in_fp32_modules = [
        "time_embedder",
        "time_proj",
        "scale_shift_table",
        "norm_out",
    ]


def _normalize_latents(z: torch.Tensor, vae) -> torch.Tensor:
    mean = torch.tensor(vae.config.latents_mean, device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std, device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
    return (z - mean) * inv_std


class RenderConditionedWanI2VPipeline(WanImageToVideoPipeline):
    """Wan 2.1 I2V sampling with a conditioned DiT.

    ``render_latents`` (or a ``render_video`` to encode) and ``actions`` are
    passed to the transformer at every denoising step. As in training, the
    first temporal latent slice is kept pinned to the clean VAE latent of the
    conditioning image.
    """

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale_2=None,
        render_latents=None,
    ):
        super().check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )
        if render_latents is not None:
            if render_latents.ndim != 5:
                raise ValueError(
                    f"`render_latents` must be (B, C, T_lat, H_lat, W_lat), got shape {tuple(render_latents.shape)}"
                )

    def _encode_render_video(
        self,
        render_video,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Preprocess + VAE-encode a render video to normalized latents."""
        vid = self.video_processor.preprocess_video(render_video, height=height, width=width)
        if vid.shape[2] != num_frames:
            import numpy as np
            idxs = np.linspace(0, vid.shape[2] - 1, num_frames).round().astype(int)
            vid = vid[:, :, idxs]
        vid = vid.to(device=device, dtype=self.vae.dtype)
        z = self.vae.encode(vid).latent_dist.sample()
        z = _normalize_latents(z, self.vae)
        return z.to(dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        render_video: Optional[Any] = None,
        render_latents: Optional[torch.Tensor] = None,
        drop_render_conditioning: bool = False,
        actions: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        if self.config.expand_timesteps or self.config.boundary_ratio is not None or self.transformer_2 is not None:
            raise NotImplementedError(
                "RenderConditionedWanI2VPipeline supports Wan 2.1-style I2V only "
                "(single transformer, expand_timesteps=False, boundary_ratio=None)."
            )
        if not drop_render_conditioning and render_latents is None and render_video is None:
            raise ValueError(
                "Pass `render_latents` (B, C, T_lat, H_lat, W_lat) or `render_video` (PIL/ndarray frames), "
                "or set `drop_render_conditioning=True` for vanilla I2V (no render branch)."
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        eff_bs = batch_size * num_videos_per_prompt
        transformer_dtype = self.transformer.dtype

        if drop_render_conditioning:
            render_latents = None
        elif render_latents is None:
            render_latents = self._encode_render_video(
                render_video, height, width, num_frames, device, transformer_dtype
            )
        else:
            render_latents = render_latents.to(device=device, dtype=transformer_dtype)

        if render_latents is not None:
            if render_latents.shape[0] == 1 and eff_bs > 1:
                render_latents = render_latents.expand(eff_bs, -1, -1, -1, -1)
            elif render_latents.shape[0] != eff_bs:
                raise ValueError(
                    f"render_latents batch {render_latents.shape[0]} != batch_size*num_videos_per_prompt={eff_bs}"
                )

        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
            render_latents,
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # encode_prompt only expands by num_videos_per_prompt when it computes
        # the embeddings itself; pre-supplied ``prompt_embeds`` must already
        # match ``latents`` (batch eff_bs).
        if prompt_embeds.shape[0] != eff_bs:
            raise ValueError(
                f"prompt_embeds batch {prompt_embeds.shape[0]} != eff_bs={eff_bs} "
                f"(batch_size*num_videos_per_prompt). When pre-supplying "
                f"`prompt_embeds`, expand it to batch={eff_bs} before calling."
            )
        if negative_prompt_embeds is not None and negative_prompt_embeds.shape[0] != eff_bs:
            raise ValueError(
                f"negative_prompt_embeds batch {negative_prompt_embeds.shape[0]} "
                f"!= eff_bs={eff_bs}. When pre-supplying "
                f"`negative_prompt_embeds`, expand it to batch={eff_bs} before calling."
            )

        if self.transformer is not None and self.transformer.config.image_dim is not None:
            if image_embeds is None:
                if last_image is None:
                    image_embeds = self.encode_image(image, device)
                else:
                    image_embeds = self.encode_image([image, last_image], device)
            # expand to eff_bs (encode_image returns batch 1)
            cur = image_embeds.shape[0]
            if cur == 1:
                image_embeds = image_embeds.repeat(eff_bs, 1, 1)
            elif cur != eff_bs:
                if eff_bs % cur != 0:
                    raise ValueError(
                        f"image_embeds batch {cur} cannot be broadcast to "
                        f"eff_bs={eff_bs} (batch_size*num_videos_per_prompt)."
                    )
                image_embeds = image_embeds.repeat(eff_bs // cur, 1, 1)
            image_embeds = image_embeds.to(transformer_dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.vae.config.z_dim
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        if last_image is not None:
            last_image = self.video_processor.preprocess(last_image, height=height, width=width).to(
                device, dtype=torch.float32
            )

        latents, condition = self.prepare_latents(
            image,
            eff_bs,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
            last_image,
        )

        # As in training, latent frame 0 is the clean latent of the first frame
        # (encoded with a zero tail, like the I2V condition video), re-pinned after every step.
        img_t = image.unsqueeze(2).to(device=device, dtype=self.vae.dtype)
        b_img = img_t.shape[0]
        if b_img == 1 and eff_bs > 1:
            img_t = img_t.expand(eff_bs, -1, -1, -1, -1)
        elif b_img != eff_bs:
            raise ValueError(
                f"preprocessed image batch {b_img} must be 1 (broadcast to eff_bs={eff_bs}) "
                f"or equal to eff_bs."
            )
        vid_pad = torch.cat(
            [img_t, img_t.new_zeros(eff_bs, img_t.shape[1], num_frames - 1, height, width)],
            dim=2,
        )
        z_first = self.vae.encode(vid_pad).latent_dist.sample()
        latents_first = _normalize_latents(z_first, self.vae).to(dtype=latents.dtype)[:, :, 0:1, :, :].contiguous()
        latents[:, :, 0:1] = latents_first

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    render_latents=render_latents,
                    actions=actions,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        render_latents=render_latents,
                        actions=actions,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_uncond + self.guidance_scale * (noise_pred - noise_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                latents[:, :, 0:1] = latents_first

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
