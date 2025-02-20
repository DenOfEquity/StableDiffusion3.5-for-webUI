####    THIS IS main BRANCH (only difference - delete transformer after use)

# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Tuple, Optional, Union

import PIL.Image
import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, SD3LoraLoaderMixin
from diffusers.models.attention_processor import PAGCFGJointAttnProcessor2_0, PAGJointAttnProcessor2_0

from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.pag.pag_utils import PAGMixin

#from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from diffusers.models.controlnet_sd3 import SD3ControlNetModel, SD3MultiControlNetModel

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

#logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,                                              # (`SchedulerMixin`): scheduler to get timesteps from.
    num_inference_steps: Optional[int] = None,              # (`int`):            number of diffusion steps used  - priority 3
    device: Optional[Union[str, torch.device]] = None,      # (`str` or `torch.device`, *optional*): device to move timesteps to. If `None`, not moved.
    timesteps: Optional[List[int]] = None,                  # (`List[int]`, *optional*): custom timesteps, length overrides num_inference_steps - priority 1
    sigmas: Optional[List[float]] = None,                   # (`List[float]`, *optional*): custom sigmas, length overrides num_inference_steps - priority 2
    **kwargs,
):
    #   stop aborting on recoverable errors!
    #   default to using timesteps
    if timesteps is not None and "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys()):
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None and "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys()):
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

class SD35Pipeline_DoE_combined (DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin, PAGMixin):
#    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    model_cpu_offload_seq = "transformer->controlnet->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,

        controlnet: Union[
            SD3ControlNetModel, List[SD3ControlNetModel], Tuple[SD3ControlNetModel], SD3MultiControlNetModel
        ],
        pag_applied_layers: Union[str, List[str]] = ["blocks.14"],  # 1st transformer block
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            controlnet=controlnet,
        )

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None
            else 8
        )
        self.latent_channels = (
            self.vae.config.latent_channels
            if hasattr(self, "vae") and self.vae is not None
            else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, vae_latent_channels=self.latent_channels)
        self.mask_processor  = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, vae_latent_channels=self.vae.config.latent_channels, 
                                                 do_resize=False, do_normalize=False, do_binarize=False, do_convert_grayscale=True)

        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )
        self.set_pag_applied_layers(
            pag_applied_layers, pag_attn_processors=(PAGCFGJointAttnProcessor2_0(), PAGJointAttnProcessor2_0())
        )

    def check_inputs(
        self,
        strength,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if strength < 0:
            strength = 0.0
            print ("Warning: value of strength has been clamped to 0.0 from lower")
        elif strength > 1:
            strength = 1.0
            print ("Warning: value of strength has been clamped to 1.0 from higher")
            
        if prompt_embeds == None or negative_prompt_embeds == None or pooled_prompt_embeds == None or negative_pooled_prompt_embeds == None:
            raise ValueError(f"All prompt embeds must be provided.")
            
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    #   controlnet
    @torch.no_grad()
    def prepare_image(
        self,
        image,
        num_images_per_prompt,
        device,
        dtype,
    ):
        image = self.image_processor.preprocess(image).to(device=device, dtype=dtype)
        image = self.vae.encode(image).latent_dist.sample()
        image = (image - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        image = image.repeat_interleave(num_images_per_prompt, dim=0)

        return image
    @torch.no_grad()
    def prepare_image_with_mask(
        self,
        image,
        mask,
        num_images_per_prompt,
        device,
        dtype,
    ):
        image = self.image_processor.preprocess(image).to(device=device, dtype=dtype)
        mask = self.mask_processor.preprocess(mask).to(device=device, dtype=dtype)

        # Get masked image
        masked_image = image.clone()
        masked_image[(mask > 0.5).repeat(1,3,1,1)] = -1

        # Encode to latents
        image_latents = self.vae.encode(masked_image).latent_dist.sample()
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = image_latents.to(dtype)

        mask = torch.nn.functional.interpolate(
            mask, size = (image_latents.size(2), image_latents.size(3))
        )
        mask = 1 - mask
        
        image_latents = image_latents.repeat_interleave(num_images_per_prompt, dim=0)
        mask = mask.repeat_interleave(num_images_per_prompt, dim = 0)
 
        control_image = torch.cat([image_latents, mask], dim = 1)

        return control_image


    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        strength: float = 0.6,
        mask_cutoff: float = 1.0,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        guidance_rescale: float = 0.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],

        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        control_image: PipelineImageInput = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        controlnet_pooled_projections: Optional[torch.FloatTensor] = None,

        pag_scale: float = 3.0,
        pag_adaptive_scale: float = 0.0,

        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):

        doDiffDiff = True if (image and mask_image) else False

        # 0.01 repeat prompt embeds to match num_images_per_prompt
        prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1, 1)
        negative_prompt_embeds = negative_prompt_embeds.repeat(num_images_per_prompt, 1, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(num_images_per_prompt, 1)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(num_images_per_prompt, 1)
        

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            strength,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        self._pag_scale = pag_scale
        self._pag_adaptive_scale = pag_adaptive_scale

        do_classifier_free_guidance = (guidance_scale > 1.0)

        # 2. Define call parameters
        device = self._execution_device
        dtype = self.transformer.dtype


        if self.do_perturbed_attention_guidance:
            prompt_embeds = self._prepare_perturbed_attention_guidance(
                prompt_embeds, negative_prompt_embeds, do_classifier_free_guidance
            )
            pooled_prompt_embeds = self._prepare_perturbed_attention_guidance(
                pooled_prompt_embeds, negative_pooled_prompt_embeds, do_classifier_free_guidance
            )
        elif do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        components_multipler = 1
        if self.do_perturbed_attention_guidance:
            components_multipler += 1
        if do_classifier_free_guidance:
            components_multipler += 1


        # 3. Prepare control image
        if isinstance(self.controlnet, SD3ControlNetModel):
            if self.controlnet.config.extra_conditioning_channels == 1:    #   using inpaint controlnet, using control_image and mask_image
                control_image = self.prepare_image_with_mask(
                    image=control_image,
                    mask=mask_image,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=dtype,
                )
            else:
                control_image = self.prepare_image(
                    image=control_image,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=dtype,
                )
#        elif isinstance(self.controlnet, SD3MultiControlNetModel):  #   not used at present
#            control_images = []
#            for control_image_ in control_image:
#                control_image_ = self.prepare_image(
#                    image=control_image_,
#                    num_images_per_prompt=num_images_per_prompt,
#                    device=device,
#                    dtype=dtype,
#                )
#                control_images.append(control_image_)
#            control_image = control_images

        if self.controlnet != None:
            if controlnet_pooled_projections is None:
                controlnet_pooled_projections = torch.zeros_like(pooled_prompt_embeds)
            else:
                controlnet_pooled_projections = pooled_prompt_embeds
            control_image = torch.cat([control_image] * components_multipler)

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)


        if image is not None:
            noise = latents

            # 4. Prepare timesteps
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

            # 3. Preprocess image
            image = self.image_processor.preprocess(image).to(device='cuda', dtype=torch.float16)
            image_latents = self.vae.encode(image).latent_dist.sample(generator)
            image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            image_latents = image_latents.repeat(num_images_per_prompt, 1, 1, 1)

            if strength < 1.0:
                latent_timestep = timesteps[:1].repeat(num_images_per_prompt)# * num_inference_steps)
                latents = self.scheduler.scale_noise(image_latents, latent_timestep, noise)

            latents = latents.to(device='cuda', dtype=torch.float16)
            image_latents = image_latents.to(device='cuda', dtype=torch.float16)
            noise = noise.to(device='cuda', dtype=torch.float16)

            if mask_image is not None:
                # 5.1. Prepare masked latent variables
                w = latents.size(3)
                h = latents.size(2)
                mask = self.mask_processor.preprocess(mask_image.resize((w,h))).to(device='cuda', dtype=torch.float16)


        if self.do_perturbed_attention_guidance:
            original_attn_proc = self.transformer.attn_processors
            self._set_pag_attn_processor(
                pag_applied_layers=self.pag_applied_layers,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )

        # 6. Denoising loop
        num_timesteps = len(timesteps)
        num_warmup_steps = max(num_timesteps - num_inference_steps * self.scheduler.order, 0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                thisStep = float((i+1) / num_timesteps)

                if doDiffDiff and thisStep <= mask_cutoff:# and i > 0 :
                    tmask = (mask >= thisStep)
                    init_latents_proper = self.scheduler.scale_noise(image_latents, torch.tensor([t]), noise)
                    latents = (init_latents_proper * ~tmask) + (latents * tmask)

                latent_model_input = torch.cat([latents] * components_multipler)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                if self.controlnet != None and thisStep >= control_guidance_start and thisStep <= control_guidance_end:
#                    if self.controlnet.config.extra_conditioning_channels == 1: #   inpaint model is large
#                        self.transformer.to('cpu')
#                        self.controlnet.to('cuda')
                    # controlnet inference
                    control_block_samples = self.controlnet(
                        hidden_states               =   latent_model_input,
                        timestep                    =   timestep,
                        encoder_hidden_states       =   prompt_embeds,
                        pooled_projections          =   controlnet_pooled_projections,
                        joint_attention_kwargs      =   None,#self.joint_attention_kwargs,   #for 'scale', default set to 1.0 - but scale used by LoRAs
                        controlnet_cond             =   control_image,
                        conditioning_scale          =   controlnet_conditioning_scale,
                        return_dict                 =   False,
                    )[0]
#                    if self.controlnet.config.extra_conditioning_channels == 1:
#                        self.controlnet.to('cpu')
#                        self.transformer.to('cuda')
                else:
                    control_block_samples = None

                noise_pred = self.transformer(
                    hidden_states                   =   latent_model_input,
                    timestep                        =   timestep,
                    encoder_hidden_states           =   prompt_embeds,
                    pooled_projections              =   pooled_prompt_embeds,
                    block_controlnet_hidden_states  =   control_block_samples,
                    joint_attention_kwargs          =   self.joint_attention_kwargs,
                    return_dict                     =   False,
                )[0]

                # perform guidance
                if self.do_perturbed_attention_guidance:
                    noise_pred = self._apply_perturbed_attention_guidance(
                        noise_pred, do_classifier_free_guidance, guidance_scale, t
                    )

                elif do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    if guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                ### interrupt ?

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if (i == num_timesteps - 1) or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        #unsure about this? leaves vae roundtrip error, maybe better for quality to keep last step processing
        if doDiffDiff and 1.0 <= mask_cutoff:
            tmask = (mask >= 1.0)
            latents = (image_latents * ~tmask) + (latents * tmask)

        # Offload all models
        self.maybe_free_model_hooks()

        if self.do_perturbed_attention_guidance:
            self.transformer.set_attn_processor(original_attn_proc)

        return latents
