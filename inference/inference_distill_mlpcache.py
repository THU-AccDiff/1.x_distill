import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import torch
import torch.nn as nn
from safetensors.torch import load_file

from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput

ROOT_PATH = str(Path(__file__).parent.parent.absolute())
sys.path.append(ROOT_PATH)
from train.processor import DistillProcessor
from train.train_cache_utils import DeltaMLP



def attach_and_load_delta_pred(
    transformer: SD3Transformer2DModel,
    transformer_dir: str,
    mlp_hidden_mult: int = 2,
    mlp_dropout: float = 0.0,
) -> None:
    """
    Minimal and robust:
    1) Create delta_pred_hidden / delta_pred_encoder modules on transformer
    2) Load their weights from transformer_dir/diffusion_pytorch_model.safetensors
    """
    
    def _strip_common_prefix(k: str) -> str:
        """
        Strip common wrappers that may appear in saved state_dict keys.
        """
        prefixes = ["_fsdp_wrapped_module.", "module.", "transformer."]
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p):]
                    changed = True
        return k

    weight_path = os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Missing weights: {weight_path}")

    # Create modules
    hidden_dim = getattr(transformer, "inner_dim", None)
    if hidden_dim is None:
        # Fallback: derive from config if needed (rare)
        raise RuntimeError("Cannot find transformer.inner_dim; please check your diffusers version/model.")

    enc_dim = getattr(transformer.config, "caption_projection_dim", None)
    if enc_dim is None:
        raise RuntimeError("Cannot find transformer.config.caption_projection_dim; please check your model config.")

    transformer.enable_delta_pred = True
    transformer.delta_pred_hidden = DeltaMLP(dim=hidden_dim, hidden_dim=hidden_dim * mlp_hidden_mult, dropout=mlp_dropout)
    transformer.delta_pred_encoder = DeltaMLP(dim=enc_dim, hidden_dim=enc_dim * mlp_hidden_mult, dropout=mlp_dropout)

    # Load only the delta_pred_* keys
    sd = load_file(weight_path)  # dict[str, Tensor]
    hidden_sd: Dict[str, torch.Tensor] = {}
    enc_sd: Dict[str, torch.Tensor] = {}

    for k, v in sd.items():
        k2 = _strip_common_prefix(k)

        if k2.startswith("delta_pred_hidden."):
            sub = k2[len("delta_pred_hidden.") :]
            hidden_sd[sub] = v
        elif k2.startswith("delta_pred_encoder."):
            sub = k2[len("delta_pred_encoder.") :]
            enc_sd[sub] = v

    if len(hidden_sd) == 0 and len(enc_sd) == 0:
        print("[WARN] No delta_pred_* weights found in safetensors. "
              "Your inference will run, but MLP will stay at init (near-identity).")
        return

    miss_h, unexp_h = transformer.delta_pred_hidden.load_state_dict(hidden_sd, strict=False)
    miss_e, unexp_e = transformer.delta_pred_encoder.load_state_dict(enc_sd, strict=False)

    if len(miss_h) or len(unexp_h) or len(miss_e) or len(unexp_e):
        print("[WARN] delta_pred load_state_dict report:")
        if len(miss_h): print("  hidden missing:", miss_h)
        if len(unexp_h): print("  hidden unexpected:", unexp_h)
        if len(miss_e): print("  encoder missing:", miss_e)
        if len(unexp_e): print("  encoder unexpected:", unexp_e)

    print("[OK] Loaded delta_pred_hidden / delta_pred_encoder weights from:", weight_path)


# -------------------------
# Patched forward: cacheblock + delta_pred usage 
# -------------------------
def cache_forward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    pooled_projections: torch.FloatTensor = None,
    timestep: torch.LongTensor = None,
    block_controlnet_hidden_states: List = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
    delta_cache: Optional[Dict[str, torch.Tensor]] = None,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    """
    Return tuple when return_dict=False:
      (output, delta_cache_out, delta_pred)
    """
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    height, width = hidden_states.shape[-2:]

    hidden_states = self.pos_embed(hidden_states)
    temb = self.time_text_embed(timestep, pooled_projections)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    delta_cache_out = None
    delta_pred = None


    if getattr(self, "enable_cachestep", False):
        cache_step = []
        if getattr(self, "cnt", 0) in cache_step and getattr(self, "previous_residual", None) is not None:
            should_calc = False
        else:
            should_calc = True

        self.cnt = getattr(self, "cnt", 0) + 1
        if self.cnt == getattr(self, "num_steps", 0):
            self.cnt = 0

        if not should_calc:
            hidden_states = hidden_states + self.previous_residual
        else:
            ori_hidden_states = hidden_states.clone()
            for index_block, block in enumerate(self.transformer_blocks):
                if self.training and self.gradient_checkpointing:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)
                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
                    )

                if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                    interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                    hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

            self.previous_residual = hidden_states - ori_hidden_states

    else:
        cache_block = getattr(self, "cache_block", [])

        for index_block, block in enumerate(self.transformer_blocks):
            use_cache_here = bool(
                getattr(self, "enable_cacheblock", False)
                and (delta_cache is not None)
                and (index_block in cache_block)
            )

            if use_cache_here:
                # Only inject once at the last block in cache_block
                if len(cache_block) > 0 and index_block == cache_block[-1]:
                    dh = delta_cache["hidden"]
                    deh = delta_cache.get("encoder_hidden", None)

                    if getattr(self, "enable_delta_pred", False):
                        dh = self.delta_pred_hidden(dh)
                        if deh is not None:
                            deh = self.delta_pred_encoder(deh)
                        delta_pred = {"hidden": dh}
                        if deh is not None:
                            delta_pred["encoder_hidden"] = deh
                        delta_cache_out = delta_pred
                    else:
                        delta_cache_out = delta_cache

                    hidden_states = hidden_states + dh
                    if deh is not None:
                        encoder_hidden_states = encoder_hidden_states + deh

                # Skip the real block compute when cached
                continue

            # Normal block compute
            if len(cache_block) > 0 and index_block == cache_block[0]:
                inp_hidden = hidden_states.clone()
                inp_encoder_hidden = encoder_hidden_states.clone()

            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
                )

            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

            # Compute delta_cache_out at the end of cached range (teacher / true delta)
            if len(cache_block) > 0 and index_block == cache_block[-1]:
                block_delta_hid = hidden_states - inp_hidden
                block_delta_enhid = None if (encoder_hidden_states is None or inp_encoder_hidden is None) else (encoder_hidden_states - inp_encoder_hidden)

                delta_cache_out = {"hidden": block_delta_hid}
                if block_delta_enhid is not None:
                    delta_cache_out["encoder_hidden"] = block_delta_enhid

    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    # Unpatchify
    patch_size = self.config.patch_size
    height = height // patch_size
    width = width // patch_size

    hidden_states = hidden_states.reshape(
        shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
    )
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(
        shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
    )

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output, delta_cache_out, delta_pred)

    return Transformer2DModelOutput(sample=output)


# -------------------------
# Inference pipeline
# -------------------------
class InferencePipeline:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        transformer_path: Optional[str],
        device: torch.device,
        dtype: torch.dtype,
        generate_img_size: int = 1024,
        cfg: Optional[float] = 1.0,
        num_inference_steps: int = 2,
        max_t5_sequence_length: int = 256,
        student_shift: Optional[float] = None,
        student_sigmas: Optional[list] = None,
        cache_block: Optional[List[int]] = None,
        enable_cacheblock: bool = True,
        enable_delta_pred: bool = True,
    ):
        self.device = device
        self.dtype = dtype
        self.guidance = cfg
        self.num_inference_steps = num_inference_steps
        self.max_t5_sequence_length = max_t5_sequence_length
        self.generate_img_size = generate_img_size
        self.generator = torch.Generator(device=device)

        # Load SD3 pipeline for VAE / text encoder / scheduler
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=self.dtype,
        ).to(self.device)

        self.vae = self.pipe.vae
        self.vae_scale_factor = self.pipe.vae_scale_factor

        assert isinstance(self.pipe.scheduler, FlowMatchEulerDiscreteScheduler)
        self.noise_scheduler: FlowMatchEulerDiscreteScheduler = self.pipe.scheduler

        # Load transformer
        if transformer_path is None:
            self.transformer = self.pipe.transformer
            transformer_dir = None
        else:
            self.transformer = SD3Transformer2DModel.from_pretrained(
                transformer_path,
                torch_dtype=self.dtype,
            )
            transformer_dir = transformer_path

        # Patch forward
        SD3Transformer2DModel.forward = cache_forward

        # Cache flags and range
        self.transformer.__class__.enable_cachestep = False
        self.transformer.__class__.enable_cacheblock = enable_cacheblock
        self.transformer.__class__.cache_block = cache_block or [2, 3, 4, 5, 6, 7]

        # Enable delta predictor modules and load weights
        self.transformer.enable_delta_pred = enable_delta_pred
        if enable_delta_pred:
            if transformer_dir is None:
                print("[WARN] transformer_path is None. No checkpoint dir to load delta_pred weights from.")
            else:
                attach_and_load_delta_pred(
                    transformer=self.transformer,
                    transformer_dir=transformer_dir,
                    mlp_hidden_mult=2,
                    mlp_dropout=0.0,
                )
        self.transformer.to(self.device, dtype=self.dtype).eval()
        
        # Latent shape
        self.latent_h = self.generate_img_size // self.vae_scale_factor
        self.latent_w = self.generate_img_size // self.vae_scale_factor
        self.latent_channels = self.transformer.config.in_channels

        # CFG switch
        self.do_classifier_free_guidance = (self.guidance is not None) and (self.guidance > 1.0)

        # Build sigmas/timesteps from processor
        processor = DistillProcessor(
            teacher_training_steps=self.noise_scheduler.config.num_train_timesteps,
            shift=self.noise_scheduler.config.shift,
            device=self.device,
            dtype=self.dtype,
            sampling_steps_list=[num_inference_steps],
            student_shift=student_shift,
            student_sigmas=student_sigmas,
        )
        self.sigmas = processor.student_sample_info_dict[num_inference_steps].sigmas
        self.timesteps = self.sigmas[:-1] * self.noise_scheduler.config.num_train_timesteps

        print("sigmas:", self.sigmas.dtype, self.sigmas.shape, self.sigmas)
        print("timesteps:", self.timesteps.dtype, self.timesteps.shape, self.timesteps)

    @torch.inference_mode()
    def infer(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int = 42,
        img_output_path: Optional[str] = None,
        prefix: str = "",
    ) -> Optional[Image.Image]:

        # Encode text
        prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds = self.pipe.encode_prompt(
            prompt=[prompt],
            prompt_2=None,
            prompt_3=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=[negative_prompt] if self.do_classifier_free_guidance else None,
            negative_prompt_2=None,
            negative_prompt_3=None,
            max_sequence_length=self.max_t5_sequence_length,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # Init noise
        self.generator.manual_seed(seed)
        latents = torch.randn(
            (1, self.latent_channels, self.latent_h, self.latent_w),
            device=self.device,
            dtype=self.dtype,
            generator=self.generator,
        )

        # Maintain delta_cache across steps
        delta_cache = None

        for i, t in enumerate(self.timesteps):
            # Prepare model input
            if self.do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2, dim=0)
            else:
                latent_model_input = latents

            timestep = t.expand(latent_model_input.shape[0])

            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                out, delta_cache_out, _delta_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                    delta_cache=delta_cache,
                )

            pred = out
            if self.do_classifier_free_guidance:
                pred_uncond, pred_text = pred.chunk(2)
                pred = pred_uncond + self.guidance * (pred_text - pred_uncond)

            # Update cache for next step
            delta_cache = delta_cache_out

            current_sigma = self.sigmas[i]
            next_sigma = self.sigmas[i + 1]
            pred_dtype = pred.dtype

            latents = latents.to(torch.float32)
            pred = pred.to(torch.float32)
            latents = latents + (next_sigma - current_sigma) * pred
            latents = latents.to(pred_dtype)

        # Decode VAE
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            image = self.vae.decode(latents, return_dict=False)[0]

        image = image.to(torch.float32)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        pil_image = Image.fromarray(image[0])

        if img_output_path is None:
            return pil_image

        os.makedirs(img_output_path, exist_ok=True)
        save_name = prefix + format_file_name(prompt.strip())
        save_path = os.path.join(img_output_path, save_name + ".png")
        pil_image.save(save_path, quality=100)
        return None

def format_file_name(s: str) -> str:
    s = (
        s.replace(" ", "_")
        .replace(".", "")
        .replace(",", "")
        .replace(":", "")
        .replace('"', "_")
        .replace("/", "_")
    )
    return s[:100]

def main():
    # Replace this with your local path of orignal teacher model. Need to load VAE / text encoder from original model
    pretrained_model_name_or_path = "/cache/stable-diffusion-3-medium-diffusers"
    
    cache_block=[2, 3, 4, 5, 6, 7]
    # Replace this with your local path of distilled model transformer with mlpcache
    transformer_path = "/cache/stage2_mlpcache"

    prompts_path = "dataset/test.txt"
    img_output_path = f"outputs/test"

    num_inference_steps = 2
    seeds = [42]

    generate_img_size = 1024
    device = torch.device("cuda:0")
    dtype = torch.float16

    with open(prompts_path, "rt", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    pipeline = InferencePipeline(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        transformer_path=transformer_path,
        generate_img_size=generate_img_size,
        device=device,
        dtype=dtype,
        cfg=1.0,  
        num_inference_steps=num_inference_steps,
        max_t5_sequence_length=256,
        student_shift=3,
        student_sigmas=None,
        cache_block=cache_block,
        enable_cacheblock=True,
        enable_delta_pred=True,
    )

    for idx, prompt in enumerate(prompts):
        print(f"[{idx}/{len(prompts)}] {prompt}")
        for seed in seeds:
            img_output_path = img_output_path
            pipeline.infer(
                prompt=prompt,
                negative_prompt="",
                seed=seed,
                img_output_path=img_output_path,
                prefix=f"{str(idx).rjust(4, '0')}_seed{seed}_",
            )


if __name__ == "__main__":
    main()
