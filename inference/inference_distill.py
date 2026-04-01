import os
import sys
from pathlib import Path
from typing import Optional
from PIL import Image
import torch
import numpy as np
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler

ROOT_PATH = str(Path(__file__).parent.parent.absolute())
sys.path.append(ROOT_PATH)
from train.processor import DistillProcessor


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
        student_sigmas: Optional[list] = None):
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
            )
        self.pipe.to(self.device)

        # VAE & scale factor
        self.vae = self.pipe.vae
        self.vae_scale_factor = self.pipe.vae_scale_factor

        # Scheduler
        assert isinstance(self.pipe.scheduler, FlowMatchEulerDiscreteScheduler)
        self.noise_scheduler: FlowMatchEulerDiscreteScheduler = self.pipe.scheduler

        # ============ Transformer ============
        if transformer_path is None:
            self.transformer = self.pipe.transformer
        else:
            self.transformer = SD3Transformer2DModel.from_pretrained(
                transformer_path,
                torch_dtype=self.dtype,
                )

        self.transformer.to(self.device, dtype=self.dtype)
        self.transformer.eval()

        self.latent_h = self.generate_img_size // self.vae_scale_factor
        self.latent_w = self.generate_img_size // self.vae_scale_factor
        self.latent_channels = self.transformer.config.in_channels  # 一般 16

        self.do_classifier_free_guidance = self.guidance is not None and self.guidance > 1.0
        
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
        print("sigmas info: ", self.sigmas.dtype, self.sigmas.shape, self.sigmas)
        print("timesteps info: ", self.timesteps.dtype, self.timesteps.shape, self.timesteps)
        

    @torch.inference_mode()
    def infer(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int = 42,
        img_output_path: Optional[str] = None,
        prefix: str = "",
        save_denoising: bool = False,
    ) -> Optional[Image.Image]:

        with torch.no_grad():
            prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds = (
                self.pipe.encode_prompt(
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
            )
 
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        self.generator.manual_seed(seed)
        latents = torch.randn(
            (1, self.latent_channels, self.latent_h, self.latent_w),
            device=self.device,
            dtype=self.dtype,
            generator=self.generator,
        )


        for i, t in enumerate(self.timesteps):
            if self.do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2, dim=0)
            else:
                latent_model_input = latents
            timestep = t.expand(latent_model_input.shape[0]) 
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

            if self.do_classifier_free_guidance:
                pred_uncond, pred_text = pred.chunk(2)
                pred = pred_uncond + self.guidance * (pred_text - pred_uncond)

            current_sigma = self.sigmas[i]
            next_sigma = self.sigmas[i + 1]
            dtype = pred.dtype
            latents = latents.to(torch.float32)
            pred = pred.to(torch.float32)
            if save_denoising:
                temp_latents = latents + (self.sigmas[-1] - current_sigma) * pred
                temp_latents = temp_latents.to(dtype)
                temp_latents = temp_latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
            latents = latents + (next_sigma - current_sigma) * pred
            latents = latents.to(dtype)

            if save_denoising:
                assert img_output_path is not None
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    image = self.vae.decode(temp_latents, return_dict=False)[0] # [1, 3, H, W]

                image = image.to(torch.float32)
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                image = (image * 255).round().astype("uint8")
                pil_image = Image.fromarray(image[0])

                os.makedirs(img_output_path, exist_ok=True)
                save_name = prefix + format_file_name(prompt.strip())
                save_path = os.path.join(img_output_path, save_name + f"_step{i}.png")
                pil_image.save(save_path, quality=100)

        if not save_denoising:
            latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor

            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                image = self.vae.decode(latents, return_dict=False)[0] # [1, 3, H, W]

            image = image.to(torch.float32)
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image * 255).round().astype("uint8")
            pil_image = Image.fromarray(image[0])

            if img_output_path is None:
                return pil_image

            else:
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
    # Replace this with your local path of distilled model transformer without mlpcache
    transformer_path = "/cache/stage2_nocache"
    num_inference_steps = 2
    student_shift=3.0
    prompts_path = "dataset/test.txt"
    img_output_path = f"outputs/test"

    seeds=[42]
    generate_img_size = 1024

    device = torch.device("cuda:0")
    dtype = torch.float16  # torch.float16 / float32 / bfloat16

    with open(prompts_path, "rt", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    pipeline = InferencePipeline(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        transformer_path=transformer_path,
        generate_img_size=generate_img_size,
        device=device,
        dtype=dtype,
        cfg=1.0,                 # disable cfg when using distilled model
        num_inference_steps=num_inference_steps,  
        max_t5_sequence_length=256,
        student_shift=student_shift,
        student_sigmas=None,
    )

    for idx, prompt in enumerate(prompts[:1]):
        print(f"[{idx}/{len(prompts)}] {prompt}")
        for seed in seeds:
            pipeline.infer(
                prompt=prompt,
                negative_prompt="",
                seed=seed,
                img_output_path=img_output_path,
                prefix=f"{str(idx).rjust(4, '0')}_seed{seed}_",
            )

if __name__ == "__main__":
    main()
