import os
import gc
import sys
import argparse
import functools
from pathlib import Path
from pprint import pformat
from PIL import Image
from datetime import datetime
from tqdm import tqdm
from easydict import EasyDict

from torch.utils.tensorboard import SummaryWriter

ROOT_PATH = str(Path(__file__).parent.parent.absolute())
sys.path.append(ROOT_PATH)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    StateDictType,           
    FullStateDictConfig,         
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import save_file, load_file

import transformers
from diffusers.models.transformers.transformer_sd3 import JointTransformerBlock
import diffusers
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3Pipeline,
    SD3Transformer2DModel,
    AutoencoderKL,
)
from diffusers.optimization import get_scheduler

from common.utils import (
    read_yaml,
    init_distributed_mode,
    set_seed,
    CustomLogger,
)
from train.prompts_dataset import SimplePromptsDataset, JourneyDBPromptDataset
from train.processor import DistillProcessor
from train.train_cache_utils import cache_forward, DeltaMLP, fsdp_safe_global_clip_

from peft import LoraConfig
import lpips
from models.D import ImageConvNextDiscriminator
import types


class Trainer:
    def __init__(self, full_args):
        gc.disable()
        # ================= #
        # Init distribution #
        # ================= #
        self.args = full_args.training
        init_distributed_mode(self.args)
        self.rank = self.args.rank
        self.local_rank = self.args.local_rank
        self.world_size = self.args.world_size
        self.is_local_main_process = self.local_rank == 0
        self.is_main_process = self.rank == 0

        time_prefix = datetime.now().strftime("%Y%m%d%H%M%S")
        output_name = (
            f"{time_prefix}_"
            f"step{self.args.sampling_steps_list[0]}_"
            f"bs{self.args.batch_size * self.world_size * self.args.gradient_accumulation_steps}_"
            f"studentlr{self.args.learning_rate_student}_"
            f"lora{self.args.use_lora}_block{len(self.args.lora_block)}_"
            f"output{self.args.lambda_output}_feature{self.args.lambda_feature}_"
            f"lpips{self.args.lambda_lpips}_"
            f"gan{self.args.lambda_gan}"
        )
        # -----------------------------------------------------------------------

        self.args.output_dir = os.path.join(self.args.output_dir, output_name)
        logging_dir = str(Path(self.args.output_dir, self.args.logging_dir))
        if self.is_local_main_process:
            os.makedirs(self.args.output_dir, exist_ok=True)
            os.makedirs(logging_dir, exist_ok=True)
        self.writer = SummaryWriter(logging_dir)
        self.logger = CustomLogger(
            __name__,
            local_rank=self.local_rank,
            rank=self.rank,
            to_file=self.is_local_main_process,
            save_path=os.path.join(logging_dir, f"rank{self.rank}_training.log"),
        )
        self.logger.info(pformat(full_args), local_main_process_only=True)

        self.device = torch.device(f"cuda:{self.local_rank}")
        self.dtype = torch.float32
        if self.args.mixed_precision == "fp16":
            self.dtype = torch.float16
        elif self.args.mixed_precision == "bf16":
            self.dtype = torch.bfloat16
        self.logger.info(f"training dtype: {self.dtype}", local_main_process_only=True)

        if self.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_error()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if self.args.seed is not None:
            seed = self.args.seed + self.rank
            self.logger.info("rank:%d set seed %d", self.rank, seed)
            set_seed(seed)


        # =========================== #
        # Init Dataset and Dataloader #
        # =========================== #
        self.logger.info("***** Loading Dataset *****", local_main_process_only=True)
        self.dataset = JourneyDBPromptDataset(
            jsonl_path=self.args.train_prompts_path, 
            local_rank=self.local_rank, 
            max_samples=self.args.max_samples) 
        if dist.is_initialized():
            self.dataset_sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
        else:
            self.dataset_sampler = None

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.args.batch_size,
            sampler=self.dataset_sampler,
            shuffle=(self.dataset_sampler is None),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=self.args.prefetch_factor,
            persistent_workers=True,
            collate_fn=JourneyDBPromptDataset.collate_fn,
        )

        self.total_batch_size = (self.args.batch_size * self.world_size * self.args.gradient_accumulation_steps)

        # ========= Eval Dataset =========
        # Each line in the txt file is one prompt
        if self.args.enable_eval:
            self.eval_dataset = SimplePromptsDataset(prompts_path=self.args.eval_prompts_path, local_rank=self.local_rank,)

            if dist.is_initialized():
                eval_sampler = DistributedSampler(
                    self.eval_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                )
            else:
                eval_sampler = None

            self.eval_dataloader = DataLoader(
                dataset=self.eval_dataset,
                batch_size=1,
                shuffle=False,
                sampler=eval_sampler,
                num_workers=1,
                persistent_workers=True,
                collate_fn=SimplePromptsDataset.collate_fn,
            )
        if self.is_main_process:
            self.logger.info("***** Running training *****")
            self.logger.info(f"  Num examples = {len(self.dataset)}")
            self.logger.info(f"  Num batches each epoch = {len(self.dataloader)}")
            self.logger.info(f"  Num Epochs = {self.args.num_epochs}")
            self.logger.info(f"  Instantaneous batch size per device = {self.args.batch_size}")
            self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
            self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
            self.logger.info(f"  Total optimization steps = {self.args.max_train_steps}")

        # ================== #
        # define training vals
        # ================== #
        self.step = 0
        self.sync_gradients_step = 0
        self.accumulate_time = 0.0
        self.training_stop_flag = False
        self.bsz = self.args.batch_size
        self.img_size = self.args.img_size

        self.accu_g_loss = 0.0
        self.accu_output_loss = 0.0
        self.accu_feature_loss = 0.0
        self.accu_lpips_loss = 0.0
        self.accu_disc_loss = 0.

        self.accu_d_loss = 0.0
        self.accu_r_loss = 0.0
        self.accu_p_loss = 0.0
        # =============================== #
        # Init models and noise_scheduler #
        # =============================== #
        self.vae_scale_factor = 8
        self.latents_shape = (
            self.bsz,
            16,
            self.img_size // self.vae_scale_factor,
            self.img_size // self.vae_scale_factor,
        )
        self.logger.info("***** Loading Pipeline *****", local_main_process_only=True)
        self.sd3_pipe = StableDiffusion3Pipeline.from_pretrained(
            self.args.pretrained_model_name_or_path,
            torch_dtype=self.dtype,
        )
        del self.sd3_pipe.transformer
        gc.collect()
        
        self.sd3_pipe.to(self.device)

        # scheduler
        self.noise_scheduler = self.sd3_pipe.scheduler
            
        # -------- vae --------
        self.vae: AutoencoderKL = self.sd3_pipe.vae
        self.vae.requires_grad_(False)
        # self.vae.enable_slicing()
        # self.vae.enable_tiling()

        # =============================== #
        # Init processor #
        # =============================== #
        self.processor = DistillProcessor(
            teacher_training_steps=self.noise_scheduler.config.num_train_timesteps,
            shift=self.noise_scheduler.config.shift,
            device=self.device,
            dtype=self.dtype,
            sampling_steps_list=self.args.sampling_steps_list,
            student_shift=self.args.student_shift,
            student_sigmas=self.args.student_sigmas,
        )

        if self.args.generate_ref_img:
            self.processor_ref = DistillProcessor(
                teacher_training_steps=self.noise_scheduler.config.num_train_timesteps,
                shift=self.noise_scheduler.config.shift,
                device=self.device,
                dtype=self.dtype,
                sampling_steps_list=[self.args.generate_ref_step],
                student_shift=self.args.student_shift,
                student_sigmas=self.args.student_sigmas,
            )

        self.logger.info(f"using sampling-steps:{self.processor.sampling_steps_list}", local_main_process_only=True,)
        # =============================== #
        # Init processor.uncond_model_kwargs #
        # =============================== #
        if self.processor.uncond_model_kwargs is None and self.args.guidance_scale is not None:
            uncond_prompts = [""] * self.bsz
            with torch.no_grad():
                uncond_prompt_embeds, _, uncond_pooled_prompt_embeds, _ = self.sd3_pipe.encode_prompt(
                    prompt=uncond_prompts,
                    prompt_2=None,
                    prompt_3=None,
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,  
                    negative_prompt=None,
                    negative_prompt_2=None,
                    negative_prompt_3=None,
                    max_sequence_length=self.args.t5_max_sequence_length,
                )
            self.processor.uncond_model_kwargs = dict(
                encoder_hidden_states=uncond_prompt_embeds,      # [B, seq, dim]
                pooled_projections=uncond_pooled_prompt_embeds,  # [B, pooled_dim]
                # return_dict=False,
            )

        # fsdp setting
        precision_for_train = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=self.dtype,
        )
        precision_for_infer = MixedPrecision(
            param_dtype=self.dtype,
            reduce_dtype=self.dtype,
            buffer_dtype=self.dtype,
        )
        my_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={JointTransformerBlock},
        )

        # ============ #
        # score models #
        # ============ #
        self.logger.info("***** Loading Transformer *****", local_main_process_only=True)

        # student
        self.transformer = SD3Transformer2DModel.from_pretrained(self.args.init_model_name_or_path, subfolder="transformer")
        self.transformer.requires_grad_(True)
        
        # lora setting
        if self.args.use_lora:
            base_targets = [
                "attn.to_k",
                "attn.to_q",
                "attn.to_v",
                "attn.to_out.0",
                "attn.add_k_proj",
                "attn.add_q_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "ff.net.0.proj",
                "ff.net.2",
                "ff_context.net.0.proj",
                "ff_context.net.2",
            ]

            target_modules = [f"transformer_blocks.{i}.{t}" for i in self.args.lora_block for t in base_targets]
            self.logger.info(f"add lora to:{self.args.lora_block}", local_main_process_only=True)
            transformer_lora_config = LoraConfig(
                r=self.args.lora_rank,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
            self.transformer.add_adapter(transformer_lora_config)   
            # add_adapter will set transformer.requires_grad_(False) and lora.requires_grad_(True)
        
        # cache setting
        self.transformer.forward = types.MethodType(cache_forward, self.transformer)
        self.transformer.__class__.enable_cachestep = False
        self.transformer.__class__.enable_cacheblock = True
        self.transformer.__class__.cnt = 0
        self.transformer.__class__.num_steps = self.args.sampling_steps_list[0]
        self.transformer.__class__.previous_residual = None
        self.transformer.__class__.block_delta = {}
        self.transformer.__class__.block_delta["hidden"] = {}
        self.transformer.__class__.block_delta["encoder_hidden"] = {}
        self.transformer.__class__.cache_block = self.args.cache_block

        # =========================
        # delta predictor f setting
        # =========================
        self.transformer.enable_delta_pred = self.args.enable_delta_pred
        self.logger.info(f"enable delta pred:{self.transformer.enable_delta_pred}", local_main_process_only=True)
        if self.transformer.enable_delta_pred:
            # init f
            hidden_dim = self.transformer.inner_dim
            enc_dim = self.transformer.config.caption_projection_dim
            self.transformer.delta_pred_hidden = DeltaMLP(
                dim=hidden_dim,
                hidden_dim=getattr(self.args, "delta_pred_mlp_hidden_dim", hidden_dim * 2),
                dropout=getattr(self.args, "delta_pred_dropout", 0.0),
            )
            self.transformer.delta_pred_encoder = DeltaMLP(
                dim=enc_dim,
                hidden_dim=getattr(self.args, "delta_pred_mlp_hidden_dim", enc_dim * 2),
                dropout=getattr(self.args, "delta_pred_dropout", 0.0),
            )
            for p in self.transformer.delta_pred_hidden.parameters():
                p.requires_grad = True
            for p in self.transformer.delta_pred_encoder.parameters():
                p.requires_grad = True

        if self.is_main_process:
            transformer_params = sum(p.numel() for p in self.transformer.parameters())
            transformer_params_require_grad = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            self.logger.info(f"Total number of transformer parameters: {transformer_params}")
            self.logger.info(f"Total number of transformer trainable parameters: {transformer_params_require_grad}")

        if self.args.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        self.transformer = FSDP(
            self.transformer,
            auto_wrap_policy=my_auto_wrap_policy,
            mixed_precision=precision_for_train,
            device_id=self.device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            use_orig_params=True,
            limit_all_gathers=True,
        )


        if self.args.generate_ref_img:
            self.transformer_ref = SD3Transformer2DModel.from_pretrained(self.args.teacher_model_name_or_path, subfolder="transformer")
            self.transformer_ref.requires_grad_(False)
            self.transformer_ref.eval()
            self.transformer_ref = FSDP(
                self.transformer_ref,
                auto_wrap_policy=my_auto_wrap_policy,
                mixed_precision=precision_for_infer,
                device_id=self.device,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                use_orig_params=True,
                limit_all_gathers=True,
            )

        # discriminator
        self.discriminator = ImageConvNextDiscriminator(pretrained="laion2b_s34b_b82k_augreg_soup", precision="bf16")
        self.logger.info("init ImageConvNextDiscriminator complete.")
        self.discriminator.to(self.device)
        self.discriminator = torch.nn.parallel.DistributedDataParallel(self.discriminator, device_ids=[self.device])

        self.net_lpips = lpips.LPIPS(net="vgg", verbose=True).to(self.device).requires_grad_(False).eval()
        self.logger.info("init lpips complete.")
        # =============================== #
        # Init optimizer and lr_scheduler #
        # =============================== #

        self.delta_parameters = []
        self.base_transformer_parameters = []

        for name, param in self.transformer.named_parameters():
            if not param.requires_grad:
                continue
            if "delta_pred_hidden" in name or "delta_pred_encoder" in name:
                self.delta_parameters.append(param)
            else:
                self.base_transformer_parameters.append(param)

        self.logger.info(
            f"delta_params={len(self.delta_parameters)}, "
            f"base_transformer_params={len(self.base_transformer_parameters)}",
            local_main_process_only=True,
        )

        # optimizer for delta mlp only
        self.optimizer_delta = torch.optim.AdamW(
            params=self.delta_parameters,
            lr=self.args.learning_rate_mlp,
            betas=(0.9, 0.95),
            weight_decay=1e-4,
            eps=1e-8,
        )
        # optimizer for original transformer trainable params
        self.optimizer = torch.optim.AdamW(
            params=self.base_transformer_parameters,
            lr=self.args.learning_rate_student,
            betas=(0.9, 0.95),
            weight_decay=1e-4,
            eps=1e-8,
        )

        dis_parameters = list(filter(lambda p: p.requires_grad, self.discriminator.parameters()))
        dis_parameters_with_lr = {"params": dis_parameters, "lr": self.args.learning_rate_dis}
        self.dis_optimizer = torch.optim.AdamW(
            [dis_parameters_with_lr],
            betas=(0.9, 0.95),
            weight_decay=1e-4,
            eps=1e-8
        )

        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )
        self.dis_lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.dis_optimizer,
            num_warmup_steps=self.args.lr_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )

        self.logger.info(f"init complete (rank:{self.rank}), waiting for other ranks...")
        dist.barrier()

    def encode_prompts(self, prompts):
        with torch.no_grad():
            prompt_embeds, _neg_prompt_embeds, pooled_prompt_embeds, _neg_pooled_prompt_embeds = self.sd3_pipe.encode_prompt(
                prompt=prompts,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=None,
                prompt_2=None,
                prompt_3=None,
                negative_prompt_2=None,
                negative_prompt_3=None,
                clip_skip=None,
                max_sequence_length=self.args.t5_max_sequence_length,
            )
        return pooled_prompt_embeds, prompt_embeds

    @property
    def sync_gradients(self) -> bool:
        return self.step % self.args.gradient_accumulation_steps == 0


    def train_one_step(self, data: dict, epoch: int, step: int):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            prompts = data["prompts"]
            pooled_prompt_embeds, prompt_embeds = self.encode_prompts(prompts)
            model_kwargs = dict(
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )
            
            self.processor.sample_student_sampling_steps()
            sigmas = self.processor.student_sample_info.sigmas
            noise = torch.randn(size=self.latents_shape, device=self.device, dtype=self.dtype,)
            latents = noise

        ## training
        self.transformer.train()
        # ============================
        # step0 fully infer
        # ============================
        with torch.no_grad():
            with torch.autocast(self.device.type, dtype=self.dtype):
                pred_v, delta_0, _ = self.transformer(
                    hidden_states=latents,
                    timestep=sigmas[0].reshape([-1])*self.noise_scheduler.config.num_train_timesteps,
                    **model_kwargs,
                )
            latents = latents.to(torch.float32) + (sigmas[1]-sigmas[0]).reshape(-1,1,1,1) * pred_v.to(torch.float32)
            latents = latents.to(self.dtype)
        
        # ============================
        # step1 fully infer
        # ============================
        with torch.no_grad():
            with torch.autocast(self.device.type, dtype=self.dtype):
                pred_v_fullycalc, delta_1, _ = self.transformer(
                    hidden_states=latents,
                    timestep=sigmas[1].reshape([-1])*self.noise_scheduler.config.num_train_timesteps,
                    **model_kwargs,
                )

            if not self.args.generate_ref_img:
                # use 2-step fully infer img as ref img
                latents_fullyinfer = latents.to(torch.float32) + (sigmas[2]-sigmas[1]).reshape(-1,1,1,1) * pred_v_fullycalc.to(torch.float32)
                latents_fullyinfer = latents_fullyinfer.to(self.dtype)
                latents_fullyinfer = latents_fullyinfer / self.vae.config.scaling_factor + self.vae.config.shift_factor
                fullyinfer_img = self.vae.decode(latents_fullyinfer, return_dict=False)[0]

        # use a distlled model to generate ref img
        if self.args.generate_ref_img:
            latents_ref = self.processor_ref.student_generate(self.transformer_ref, noise, model_kwargs)[0][-1]
            latents_ref = latents_ref / self.vae.config.scaling_factor + self.vae.config.shift_factor
            fullyinfer_img = self.vae.decode(latents_ref, return_dict=False)[0]

        if self.sync_gradients_step >= self.args.start_update_generator_step and \
          self.sync_gradients_step % (self.args.update_generater_interval + 1) == 0:
            self.phase = "G"
            # train generator
            self.transformer.train()
            self.discriminator.module.decoder.eval()
            self.discriminator.module.decoder.requires_grad_(False)

            # ============================
            # step1 cache infer
            # ============================
            with torch.autocast(self.device.type, dtype=self.dtype):
                pred_v_cache, _, delta_1_pred = self.transformer(
                    hidden_states=latents,
                    timestep=sigmas[1].reshape([-1])*self.noise_scheduler.config.num_train_timesteps,
                    delta_cache = delta_0,
                    **model_kwargs,
                ) 
            latents_cache = latents.to(torch.float32) + (sigmas[2]-sigmas[1]).reshape(-1,1,1,1) * pred_v_cache.to(torch.float32)
            latents_cache = latents_cache.to(self.dtype)
            latents_cache = latents_cache / self.vae.config.scaling_factor + self.vae.config.shift_factor
            cache_img = self.vae.decode(latents_cache, return_dict=False)[0]

            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                loss_lpips = self.net_lpips(cache_img, fullyinfer_img).mean() * self.args.lambda_lpips

            loss_disc = self.discriminator(cache_img, for_G=True).mean() * self.args.lambda_gan

            loss_output = F.mse_loss(pred_v_fullycalc, pred_v_cache, reduction="none").mean()*self.args.lambda_output
            loss_feature = (F.mse_loss(delta_0["hidden"], delta_1["hidden"], reduction="none").mean() + F.mse_loss(delta_0["encoder_hidden"], delta_1["encoder_hidden"], reduction="none").mean())*self.args.lambda_feature

            loss = loss_output.float() + loss_feature.float()
            loss += loss_lpips.float() + loss_disc.float()

            loss = loss / self.args.gradient_accumulation_steps
            loss.backward()

            if self.sync_gradients:
                self.transformer.clip_grad_norm_(2.0, 2)

                # delta optimizer always updates
                self.optimizer_delta.step()
                self.optimizer_delta.zero_grad()

                # base transformer optimizer starts after MLP warmup
                if self.sync_gradients_step > self.args.f_warmup_iter:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                else:
                    self.optimizer.zero_grad()

            dist.all_reduce(loss)
            avg_loss= loss.detach() / self.world_size
            self.accu_g_loss += avg_loss.item()

            dist.all_reduce(loss_output)
            dist.all_reduce(loss_feature)
            avg_output_loss= loss_output.detach() / self.world_size / self.args.gradient_accumulation_steps
            self.accu_output_loss += avg_output_loss.item()
            avg_feature_loss = loss_feature.detach() / self.world_size / self.args.gradient_accumulation_steps
            self.accu_feature_loss += avg_feature_loss.item()

            dist.all_reduce(loss_lpips)
            dist.all_reduce(loss_disc)
            avg_lpips_loss= loss_lpips.detach() / self.world_size / self.args.gradient_accumulation_steps
            self.accu_lpips_loss += avg_lpips_loss.item()
            avg_disc_loss= loss_disc.detach() / self.world_size / self.args.gradient_accumulation_steps
            self.accu_disc_loss += avg_disc_loss.item()

        else:
            self.phase = "D"
            # train discriminator
            self.transformer.eval()
            self.discriminator.module.decoder.train()
            self.discriminator.module.decoder.requires_grad_(True)

            r_loss, real_feats = self.discriminator(fullyinfer_img, for_real=True, return_logits=True)

            # self.logger.info(f"real_logits:{real_logits.detach().mean(dim=1).flatten().tolist()}")
            r_loss = r_loss.mean()
            r_loss = r_loss / self.args.gradient_accumulation_steps
            r_loss.backward()

            with torch.no_grad():
                with torch.autocast(self.device.type, dtype=self.dtype):
                    pred_v_cache = self.transformer(
                        hidden_states=latents,
                        timestep=sigmas[1].reshape([-1])*self.noise_scheduler.config.num_train_timesteps,
                        delta_cache = delta_0,
                        **model_kwargs,
                    )[0]
                latents_cache = latents.to(torch.float32) + (sigmas[2]-sigmas[1]).reshape(-1,1,1,1) * pred_v_cache.to(torch.float32)
                latents_cache = latents_cache.to(self.dtype)
                latents_cache = latents_cache / self.vae.config.scaling_factor + self.vae.config.shift_factor
                cache_img = self.vae.decode(latents_cache, return_dict=False)[0]

            p_loss, pred_feats = self.discriminator(cache_img, for_real=False, return_logits=True)

            # self.logger.info(f"pred_logits:{pred_logits.detach().mean(dim=1).flatten().tolist()}")
            p_loss = p_loss.mean()
            p_loss = p_loss / self.args.gradient_accumulation_steps
            p_loss.backward()

            if self.sync_gradients:
                self.dis_optimizer.step()
                self.dis_lr_scheduler.step()
                self.dis_optimizer.zero_grad()

            d_loss = r_loss + p_loss
            dist.all_reduce(d_loss)
            avg_d_loss = d_loss.detach() / self.world_size
            self.accu_d_loss += avg_d_loss.item()

            dist.all_reduce(r_loss)
            dist.all_reduce(p_loss)
            avg_r_loss= r_loss.detach() / self.world_size
            self.accu_r_loss += avg_r_loss.item()
            avg_p_loss= p_loss.detach() / self.world_size
            self.accu_p_loss += avg_p_loss.item()
        end_event.record()
        torch.cuda.synchronize()
        self.accumulate_time += start_event.elapsed_time(end_event)

        if self.sync_gradients:

            self.sync_gradients_step += 1

            if self.is_main_process:
                student_lr_step = self.lr_scheduler.get_last_lr()[0]
                _message = f'[epoch:{epoch}]iter {self.sync_gradients_step} / {self.args.max_train_steps} '
                _message += f'| student steps: {self.processor.sampling_steps} '
                _message += f'| student lr: {student_lr_step:.2e} '
                _message += f'| total bs: {self.total_batch_size} '
                _message += f'| time per iter(ms): {self.accumulate_time:.3f} '
                _message += f'| g_loss/output/feature/lpips/disc: {self.accu_g_loss:.5f}/{self.accu_output_loss:.5f}/{self.accu_feature_loss:.5f}/{self.accu_lpips_loss:.5f}/{self.accu_disc_loss:.5f}'
                _message += f'| d_loss/p/r: {self.accu_d_loss:.5f}/{self.accu_p_loss:.5f}/{self.accu_r_loss:.5f} '

                self.logger.info(_message)

                if self.phase == "G":
                    self.writer.add_scalars(
                        "G/losses",
                        {
                            "loss":     self.accu_g_loss,
                            "feature":    self.accu_feature_loss,
                            "output": self.accu_output_loss,
                            "lpips":    self.accu_lpips_loss,
                            "disc": self.accu_disc_loss,
                        },
                        self.sync_gradients_step
                    )
                else:
                    self.writer.add_scalars(
                        "D/losses",
                        {
                            "d": self.accu_d_loss,
                            "p": self.accu_p_loss,
                            "r": self.accu_r_loss,
                        },
                        self.sync_gradients_step
                    )

            self.accumulate_time = 0.
            self.accu_g_loss = 0.

            self.accu_feature_loss = 0.
            self.accu_output_loss = 0.
            self.accu_lpips_loss = 0.
            self.accu_disc_loss = 0.

            self.accu_d_loss = 0.
            self.accu_r_loss = 0.
            self.accu_p_loss = 0.

            # Save weights
            if self.sync_gradients_step % self.args.save_checkpoint_interval == 0:
                save_path = os.path.join(self.args.output_dir, f"checkpoint-{self.sync_gradients_step}")
                os.makedirs(save_path, exist_ok=True)

                if self.args.enable_eval:
                    self.eval(save_path)

                if self.args.use_fsdp:
                    if self.args.save_weights:
                        save_weights_path = os.path.join(self.args.output_dir, f"checkpoint-{self.sync_gradients_step}")
                        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

                        with FSDP.state_dict_type(self.transformer, StateDictType.FULL_STATE_DICT, full_cfg):
                            full_sd = self.transformer.state_dict()

                        if self.is_main_process:
                            transformer_dir = os.path.join(save_weights_path, "transformer")
                            os.makedirs(transformer_dir, exist_ok=True)
                            # Ensure they are all on CPU
                            full_sd_cpu = {k: v.cpu() for k, v in full_sd.items()}
                            full_path = os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors")
                            save_file(full_sd_cpu, full_path)
                            os.system("cp " + os.path.join(f"{self.args.pretrained_model_name_or_path}", "transformer/config.json ") + transformer_dir)
                            self.logger.info(f"[rank{self.rank}] Saved FULL_STATE_DICT (safetensors) to {full_path}")

                else:
                    if self.is_main_process:
                        model = self.transformer.module
                        model.save_pretrained(save_path)
                        self.logger.info(f"Saved state to {save_path}")
                gc.collect()

    @torch.inference_mode()
    def eval(self, save_dir: str):
        save_img_dir = os.path.join(save_dir, "img")

        self.processor.set_sampling_steps(self.args.eval_step)
        dataloader = self.eval_dataloader
        generator = torch.Generator(self.device)
        latents_shape = (
            1,
            self.latents_shape[1],
            self.latents_shape[2],
            self.latents_shape[3],
        )
        if self.local_rank == 0:
            dataloader = tqdm(self.eval_dataloader)
        for i, data in enumerate(dataloader):
            prompts = data["prompts"]

            generator.manual_seed(self.args.eval_seed)
            noise = torch.randn(
                size=latents_shape,
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            )

            pooled_prompt_embeds, prompt_embeds = self.encode_prompts(prompts)
            model_kwargs = dict(
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )

            # latents_list, _ = self.processor.student_generate(
            #     model=self.transformer,
            #     noises=noise,
            #     model_kwargs=model_kwargs,
            #     eta=1.0,
            # )
            sigmas = self.processor.student_sample_info.sigmas
            with torch.no_grad():
                latents = noise
                noisy_latent_list =[]
                for step in range(2):
                    noisy_latent_list.append(latents)
                    if step == 0:
                        with torch.autocast(self.device.type, dtype=self.dtype):
                            pred_v, delta_0, _ = self.transformer(
                                hidden_states=latents,
                                timestep=sigmas[0].reshape([-1])*self.noise_scheduler.config.num_train_timesteps,
                                **model_kwargs,
                            )
                        latents = latents.to(torch.float32) + (sigmas[1]-sigmas[0]).reshape(-1,1,1,1) * pred_v.to(torch.float32)
                        latents = latents.to(self.dtype)
                        noisy_latent_list.append(latents)
                    else:
                        with torch.autocast(self.device.type, dtype=self.dtype):
                            pred_v = self.transformer(
                                hidden_states=latents,
                                timestep=sigmas[1].reshape([-1])*self.noise_scheduler.config.num_train_timesteps,
                                delta_cache=delta_0,
                                **model_kwargs,
                            )[0]
                        latents = latents.to(torch.float32) + (sigmas[2]-sigmas[1]).reshape(-1,1,1,1) * pred_v.to(torch.float32)
                        latents = latents.to(self.dtype)
                        noisy_latent_list.append(latents)
            # latents = latents_list[-1]

            latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor

            indices = data["indices"]

            os.makedirs(save_img_dir, exist_ok=True)
            image = self.vae.decode(latents, return_dict=False)[0]
            image = image.to(torch.float32)
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image * 255).round().astype("uint8")
            pil_image = Image.fromarray(image[0])
            image_save_path = os.path.join(
                save_img_dir, f"{str(indices[0].item()).rjust(4,'0')}_seed{self.args.eval_seed}.png"
            )
            pil_image.save(image_save_path)

    def train(self):
        for epoch in range(self.args.num_epochs):
            if hasattr(self.dataset_sampler, "set_epoch"):
                self.dataset_sampler.set_epoch(epoch)

            gc.collect()
            for step, data in enumerate(self.dataloader):
                self.step += 1
                self.train_one_step(data, epoch, step)
                if (self.sync_gradients_step >= self.args.max_train_steps):
                    self.training_stop_flag = True
                    break

            if self.training_stop_flag:
                self.logger.info(f"reaching the max_train_steps:{self.args.max_train_steps}")
                break
        else:
            self.logger.info(
                f"reaching the max_epochs:{self.args.num_epochs}"
            )

        dist.barrier()
        dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description="training script")
    parser.add_argument(
        "--train_cfg_file",
        type=str,
        help="Path to train_cfg_file",
    )
    args = parser.parse_args()

    train_cfg_file = args.train_cfg_file
    if not os.path.exists(train_cfg_file):
        raise FileNotFoundError(train_cfg_file)

    args = EasyDict(read_yaml(train_cfg_file))
    return args



def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
