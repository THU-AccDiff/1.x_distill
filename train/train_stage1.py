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
from safetensors.torch import save_file


import transformers
import diffusers
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3Pipeline,
    SD3Transformer2DModel,
    AutoencoderKL,
)
from diffusers.models.transformers.transformer_sd3 import JointTransformerBlock
from diffusers.optimization import get_scheduler

from common.utils import (
    read_yaml,
    init_distributed_mode,
    set_seed,
    CustomLogger,
)
from train.prompts_dataset import SimplePromptsDataset, JourneyDBPromptDataset
from train.processor import DistillProcessor

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
        # --------- output name  --------------------
        student_sigmas = getattr(self.args, "student_sigmas", None)
        if student_sigmas is None or (isinstance(student_sigmas, (list, tuple)) and len(student_sigmas) == 0):
            shift_or_sigmas_str = f"shift{self.args.student_shift}"
        else:
            if isinstance(student_sigmas, (list, tuple)):
                sigma_str = "_".join(f"{s}" for s in student_sigmas)
            else:
                sigma_str = str(student_sigmas)
            shift_or_sigmas_str = f"sigmas{sigma_str}"


        output_name = (
            f"{time_prefix}_"
            f"step{self.args.sampling_steps_list[0]}_"
            f"bs{self.args.batch_size * self.world_size * self.args.gradient_accumulation_steps}_"
            f"guidance{self.args.guidance_scale}_{self.args.cfg_control}_"
            f"fakelr{self.args.learning_rate_fake}_"
            f"studentlr{self.args.learning_rate_student}_"
            f"{shift_or_sigmas_str}_"
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

        # =============== #
        # Create profiler #
        # =============== #
        # self.profiler = create_profiler(
        #     full_args.prof, save_path=f"prof_results/rank_{self.rank}"
        # )

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

        self.accu_score_loss = 0.0
        self.accu_dm_loss = 0.0

        # =============================== #
        # Init models and noise_scheduler #
        # =============================== #
        # SD3's VAE: latent channel 16, scaling factor 8
        self.vae_scale_factor = 8
        self.latents_shape = (
            self.bsz,
            16,
            self.img_size // self.vae_scale_factor,
            self.img_size // self.vae_scale_factor,
        )
        self.logger.info("***** Loading Pipeline *****", local_main_process_only=True)
        # Only use its text encoders / tokenizers / vae / scheduler, not internal transformer
        self.sd3_pipe = StableDiffusion3Pipeline.from_pretrained(
            self.args.pretrained_model_name_or_path,
            torch_dtype=self.dtype,
        )
        del self.sd3_pipe.transformer
        gc.collect()
        self.sd3_pipe.to(self.device)
        self.noise_scheduler = self.sd3_pipe.scheduler

        # -------- text/vae helper --------
        self.vae: AutoencoderKL = self.sd3_pipe.vae

        # =============================== #
        # Init Processor #
        # =============================== #
        self.processor = DistillProcessor(
            teacher_training_steps=self.noise_scheduler.config.num_train_timesteps,
            shift=self.noise_scheduler.config.shift,
            device=self.device,
            dtype=self.dtype,
            sampling_steps_list=self.args.sampling_steps_list,
            student_shift=self.args.student_shift,
            student_sigmas=self.args.student_sigmas,
            train_window=self.args.train_window,
            split_interval=self.args.split_interval,
            use_sigma_function=self.args.use_sigma_function,
            points_mode=self.args.points_mode,
            cfg_control=self.args.cfg_control
        )

        self.logger.info(f"using sampling-steps:{self.processor.sampling_steps_list}", local_main_process_only=True,)
        # =============================== #
        # Init uncond_model_kwargs #
        # =============================== #
        if self.processor.uncond_model_kwargs is None and self.args.guidance_scale is not None:
            uncond_prompts = [""] * self.bsz
            with torch.no_grad():
                # SD3 encode_prompt: returns (prompt_embeds, neg_embeds, pooled, pooled_neg)
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
        # FULL_SHARD on entire model will cause OOM
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
        init_model = self.args.init_model_name_or_path
        fake_subfolder = "fake_transformer" if os.path.isdir(os.path.join(init_model, "fake_transformer")) else "transformer"
        fake_model = SD3Transformer2DModel.from_pretrained(init_model, subfolder=fake_subfolder)
        fake_model.requires_grad_(True)
        if self.args.gradient_checkpointing:
            fake_model.enable_gradient_checkpointing()
        self.fake_model = FSDP(
            fake_model,
            auto_wrap_policy=my_auto_wrap_policy,
            mixed_precision=precision_for_train,
            device_id=self.device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            use_orig_params=True,
            limit_all_gathers=True,
        )

        real_model = SD3Transformer2DModel.from_pretrained(self.args.teacher_model_name_or_path, subfolder="transformer")
        real_model.requires_grad_(False)
        self.real_model = FSDP(
            real_model,
            auto_wrap_policy=my_auto_wrap_policy,
            mixed_precision=precision_for_infer,
            device_id=self.device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            use_orig_params=True,
            limit_all_gathers=True,
        )

        # student
        transformer = SD3Transformer2DModel.from_pretrained(init_model, subfolder="transformer")
        transformer.requires_grad_(True)

        if self.is_main_process:
            transformer_params = sum(p.numel() for p in transformer.parameters())
            transformer_params_require_grad = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
            self.logger.info(f"Total number of transformer parameters: {transformer_params}")
            self.logger.info(f"Total number of transformer trainable parameters: {transformer_params_require_grad}")

        if self.args.gradient_checkpointing:
            transformer.enable_gradient_checkpointing()

        self.transformer = FSDP(
            transformer,
            auto_wrap_policy=my_auto_wrap_policy,
            mixed_precision=precision_for_train,
            device_id=self.device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            use_orig_params=True,
            limit_all_gathers=True,
        )


        # =============================== #
        # Init optimizer and lr_scheduler #
        # =============================== #
        transformer_parameters_with_lr = {"params": self.transformer.parameters(), "lr": self.args.learning_rate_student,}
        self.optimizer = torch.optim.AdamW(
            params=[transformer_parameters_with_lr],
            betas=(0., 0.999),
            weight_decay=1e-4,
            eps=1e-8,
        )

        transformer_parameters_with_lr = {"params": self.fake_model.parameters(), "lr": self.args.learning_rate_fake,}
        self.optimizer_fake = torch.optim.AdamW(
            params=[transformer_parameters_with_lr],
            betas=(0., 0.999),
            weight_decay=1e-4,
            eps=1e-8,
        )

        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )

        self.lr_scheduler_fake = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer_fake,
            num_warmup_steps=self.args.lr_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )
        self.logger.info(
            f"init complete (rank:{self.rank}), waiting for other ranks..."
        )
        dist.barrier()


    def encode_prompts(self, prompts):
        """
        Use StableDiffusion3Pipeline's internal logic to encode text,
        returning pooled_prompt_embeds, prompt_embeds
        """
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

        self.transformer.train()
        prompts = data["prompts"]

        pooled_prompt_embeds, prompt_embeds = self.encode_prompts(prompts)

        model_kwargs = dict(
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )

        self.processor.sample_student_sampling_steps()

        # ====================== #
        # Train Fake Score Model #
        # ====================== #
        self.transformer.eval()
        self.fake_model.train()
        # Sample initial noise
        noise = torch.randn(
            size=self.latents_shape,
            device=self.device,
            dtype=self.dtype,
        )

        noisy_latent_list, velocity_list = self.processor.student_generate(
            model=self.transformer,
            noises=noise,
            model_kwargs=model_kwargs,
            eta=self.args.eta,
        )

        noisy_model_latents, sample_sigmas, target = (
            self.processor.sample_for_train_fake_model(
                bsz=self.bsz,
                noisy_latent_list=noisy_latent_list,
                velocity_list=velocity_list,
                sample_in_window=self.args.sample_in_window_f,
            )
        )

        with torch.autocast(self.device.type, self.dtype):
            fake_v = self.fake_model(
                hidden_states=noisy_model_latents,
                timestep=sample_sigmas.reshape([-1])*self.noise_scheduler.config.num_train_timesteps,
                **model_kwargs,
            )[0]

        dm_loss = ((fake_v.float() - target.float()) ** 2).reshape(self.bsz, -1)
        dm_loss = torch.mean(dm_loss, dim=1)
        dm_loss = torch.mean(dm_loss)
        dm_loss = dm_loss / self.args.gradient_accumulation_steps

        if self.args.lambda_reg > 0:
            with torch.no_grad():
                with torch.autocast(self.device.type, self.dtype):
                    real_v = self.real_model(
                        hidden_states=noisy_model_latents,
                        timestep=sample_sigmas.reshape([-1])*self.noise_scheduler.config.num_train_timesteps,
                        **model_kwargs,
                    )[0]
            reg_loss = ((fake_v - real_v) ** 2).reshape(self.bsz, -1)
            reg_loss = torch.mean(reg_loss, dim=1)
            reg_loss = torch.mean(reg_loss)
            reg_loss = reg_loss / self.args.gradient_accumulation_steps

            dm_loss += self.args.lambda_reg * reg_loss
        
        dm_loss.backward()
        dist.all_reduce(dm_loss)
        avg_dm_loss = dm_loss.detach() / self.world_size
        self.accu_dm_loss += avg_dm_loss.item()

        if self.sync_gradients:
            self.fake_model.clip_grad_norm_(1.0, 2)
            self.optimizer_fake.step()
            self.lr_scheduler_fake.step()
            self.optimizer_fake.zero_grad()

        if self.sync_gradients_step % self.args.ttur == 0:
            # =================== #
            # Train Student Model #
            # =================== #
            self.transformer.train()
            self.fake_model.eval()

            (
                pred_latents,
                noisy_model_latents,
                sigmas,
                next_sigmas,
            ) = self.processor.sample_for_train_student_model(
                model=self.transformer,
                bsz=self.bsz,
                noisy_latent_list=noisy_latent_list,
                model_kwargs=model_kwargs,
                sample_in_window=self.args.sample_in_window_s,
                eta=self.args.eta,
            )

            with torch.no_grad():
                _, real_latents = self.processor.model_sample(
                    self.real_model,
                    noisy_model_latents,
                    sigmas,
                    next_sigmas,
                    model_kwargs,
                    self.args.eta,
                    cfg=self.args.guidance_scale,
                )
                _, fake_latents = self.processor.model_sample(
                    self.fake_model,
                    noisy_model_latents,
                    sigmas,
                    next_sigmas,
                    model_kwargs,
                    self.args.eta,
                )
                revised_latents = pred_latents.detach() + real_latents - fake_latents
                huber_c = 1e-3 / ((64 * 64 * 4) ** 0.5) * (
                    (noisy_model_latents.shape[1:].numel()) ** 0.5
                )

            score_loss = torch.sqrt(
                (pred_latents.float() - revised_latents.float()) ** 2 + huber_c**2
            ) - huber_c

            weighting_factor = torch.abs(
                pred_latents.double() - real_latents.double()
            ).mean(dim=[1, 2], keepdim=True).detach()
            score_loss = score_loss / weighting_factor
            score_loss = score_loss.reshape(self.bsz, -1)
            score_loss = torch.mean(score_loss, dim=1)
            score_loss = torch.mean(score_loss)
            score_loss = score_loss / self.args.gradient_accumulation_steps

            score_loss.backward()
            dist.all_reduce(score_loss)
            avg_score_loss = score_loss.detach() / self.world_size
            self.accu_score_loss += avg_score_loss.item()

            if self.sync_gradients:
                self.transformer.clip_grad_norm_(2.0, 2)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

        end_event.record()
        end_event.synchronize()
        self.accumulate_time += start_event.elapsed_time(end_event)

        if self.sync_gradients:
            self.sync_gradients_step += 1

            if self.is_main_process:
                student_lr_step = self.lr_scheduler.get_last_lr()[0]
                fake_lr_step = self.lr_scheduler_fake.get_last_lr()[0]
                _message = f'[epoch:{epoch}]iter {self.sync_gradients_step} / {self.args.max_train_steps} '
                _message += f'| student steps: {self.processor.sampling_steps} '
                _message += f'| student lr: {student_lr_step:.2e} | fake lr: {fake_lr_step:.2e} '
                _message += f'| total bs: {self.total_batch_size} '
                _message += f'| time per iter(ms): {self.accumulate_time:.3f} '
                _message += f'| fake_loss: {self.accu_dm_loss:.5f} '
                _message += f'| student_loss: {self.accu_score_loss:.5f} '

                self.logger.info(_message)
                if (self.sync_gradients_step-1) % self.args.ttur == 0:
                    self.writer.add_scalar('student loss', self.accu_score_loss, self.sync_gradients_step)
                self.writer.add_scalar('fake loss', self.accu_dm_loss, self.sync_gradients_step)

            self.accumulate_time = 0.
            self.accu_score_loss = 0.
            self.accu_dm_loss = 0.

            # Save weights
            if self.sync_gradients_step % self.args.save_checkpoint_interval == 0:
                save_path = os.path.join(self.args.output_dir, f"checkpoint-{self.sync_gradients_step}")
                os.makedirs(save_path, exist_ok=True)

                if self.args.enable_eval:
                    # generate evaluation samples
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
                            os.system("cp " + os.path.join(f"{self.args.teacher_model_name_or_path}", "transformer/config.json ") + transformer_dir)
                            self.logger.info(f"[rank{self.rank}] Saved FULL_STATE_DICT (safetensors) to {full_path}")


                        if self.args.save_fake:
                            with FSDP.state_dict_type(self.fake_model, StateDictType.FULL_STATE_DICT, full_cfg):
                                fake_full_sd = self.fake_model.state_dict()

                            if self.is_main_process:
                                fake_transformer_dir = os.path.join(save_weights_path, "fake_transformer")
                                os.makedirs(fake_transformer_dir, exist_ok=True)
                                fake_full_sd_cpu = {k: v.cpu() for k, v in fake_full_sd.items()}
                                fake_full_path = os.path.join(fake_transformer_dir, "diffusion_pytorch_model.safetensors")
                                save_file(fake_full_sd_cpu, fake_full_path)
                                os.system('cp ' + os.path.join(f"{self.args.teacher_model_name_or_path}", "transformer/config.json ") + fake_transformer_dir)
                                self.logger.info(f"[rank{self.rank}] Saved fake FULL_STATE_DICT (safetensors) to {fake_full_path}")

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

            latents_list, _ = self.processor.student_generate(
                model=self.transformer,
                noises=noise,
                model_kwargs=model_kwargs,
                eta=1.0,
            )

            latents = latents_list[-1]

            # latents = latents / self.vae.config.scaling_factor
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
