import os
import sys
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, List, Optional, Union

from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_sd3 import JointTransformerBlock
ROOT_PATH = str(Path(__file__).parent.parent.parent.absolute())
sys.path.append(ROOT_PATH)
logger = logging.get_logger(__name__)  

# =========================
# Cache Forward: different versions
# =========================
# cache_forward_v1: simple
def cache_forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        delta_cache: torch.LongTensor = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        delta_cache_out = None
        delta_pred = None

        if self.enable_cachestep:
            cache_step = []
            if self.cnt in cache_step and (self.previous_residual is not None):
                should_calc = False
            else:
                should_calc = True
            self.cnt += 1 
            if self.cnt == self.num_steps:
                self.cnt = 0
        if self.enable_cachestep:
            if not should_calc:
                hidden_states += self.previous_residual
                # print(f"cache step:{self.cnt-1}")
            else:
                # print(f"calc step:{self.cnt-1}")
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

                    # controlnet residual
                    if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                        interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                        hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

                self.previous_residual = hidden_states - ori_hidden_states
        else:
            cache_block = self.cache_block
            for index_block, block in enumerate(self.transformer_blocks):
                if self.enable_cacheblock and delta_cache is not None and index_block in cache_block:
                    # if index_block == cache_block[0]:
                    if index_block == cache_block[-1]:
                        dh = delta_cache["hidden"]
                        deh = delta_cache.get("encoder_hidden", None)

                        if getattr(self, "enable_delta_pred", True):
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
                else:
                    if index_block == cache_block[0]:
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

                    # controlnet residual
                    if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                        interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                        hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]
                    
                    if index_block == cache_block[-1]:
                        block_delta_hid = hidden_states - inp_hidden
                        block_delta_enhid = None if (encoder_hidden_states is None or inp_encoder_hidden is None) else (encoder_hidden_states - inp_encoder_hidden)

                        delta_cache_out = {"hidden": block_delta_hid}
                        if block_delta_enhid is not None:
                            delta_cache_out["encoder_hidden"] = block_delta_enhid

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
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
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            # return (output,)
            return (output, delta_cache_out, delta_pred)
        
        # return Transformer2DModelOutput(sample=output), delta_cache
        return Transformer2DModelOutput(sample=output)

# cache_forward_v2: two segments
def cache_forward_stage2(
    self,   
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    pooled_projections: torch.FloatTensor = None,
    timestep: torch.LongTensor = None,
    block_controlnet_hidden_states: List = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
    delta_cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    """
    SD3Transformer2DModel forward with block-cache + MLP approximators.

    Two segments:
      - s1: mlp_block (stage1, frozen MLP delta_pred_hidden/encoder)
      - s2: train_block (stage2, trainable MLP delta_pred_hidden_s2/encoder_s2)

    Behavior:
      - If delta_cache is None: run full blocks, and return delta_cache_out containing true deltas for each segment.
      - If delta_cache is not None: skip blocks inside segments and inject predicted deltas at segment end blocks.
        Return delta_cache_out (predicted deltas) and delta_pred (for loss).
    """
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` via `joint_attention_kwargs` when not using PEFT backend is ineffective.")

    height, width = hidden_states.shape[-2:]
    hidden_states = self.pos_embed(hidden_states)
    temb = self.time_text_embed(timestep, pooled_projections)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    # -------- segments config --------
    mlp_block = getattr(self, "mlp_block", None)
    train_block = getattr(self, "train_block", None)

    if mlp_block is None:
        mlp_block = []
    if train_block is None:
        train_block = []

    segments: List[tuple] = []
    if isinstance(mlp_block, list) and len(mlp_block) > 0:
        segments.append(("s1", mlp_block))
    if isinstance(train_block, list) and len(train_block) > 0:
        segments.append(("s2", train_block))

    seg_of: Dict[int, str] = {}
    seg_start: Dict[str, int] = {}
    seg_end: Dict[str, int] = {}
    for name, seg in segments:
        seg_start[name] = seg[0]
        seg_end[name] = seg[-1]
        for b in seg:
            seg_of[b] = name

    delta_cache_out: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    delta_pred: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

    # -------- cache-step mode (unused in your setting) --------
    if getattr(self, "enable_cachestep", False):
        cache_step = []
        if self.cnt in cache_step and (self.previous_residual is not None):
            should_calc = False
        else:
            should_calc = True
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0

        if not should_calc:
            hidden_states += self.previous_residual
        else:
            ori_hidden_states = hidden_states.clone()
            for index_block, block in enumerate(self.transformer_blocks):
                if self.training and self.gradient_checkpointing:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
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

    # -------- cache-block mode (main path) --------
    else:
        enable_cacheblock = getattr(self, "enable_cacheblock", False)

        # For full-run: record segment inputs to compute true deltas
        inp_h: Dict[str, torch.Tensor] = {}
        inp_e: Dict[str, Optional[torch.Tensor]] = {}

        for index_block, block in enumerate(self.transformer_blocks):
            segname = seg_of.get(index_block, None)

            # -------- cache-run: skip segment blocks; inject at segment end --------
            if enable_cacheblock and (delta_cache is not None) and (segname is not None):
                if index_block == seg_end[segname]:
                    assert segname in delta_cache, f"delta_cache missing key {segname}, got {list(delta_cache.keys())}"
                    dh = delta_cache[segname]["hidden"]
                    deh = delta_cache[segname].get("encoder_hidden", None)

                    if getattr(self, "enable_delta_pred", True):
                        if segname == "s1":
                            assert hasattr(self, "delta_pred_hidden") and hasattr(self, "delta_pred_encoder")
                            dh2 = self.delta_pred_hidden(dh)
                            deh2 = self.delta_pred_encoder(deh) if deh is not None else None
                        else:  # s2
                            assert hasattr(self, "delta_pred_hidden_s2") and hasattr(self, "delta_pred_encoder_s2")
                            dh2 = self.delta_pred_hidden_s2(dh)
                            deh2 = self.delta_pred_encoder_s2(deh) if deh is not None else None

                        hidden_states = hidden_states + dh2
                        if deh2 is not None:
                            encoder_hidden_states = encoder_hidden_states + deh2

                        if delta_cache_out is None:
                            delta_cache_out = {}
                        if delta_pred is None:
                            delta_pred = {}

                        delta_cache_out[segname] = {"hidden": dh2}
                        if deh2 is not None:
                            delta_cache_out[segname]["encoder_hidden"] = deh2

                        delta_pred[segname] = delta_cache_out[segname]
                        if segname == "s2":
                            delta_cache_out["hidden"] = dh2
                            if deh2 is not None:
                                delta_cache_out["encoder_hidden"] = deh2

                            delta_pred["hidden"] = dh2
                            if deh2 is not None:
                                delta_pred["encoder_hidden"] = deh2
                    else:
                        hidden_states = hidden_states + dh
                        if deh is not None:
                            encoder_hidden_states = encoder_hidden_states + deh
                        if delta_cache_out is None:
                            delta_cache_out = {}
                        delta_cache_out[segname] = {"hidden": dh}
                        if deh is not None:
                            delta_cache_out[segname]["encoder_hidden"] = deh
                # Skip computing blocks inside segment
                continue

            # -------- full-run: mark segment start inputs --------
            if (delta_cache is None) and (segname is not None) and (index_block == seg_start[segname]):
                inp_h[segname] = hidden_states.clone()
                inp_e[segname] = encoder_hidden_states.clone() if encoder_hidden_states is not None else None

            # -------- normal block compute --------
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
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

            # -------- full-run: compute true deltas at segment end --------
            if (delta_cache is None) and (segname is not None) and (index_block == seg_end[segname]):
                dh_true = hidden_states - inp_h[segname]
                deh_true = None
                if encoder_hidden_states is not None and inp_e.get(segname, None) is not None:
                    deh_true = encoder_hidden_states - inp_e[segname]

                if delta_cache_out is None:
                    delta_cache_out = {}
                delta_cache_out[segname] = {"hidden": dh_true}
                if deh_true is not None:
                    delta_cache_out[segname]["encoder_hidden"] = deh_true
                if segname == "s2":
                    delta_cache_out["hidden"] = dh_true
                    if deh_true is not None:
                        delta_cache_out["encoder_hidden"] = deh_true

    # -------- output head --------
    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    patch_size = self.config.patch_size
    height2 = height // patch_size
    width2 = width // patch_size
    hidden_states = hidden_states.reshape(
        shape=(hidden_states.shape[0], height2, width2, patch_size, patch_size, self.out_channels)
    )
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(
        shape=(hidden_states.shape[0], self.out_channels, height2 * patch_size, width2 * patch_size)
    )

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output, delta_cache_out, delta_pred)

    return Transformer2DModelOutput(sample=output)

# cache_forward_v3: transformer
def cache_forward_transformer(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        delta_cache: torch.LongTensor = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        delta_cache_out = None
        delta_pred = None

        if self.enable_cachestep:
            cache_step = []
            if self.cnt in cache_step and (self.previous_residual is not None):
                should_calc = False
            else:
                should_calc = True
            self.cnt += 1 
            if self.cnt == self.num_steps:
                self.cnt = 0
        if self.enable_cachestep:
            if not should_calc:
                hidden_states += self.previous_residual
                # print(f"cache step:{self.cnt-1}")
            else:
                # print(f"calc step:{self.cnt-1}")
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

                    # controlnet residual
                    if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                        interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                        hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

                self.previous_residual = hidden_states - ori_hidden_states
        else:
            cache_block = self.cache_block
            for index_block, block in enumerate(self.transformer_blocks):
                if self.enable_cacheblock and delta_cache is not None and index_block in cache_block:
                    # if index_block == cache_block[0]:
                    if index_block == cache_block[-1]:
                        dh = delta_cache["hidden"]
                        deh = delta_cache.get("encoder_hidden", None)

                        if getattr(self, "enable_delta_pred", True):
                            assert hasattr(self, "delta_pred_proxy"), "delta_pred_proxy not found"
                            dh, deh, delta_pred = self.delta_pred_proxy(dh, deh, temb)
                            delta_cache_out = delta_pred
                        else:
                            delta_cache_out = delta_cache

                        hidden_states = hidden_states + dh
                        if deh is not None:
                            encoder_hidden_states = encoder_hidden_states + deh

                else:
                    if index_block == cache_block[0]:
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

                    # controlnet residual
                    if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                        interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                        hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]
                    
                    if index_block == cache_block[-1]:
                        block_delta_hid = hidden_states - inp_hidden
                        block_delta_enhid = None if (encoder_hidden_states is None or inp_encoder_hidden is None) else (encoder_hidden_states - inp_encoder_hidden)

                        delta_cache_out = {"hidden": block_delta_hid}
                        if block_delta_enhid is not None:
                            delta_cache_out["encoder_hidden"] = block_delta_enhid

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
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
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            # return (output,)
            return (output, delta_cache_out, delta_pred)
        
        # return Transformer2DModelOutput(sample=output), delta_cache
        return Transformer2DModelOutput(sample=output)

# cache_forward_v4: multiblocks
def cache_forward_multiblocks(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        delta_cache: torch.LongTensor = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """   
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        delta_cache_out = None
        delta_pred = None

        if self.enable_cachestep:
            cache_step = []
            if self.cnt in cache_step and (self.previous_residual is not None):
                should_calc = False
            else:
                should_calc = True
            self.cnt += 1 
            if self.cnt == self.num_steps:
                self.cnt = 0
        if self.enable_cachestep:
            if not should_calc:
                hidden_states += self.previous_residual
                # print(f"cache step:{self.cnt-1}")
            else:
                # print(f"calc step:{self.cnt-1}")
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

                    # controlnet residual
                    if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                        interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                        hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

                self.previous_residual = hidden_states - ori_hidden_states
        else:
            cache_block = set(self.cache_block)
            delta_cache_out = {"hidden": {}, "encoder_hidden": {}}
            delta_pred = {"hidden": {}, "encoder_hidden": {}}

            for index_block, block in enumerate(self.transformer_blocks):

                # -------- cache branch --------
                if self.enable_cacheblock and (delta_cache is not None) and (index_block in cache_block):
                    k = str(index_block)

                    dh = delta_cache["hidden"][index_block]
                    deh = delta_cache.get("encoder_hidden", {}).get(index_block, None)

                    if getattr(self, "enable_delta_pred", True):
                        dh = self.delta_pred_hidden[k](dh)
                        deh = self.delta_pred_encoder[k](deh) if deh is not None else None
                        delta_pred["hidden"][index_block] = dh
                        if deh is not None:
                            delta_pred["encoder_hidden"][index_block] = deh

                    delta_cache_out["hidden"][index_block] = dh
                    if deh is not None:
                        delta_cache_out["encoder_hidden"][index_block] = deh

                    hidden_states = hidden_states + dh
                    if deh is not None:
                        encoder_hidden_states = encoder_hidden_states + deh
                    continue

                # -------- normal branch --------
                inp_h, inp_eh = hidden_states, encoder_hidden_states

                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            return module(*inputs, return_dict=return_dict) if return_dict is not None else module(*inputs)
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

                if index_block in cache_block:
                    delta_cache_out["hidden"][index_block] = hidden_states - inp_h
                    if inp_eh is not None and encoder_hidden_states is not None:
                        delta_cache_out["encoder_hidden"][index_block] = encoder_hidden_states - inp_eh


        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
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
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            # return (output,)
            return (output, delta_cache_out, delta_pred)
        assert return_dict is False, "cache_forward expects return_dict=False to return (output, delta_cache_out, delta_pred)"
        return Transformer2DModelOutput(sample=output)

# =========================
# Cache Error Correction Tools
# =========================
class DeltaMLP(nn.Module):
    """
    Input/Output: [B, N, D]
    Residual per-token MLP: f(x) = x + Δ(x)
    Initialize Δ(x)=0 so f(x) starts as identity.
    """
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2

        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)

        # Make Δ(x)=0 at init => f(x)=x
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        delta = self.fc2(self.drop(self.act(self.fc1(self.ln(x)))))
        return x + delta

class DeltaResMLP(nn.Module):
    def __init__(self, dim: int, mult: int = 2, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        h = dim * mult
        self.ln = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, h),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(h, dim),
            )
            for _ in range(depth)
        ])
        nn.init.zeros_(self.blocks[-1][-1].weight)
        nn.init.zeros_(self.blocks[-1][-1].bias)

    def forward(self, x):
        x = self.ln(x)
        for b in self.blocks:
            x = x + b(x)
        return x

class DeltaProxyTransformer(nn.Module):
    """
    Fit Δ0 -> Δ1 using ONE JointTransformerBlock.
    Input:
        dh0: [B, N, D]  (hidden delta)
        deh0: [B, S, D] or None (encoder_hidden delta)
        temb: [B, D]    (time/text embedding in SD3)
    Output:
        dh1, deh1 and dict delta_pred
    Identity init:
        dh1 ≈ dh0, deh1 ≈ deh0 at start (via zero-init heads).
    """
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = None,
        use_dual_attention: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block = JointTransformerBlock(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            context_pre_only=False,
            qk_norm=qk_norm,
            use_dual_attention=use_dual_attention,
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.out_h = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_h.weight); nn.init.zeros_(self.out_h.bias)

        self.out_e = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_e.weight); nn.init.zeros_(self.out_e.bias)

    def forward(self, dh0, deh0, temb):
        x = dh0
        c = deh0

        if c is None:
            c = x[:, :1, :]  # [B,1,D]

        c, x = self.block(hidden_states=x, encoder_hidden_states=c, temb=temb)

        dh1 = dh0 + self.out_h(self.drop(x))
        deh1 = None if deh0 is None else (deh0 + self.out_e(self.drop(c)))

        out = {"hidden": dh1}
        if deh1 is not None:
            out["encoder_hidden"] = deh1
        return dh1, deh1, out

def fsdp_safe_global_clip_(modules, max_norm: float, device, eps: float = 1e-6):
    """
    FSDP FULL_SHARD safe clip (no debug prints):
    - all ranks participate in all_reduce
    - compute global grad norm over modules
    - scale local grads by same coef
    """
    import torch
    import torch.distributed as dist

    local_sq = torch.zeros([], device=device, dtype=torch.float32)

    for mod in modules:
        if mod is None:
            continue
        for p in mod.parameters():
            g = p.grad
            if g is None or g.numel() == 0:
                continue
            local_sq += (g.detach().float() ** 2).sum()

    dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
    global_norm = torch.sqrt(local_sq + eps)

    coef = max_norm / (global_norm + eps)
    coef = torch.clamp(coef, max=1.0)

    for mod in modules:
        if mod is None:
            continue
        for p in mod.parameters():
            g = p.grad
            if g is None or g.numel() == 0:
                continue
            g.mul_(coef)
