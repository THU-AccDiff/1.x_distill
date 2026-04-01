import os
import shutil
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from safetensors.torch import load_file, save_file


def merge_lora_weights(
        base_weighst: torch.Tensor,
        lora_A_weights: torch.Tensor,
        lora_B_weights: torch.Tensor,
        scale: float
    ) -> torch.Tensor:
    device = base_weighst.device
    dtype = base_weighst.dtype

    # In case users wants to merge the adapter weights that are in
    # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
    # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
    cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

    if cast_to_fp32:
        base_weighst = base_weighst.float()
        lora_A_weights = lora_A_weights.float()
        lora_B_weights = lora_B_weights.float()

    output_tensor = base_weighst + lora_B_weights @ lora_A_weights * scale

    if cast_to_fp32:
        output_tensor = output_tensor.to(dtype=dtype)

    return output_tensor


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--lora_weight_path", type=str,
                        default="/data/transformer/diffusion_pytorch_model.safetensors")
    parser.add_argument("--save_dir", type=str, default="/cache//merge_lora_weight")
    parser.add_argument("--config_path", default="/cache/stable-diffusion-3-medium-diffusers/transformer/config.json")
    parser.add_argument("--lora_scale", default=1.0)

    args = parser.parse_args()
    return args


def main(args):
    lora_scale = args.lora_scale  # lora_alpha / lora_rank
    base_model_config_path = args.config_path
    lora_weight_path = args.lora_weight_path
    save_dir = args.save_dir
    assert os.path.exists(base_model_config_path)
    assert os.path.exists(lora_weight_path)

    lora_weights_dict = load_file(lora_weight_path, device="cpu")
    lora_weights_dict = dict([(k.replace("transformer.", ""), v) for k, v in lora_weights_dict.items()])
    base_weights_dict = dict([(k.replace("base_layer.", ""), v) for k, v in lora_weights_dict.items() if "lora" not in k])
    lora_weights_dict = dict([(k, v) for k, v in lora_weights_dict.items() if "lora" in k])

    # get lora layer list
    name_list = [k.replace(".lora_A.default.weight", "").replace(".lora_B.default.weight", "")
                 for k in lora_weights_dict.keys()]
    name_list = sorted(list(set(name_list)))
    if len(name_list) == 0:
        print("not find valid lora name.")
        return

    # merge lora weights to base model
    for k in tqdm(name_list):
        assert f"{k}.weight" in base_weights_dict, f"{k}.weight"
        new_weights = merge_lora_weights(
            base_weighst=base_weights_dict[f"{k}.weight"],
            lora_A_weights=lora_weights_dict[f"{k}.lora_A.default.weight"],
            lora_B_weights=lora_weights_dict[f"{k}.lora_B.default.weight"],
            scale=lora_scale
        )
        base_weights_dict[f"{k}.weight"] = new_weights

    # save merged weights
    os.makedirs(save_dir, exist_ok=True)
    save_file(base_weights_dict, os.path.join(save_dir, "diffusion_pytorch_model.safetensors"))
    shutil.copy(base_model_config_path, os.path.join(save_dir, "config.json"))


if __name__ == '__main__':
    args_ = arg_parser()
    main(args_)
