import os
import yaml
import random
from datetime import datetime
from datetime import timezone, timedelta
from logging import Logger, Formatter, StreamHandler, FileHandler
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
import numpy as np


class Dist:
    @staticmethod
    def is_dist():
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def get_local_world_size():
        if dist.is_available() and dist.is_initialized():
            if "LOCAL_WORLD_SIZE" in os.environ:
                return int(os.environ['LOCAL_WORLD_SIZE'])
            return torch.cuda.device_count()

        return 1

    @staticmethod
    def get_world_size():
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()

        return 1

    @staticmethod
    def get_num_nodes():
        if dist.is_available() and dist.is_initialized():
            if "LOCAL_WORLD_SIZE" in os.environ:
                return dist.get_world_size() // int(os.environ['LOCAL_WORLD_SIZE'])
            return dist.get_world_size() // torch.cuda.device_count()

        return 1

    @staticmethod
    def get_node_rank():
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() // torch.cuda.device_count()

        return 0


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    else:
        raise EnvironmentError("init distributed failed, can't find RANK and WORLD_SIZE env")

    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank})', flush=True)
    dist.init_process_group(backend=args.dist_backend,
                            world_size=args.world_size,
                            rank=args.rank)
    dist.barrier()


def read_yaml(yml_path: str):
    with open(yml_path, mode="rt", encoding="utf-8") as f:
        parse_info = yaml.safe_load(f)

    return parse_info


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def add_terminal_handler(logger: Logger, log_fmt: Optional[Formatter] = None):
    if log_fmt is None:
        log_fmt = Formatter("[%(levelname)s] %(asctime)s %(filename)s[%(lineno)d]:%(message)s")

    stream_handler = StreamHandler()
    stream_handler.setFormatter(log_fmt)
    logger.addHandler(stream_handler)


def add_file_handler(logger: Logger,
                     log_path: Optional[str] = None,
                     log_fmt: Optional[Formatter] = None):
    if log_fmt is None:
        log_fmt = Formatter("[%(levelname)s] %(asctime)s %(filename)s[%(lineno)d]:%(message)s")

    if log_path is None:
        log_path = datetime.now(tz=timezone(offset=timedelta(hours=8))).strftime("%Y_%m_%d_%H_%M_%S") + ".log"

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_handler = FileHandler(log_path)
    file_handler.setFormatter(log_fmt)
    logger.addHandler(file_handler)


class CustomLogger:
    def __init__(self,
                 name: str,
                 local_rank: int,
                 rank: int,
                 to_terminal: bool = True,
                 to_file: bool = False,
                 save_path: Optional[str] = None,
                 fmt: Optional[Formatter] = None):
        self.local_rank = local_rank
        self.rank = rank
        self.local_rank = local_rank
        self._logger = Logger(name)
        if to_terminal:
            add_terminal_handler(self._logger, log_fmt=fmt)
        if to_file:
            add_file_handler(self._logger, log_path=save_path, log_fmt=fmt)

    def debug(self, msg, *args, main_process_only: bool = False, local_main_process_only: bool = False) -> None:
        if main_process_only and self.rank != 0:
            return
        if local_main_process_only and self.local_rank != 0:
            return
        self._logger.debug(msg, *args, stacklevel=2)

    def info(self, msg, *args, main_process_only: bool = False, local_main_process_only: bool = False) -> None:
        if main_process_only and self.rank != 0:
            return
        if local_main_process_only and self.local_rank != 0:
            return
        self._logger.info(msg, *args, stacklevel=2)

    def warning(self, msg, *args, main_process_only: bool = False, local_main_process_only: bool = False) -> None:
        if main_process_only and self.rank != 0:
            return
        if local_main_process_only and self.local_rank != 0:
            return
        self._logger.warning(msg, *args, stacklevel=2)

    def error(self, msg, *args, main_process_only: bool = False, local_main_process_only: bool = False) -> None:
        if main_process_only and self.rank != 0:
            return
        if local_main_process_only and self.local_rank != 0:
            return
        self._logger.error(msg, *args, stacklevel=2)

    def exception(self, e, main_process_only: bool = False, local_main_process_only: bool = False):
        if main_process_only and self.rank != 0:
            return
        if local_main_process_only and self.local_rank != 0:
            return
        self._logger.exception(e, stacklevel=2)


def save_fsdp_weights(model: FSDP,
                      save_path: str,
                      sub_folder: str = "pytorch_model_fsdp_0"
                      ):
    save_policy = ShardedStateDictConfig()
    with FSDP.state_dict_type(
        model, StateDictType.SHARDED_STATE_DICT, save_policy
    ):
        ckpt_dir = os.path.join(save_path, sub_folder)
        os.makedirs(ckpt_dir, exist_ok=True)
        state_dict = {"model": model.state_dict()}

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=dist_cp.FileSystemWriter(ckpt_dir),
            planner=DefaultSavePlanner()
        )


class FakeProfileOps:
    def step(self):
        pass

    def stop(self):
        pass


class FakeProfile:
    def __enter__(self):
        return FakeProfileOps()

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def step(self):
        pass

    def stop(self):
        pass
