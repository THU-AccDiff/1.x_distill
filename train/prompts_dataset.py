import os
from typing import List, Dict, Any, Optional
import json

import torch
from torch.utils.data import Dataset


class SimplePromptsDataset(Dataset):
    """
    Load prompts from a single .txt file 
    """
    def __init__(self, prompts_path: str, local_rank: int = 0):
        if not os.path.exists(prompts_path):
            raise FileNotFoundError(prompts_path)

        self.prompts: List[str] = []
        with open(prompts_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.prompts.append(line)

        self.local_rank = local_rank

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "prompts": self.prompts[idx],
            "indices": idx,       
        }

    @staticmethod
    def collate_fn(batch):
        return {
            "prompts": [b["prompts"] for b in batch],
            "indices": torch.tensor([b["indices"] for b in batch], dtype=torch.long),
        }



class JourneyDBPromptDataset(Dataset):
    """
    Only load prompt from the .json file of JourneyDB:
      {
        "img_path": "...",
        "prompt": "...",
        "ori_prompt": "...",
        "Task1": {...},
        "Task2": {...},
        ...
      }
    """

    def __init__(
        self,
        jsonl_path: str,
        local_rank: int = 0,
        max_samples: Optional[int] = None,
    ):
        self.local_rank = local_rank
        self.prompts: List[str] = []

        if self.local_rank == 0:
            print(f"[JourneyDBPromptDataset] Loading jsonl from: {jsonl_path}")

        with open(jsonl_path, "rt", encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    if self.local_rank == 0:
                        print(f"[JourneyDBPromptDataset] JSON decode error at line {line_id}")
                    continue

                prompt = obj.get("prompt") or obj.get("ori_prompt")
                if not prompt:
                    continue

                prompt = str(prompt).strip()
                if not prompt:
                    continue

                self.prompts.append(prompt)

                if max_samples is not None and len(self.prompts) >= max_samples:
                    break

        if self.local_rank == 0:
            print(f"[JourneyDBPromptDataset] Loaded {len(self.prompts)} prompts")

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "prompts": self.prompts[idx], 
            # "indices": idx,            
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts = [b["prompts"] for b in batch]
        # indices = torch.tensor([b["indices"] for b in batch], dtype=torch.long)
        return {
            "prompts": prompts,
            # "indices": indices,
        }
