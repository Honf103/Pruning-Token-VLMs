import json
import os
import random
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class LLaVAInstructDataset(Dataset):
    """
    Load LLaVA-style metadata and resolve image filenames from local folders.

    Expected metadata sample:
    {
        "id": "...",
        "image": "000000215677.jpg",
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ]
    }

    Output format:
    {
        "image": Tensor[3, H, W],
        "instruction": str,
        "text": str,
        "image_path": str,
        "sample_id": str | None,
    }
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        metadata_path: str,
        image_roots: List[str],
        image_size: int = 224,
        max_samples: int = 5000,
        shuffle_before_select: bool = True,
        seed: int = 42,
        require_image: bool = True,
    ):
        self.metadata_path = metadata_path
        self.image_roots = [str(Path(p)) for p in image_roots]
        self.image_size = image_size
        self.max_samples = max_samples
        self.shuffle_before_select = shuffle_before_select
        self.seed = seed
        self.require_image = require_image

        self.samples = self._load_and_filter()

        if len(self.samples) == 0:
            raise RuntimeError(
                "No valid samples found. Check metadata_path and image_roots."
            )

    def _load_json(self):
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_image_path(self, image_name: str) -> Optional[str]:
        """
        Try to find image_name inside any provided image root.

        Search strategy:
        1. root / image_name
        2. recursive search under root by basename
        """
        if image_name is None:
            return None

        image_name = str(image_name).strip()
        basename = os.path.basename(image_name)

        for root in self.image_roots:
            root_path = Path(root)

            # direct
            direct = root_path / basename
            if direct.exists():
                return str(direct)

            # recursive search
            try:
                matches = list(root_path.rglob(basename))
                if len(matches) > 0:
                    return str(matches[0])
            except Exception:
                pass

        return None

    def _is_valid_conversation(self, conversations) -> bool:
        if not isinstance(conversations, list) or len(conversations) < 2:
            return False

        has_human = any(turn.get("from") == "human" for turn in conversations if isinstance(turn, dict))
        has_gpt = any(turn.get("from") == "gpt" for turn in conversations if isinstance(turn, dict))
        return has_human and has_gpt

    def _load_and_filter(self):
        raw = self._load_json()

        if self.shuffle_before_select:
            rng = random.Random(self.seed)
            rng.shuffle(raw)

        valid = []
        missing_images = 0
        bad_convs = 0

        for ex in raw:
            image_name = ex.get("image", None)
            conversations = ex.get("conversations", None)

            if not self._is_valid_conversation(conversations):
                bad_convs += 1
                continue

            image_path = self._resolve_image_path(image_name)

            if self.require_image and image_path is None:
                missing_images += 1
                continue

            valid.append(
                {
                    "id": ex.get("id", None),
                    "image": image_name,
                    "image_path": image_path,
                    "conversations": conversations,
                }
            )

            if len(valid) >= self.max_samples:
                break

        print(f"[LLaVAInstructDataset] loaded valid samples: {len(valid)}")
        print(f"[LLaVAInstructDataset] skipped bad conversations: {bad_convs}")
        print(f"[LLaVAInstructDataset] skipped missing images: {missing_images}")

        return valid

    def __len__(self):
        return len(self.samples)

    def _process_image(self, path: str) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))

        # convert to float tensor [0,1], shape [3,H,W]
        image = torch.from_numpy(
            __import__("numpy").array(image)
        ).float() / 255.0
        image = image.permute(2, 0, 1).contiguous()
        return image

    def _build_prompt_and_full_text(self, conversations):
        """
        Convert conversation list into:
        - instruction: prompt ending with ASSISTANT:
        - text: prompt + answer(s)

        We keep the first full assistant answer after human turns.
        """
        prompt_lines = []
        answer_lines = []

        for turn in conversations:
            speaker = turn.get("from", "")
            value = str(turn.get("value", "")).strip()

            if speaker == "human":
                prompt_lines.append(f"USER: {value}")
            elif speaker == "gpt":
                answer_lines.append(f"ASSISTANT: {value}")

        if len(prompt_lines) == 0:
            prompt = "USER: <image>\nDescribe the image.\nASSISTANT:"
        else:
            prompt = "\n".join(prompt_lines) + "\nASSISTANT:"

        if len(answer_lines) == 0:
            full_text = prompt
        else:
            # use all assistant turns
            full_text = prompt + " " + " ".join(
                line.replace("ASSISTANT: ", "", 1) for line in answer_lines
            )

        return prompt, full_text

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = self._process_image(sample["image_path"])
        instruction, text = self._build_prompt_and_full_text(sample["conversations"])

        return {
            "image": image,
            "instruction": instruction,
            "text": text,
            "image_path": sample["image_path"],
            "sample_id": sample["id"],
        }


def llava_collate_fn(batch):
    """
    Keep strings as lists, stack images into a tensor.
    """
    images = torch.stack([x["image"] for x in batch], dim=0)
    instructions = [x["instruction"] for x in batch]
    texts = [x["text"] for x in batch]
    image_paths = [x["image_path"] for x in batch]
    sample_ids = [x["sample_id"] for x in batch]

    return {
        "image": images,
        "instruction": instructions,
        "text": texts,
        "image_path": image_paths,
        "sample_id": sample_ids,
    }