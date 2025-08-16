import os
import random
from typing import Dict


def set_fast_train_from_env() -> bool:
    return os.getenv("FAST_TRAIN", "0").strip() == "1"


def seed_everything(seed: int = 123) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_hf_token() -> str:
    """Return HF token from env if available."""
    from dotenv import load_dotenv

    load_dotenv()
    return os.getenv("HUGGINGFACE_HUB_TOKEN", "")


def default_training_args() -> Dict:
    return dict(
        output_dir="mistral-7b-style",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=False,
    )


