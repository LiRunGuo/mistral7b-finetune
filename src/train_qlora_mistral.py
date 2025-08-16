import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer

from utils import set_fast_train_from_env, seed_everything, default_training_args
from format_instructions import format_instruction


@dataclass
class TrainConfig:
    model_id: str = "mistralai/Mistral-7B-v0.1"
    output_dir: str = "mistral-7b-style"
    max_seq_length: int = 2048
    use_4bit: bool = True
    gradient_checkpointing: bool = True


def load_base_model(cfg: TrainConfig):
    bnb_config = None
    if cfg.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if cfg.use_4bit else None,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_peft_model(model):
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    return model


def main():
    load_dotenv()
    seed_everything(123)

    cfg = TrainConfig()
    model, tokenizer = load_base_model(cfg)
    model = build_peft_model(model)

    ds = load_dataset("neuralwork/fashion-style-instruct")
    if set_fast_train_from_env():
        # 仅抽样少量数据，快速跑通
        ds = ds.select_columns(["input", "context", "completion"]).shuffle(seed=123)
        ds["train"] = ds["train"].select(range(64))

    def formatting_func(examples):
        outputs = []
        for inp, ctx, comp in zip(examples["input"], examples["context"], examples["completion"]):
            outputs.append(format_instruction({"input": inp, "context": ctx, "completion": comp}))
        return outputs

    train_args = TrainingArguments(**default_training_args())
    train_args.output_dir = cfg.output_dir

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        peft_config=None,  # 已手动注入 LoRA
        max_seq_length=cfg.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=formatting_func,
        args=train_args,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()


