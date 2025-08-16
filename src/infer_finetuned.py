import os
from random import randrange
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from format_instructions import format_instruction


def main():
    load_dotenv()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model_id = "mistralai/Mistral-7B-v0.1"
    output_dir = "mistral-7b-style"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
    )
    base_model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    ft_model = PeftModel.from_pretrained(base_model, output_dir)

    ds = load_dataset("neuralwork/fashion-style-instruct")
    sample = ds["train"][randrange(len(ds["train"]))]
    prompt = format_instruction(sample)

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(device)
    with torch.inference_mode():
        outputs = ft_model.generate(
            input_ids=input_ids,
            max_new_tokens=800,
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
        )

    outputs = outputs.detach().cpu().numpy()
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    output = outputs[0][len(prompt) :]

    print(f"Instruction:\n{sample['input']}\n")
    print(f"Context:\n{sample['context']}\n")
    print(f"Ground truth:\n{sample['completion']}\n")
    print(f"Generated output:\n{output}\n")


if __name__ == "__main__":
    main()


