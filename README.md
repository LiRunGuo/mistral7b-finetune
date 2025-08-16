## 微调 Mistral 7B（QLoRA 实战项目）

本项目依据 `微调Mistral7B.txt` 文档内容复刻与整理，提供使用 Hugging Face Transformers + PEFT + TRL 进行 QLoRA 微调的可运行示例；并附带基线与微调后推理脚本。

目标场景：生成式推荐（以“时尚搭配推荐”为例）。

---

### 目录结构

```
微调Mistral7B项目/
  ├─ README.md
  ├─ requirements.txt
  ├─ .env.example
  └─ src/
      ├─ utils.py
      ├─ format_instructions.py
      ├─ prepare_data.py
      ├─ train_qlora_mistral.py
      ├─ infer_finetuned.py
      └─ infer_base.py
```

---

### 环境建议

- Python 3.10（推荐）
- NVIDIA GPU + CUDA（强烈建议，QLoRA/7B 训练推理需显存与算力）
- Windows 用户建议 WSL2 + CUDA，或使用 Linux 环境

安装依赖：
```
pip install -r requirements.txt
```

如需从 Hugging Face 拉取模型/数据，建议配置访问令牌：
```
复制 .env.example 为 .env 并填写 HUGGINGFACE_HUB_TOKEN
```

> 重要：`mistralai/Mistral-7B-v0.1` 模型较大，4-bit 量化可降低显存占用，但仍需较强 GPU 资源。资源不足可将训练脚本中的 `model_id` 替换为更小的开源模型做演示，流程一致。

---

### 数据集

- 使用 `neuralwork/fashion-style-instruct` 指令风格数据集（Hugging Face）。
- 结构包含 `input`（体型与风格）、`context`（场景/事件）与 `completion`（目标搭配）。

运行示例：
```
python src/prepare_data.py
```

---

### 训练（QLoRA）

训练脚本要点：
- 使用 BitsAndBytes 4-bit 量化（NF4）加载基座模型
- 使用 PEFT 的 LoRA（q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj）
- 使用 TRL 的 `SFTTrainer` 进行监督微调
- 提供 `FAST_TRAIN=1` 环境变量以抽样少量数据快速验证流程

启动训练：
```
# 可选：快速演示模式，仅抽样少量数据
# 在 PowerShell: $env:FAST_TRAIN = "1"

python src/train_qlora_mistral.py
```

训练产物：默认输出到 `mistral-7b-style` 目录（包含 LoRA 权重等）。

---

### 推理

- 基线模型推理（未微调）：
```
python src/infer_base.py
```

- 微调后推理（合并 LoRA 权重方式加载）：
```
python src/infer_finetuned.py
```

脚本会随机抽取一条训练样本，构造指令并生成 5 套搭配作为输出示例。

---

### 官方仓库路径（可选）

- 如需体验官方 `mistral-src` 参考实现，请参阅其 README 操作，包括下载官方 `.tar` 模型与 Demo/Interactive；本项目以 Hugging Face Transformers + PEFT + TRL 为主线，便于快速复现与扩展。

---

### 常见问题

- BitsAndBytes 在 Windows 原生环境兼容性有限，建议 WSL2 或 Linux；或使用更小模型验证流程。
- 显存不够：尝试减小 `per_device_train_batch_size`、开启 `gradient_checkpointing`、或改用更小模型。
- 速度慢：减少 `num_train_epochs`，开启 `FAST_TRAIN`，或仅测试推理脚本。

---

### 参考

- Mistral 7B 论文：`https://arxiv.org/abs/2310.06825`
- TRL SFTTrainer：`https://huggingface.co/docs/trl/sft_trainer`
- Transformers Trainer：`https://huggingface.co/docs/transformers/main_classes/trainer`


