from typing import Dict


def format_instruction(sample: Dict) -> str:
    return f"""You are a personal stylist recommending fashion advice and clothing combinations. Use the self body and style description below, combined with the event described in the context to generate 5 self-contained and complete outfit combinations.
### Input:
{sample['input']}
### Context:
{sample['context']}
### Response:
{sample['completion']}
"""


def format_instruction_cn(sample: Dict) -> str:
    return f"""你是一名个人造型师，负责推荐时尚建议和服装搭配。使用下面的自我主体和风格描述，结合上下文中描述的事件，生成 5 套自成一体的完整服装搭配。
#### 输入：
{sample['input']}
### 背景：
{sample['context']}
### 回应：
{sample['completion']}
"""


