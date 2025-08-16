import os
from dotenv import load_dotenv
from datasets import load_dataset


def main():
    load_dotenv()
    ds = load_dataset("neuralwork/fashion-style-instruct")
    print(ds)
    # 打印一个样本
    sample = ds["train"][0]
    print({k: sample[k] for k in ["input", "context", "completion"]})


if __name__ == "__main__":
    main()


