import csv
import glob
import random

from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Specify paths and hyperparameters for quantization
model_path = "models/DeepSeek-R1-Distill-Qwen-7B"
quant_path = "models/DeepSeek-R1-Distill-Qwen-7B-Awq"
quant_config = {"zero_point": True,
                "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
# max_memory = {0: "8GB", "cpu": "24GB"}

# Load your tokenizer and model with AutoAWQ
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoAWQForCausalLM.from_pretrained(model_path,
                                           device_map="auto",
                                           # device_map="cpu",
                                           # max_memory=max_memory,
                                           safetensors=True)


def load_math500():
    data = load_dataset('dataset/Math-500', split="test[:100]")
    return [text.strip() for text in data["problem"] if text.strip() != '']


def load_coding():
    data = load_dataset('dataset/python_coding', split="train[:100]")
    return [text.strip() for text in data["problem"] if text.strip() != '']


def load_reasoning():
    data = load_dataset('dataset/natural_reasoning', split="train[:100]")
    return [text.strip() for text in data["question"] if text.strip() != '']


def load_aimo2():
    data = []
    for file in glob.glob('dataset/AIMO-*/*.csv'):
        with open(file) as f:
            reader = csv.reader(f)
            headers = next(reader)
            problem_index = headers.index('problem')
            for row in reader:
                data.append(row[problem_index])
    return data


data = []
for problem in load_math500():
    text = tokenizer.apply_chat_template([
        {"role": "user", "content": "Please reason step by step. Return final answer within \\boxed{}."},
        {"role": "user", "content": problem},
    ], tokenize=False, add_generation_prompt=True)
    data.append(text)

for problem in load_reasoning() + load_coding():
    text = tokenizer.apply_chat_template([
        {"role": "user", "content": problem},
    ], tokenize=False, add_generation_prompt=True)
    data.append(text)

for problem in load_aimo2():
    text = tokenizer.apply_chat_template([
        {"role": "user", "content": "Solve the math problem from the user. Only submit an answer if you are sure. Return final answer within \\boxed{}."},
        {"role": "user", "content": problem},
    ], tokenize=False, add_generation_prompt=True)
    data.append(text)

    text = tokenizer.apply_chat_template([
        {"role": "user", "content": "请通过逐步推理来解答问题，并把最终答案放置于\\boxed{}中。"},
        {"role": "user", "content": problem},
    ], tokenize=False, add_generation_prompt=True)
    data.append(text)

print(f'start quantize, data length: {len(data)}')

# Quantize
model.quantize(tokenizer,
               quant_config=quant_config,
               calib_data=data,
               n_parallel_calib_samples=4,
               max_calib_samples=256,
               max_calib_seq_len=4096)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
