import random

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
# 参数解析
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="", help="模型名称")
parser.add_argument("--tokenizer", type=str, default="", help="分词器名称")
args = parser.parse_args()

# bloom基类
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.tokenizer, device_map="auto", trust_remote_code=True)
# model = model.to(device)

# baichuan基类
# tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-7B", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("pretrain_baichuan/001", trust_remote_code=True)
# model = model.to(device)

# chatglm
# tokenizer = AutoTokenizer.from_pretrained("../models/chatglm2-6b", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("../models/chatglm2-6b", device_map="auto",trust_remote_code=True)

# baichuan7b
# from transformers import AutoModelForCausalLM, AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("../models/Baichuan-7B", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("../models/Baichuan-7B", device_map="auto", trust_remote_code=True)

# baichuan13b
# from transformers import AutoModelForCausalLM, AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("../models/Baichuan-13B-Base", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("../models/Baichuan-13B-Base", device_map="auto", trust_remote_code=True)

# Qwen-7B
# tokenizer = AutoTokenizer.from_pretrained("../models/Qwen-7B", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("../models/Qwen-7B", device_map="auto",trust_remote_code=True)

# Atom-7B
# tokenizer = AutoTokenizer.from_pretrained("../models/Atom-7B", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("../models/Atom-7B", device_map="auto",trust_remote_code=True)

print("模型加载完毕")
import json

with open("eval.json", encoding="utf-8") as f:
    datasets = json.load(f)

print("数据加载完毕", len(datasets))
from tqdm import tqdm


def build_example(data, with_answer: bool = True):
    question = data["Question"]
    choice = "\n".join(
        [
            "A. " + data["A"],
            "B. " + data["B"],
            "C. " + data["C"],
            "D. " + data["D"],
        ]
    )
    answer = data["Answer"].strip().upper() if with_answer else ""
    return f"{question}\n{choice}\n答案：{answer}"


def run(datasets, shot=3):
    results = []
    acc = 0
    correct = 0
    incorrect = 0

    system_prompt = f"你是专业的网络安全红队专家,以下是渗透测试考试的单项选择题，请选出其中的正确答案。\n"
    if shot != 0:
        for i in range(shot):
            system_prompt += "\n" + build_example(datasets[i], with_answer=True)
    datasets = datasets[shot:]
    for data in tqdm(datasets):
        prompt = system_prompt + "\n" + build_example(data, with_answer=False)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output = model.generate(
            input_ids,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            temperature=0.8,
            top_p=0.7
        )
        scores = output.scores[0][0].to(torch.float32)
        label_score = []
        candidates = ["A", "B", "C", "D"]
        for can in candidates:
            can_id = tokenizer.encode(can)[-1]
            label_score.append(scores[can_id].item())
        answer = candidates[np.argmax(label_score)]
        results.append(
            {
                "prompt": prompt,
                "correct": answer == data["Answer"].strip().upper(),
                "answer": answer,
                "question": data["Question"]
            }
        )
        acc += answer == data["Answer"].strip().upper()
        if answer == data["Answer"].strip().upper():
            correct += 1
        else:
            incorrect += 1
    acc /= len(datasets)
    return results, acc, correct, incorrect


if __name__ == '__main__':
    results, acc, correct, incorrect = run(datasets, shot=5)
    print(f"acc:{acc},correct:{correct},incorrect:{incorrect}")
    for item in results:
        if item["correct"]:
            print("[正确]", item["question"])
        else:
            print("[错误]", item["question"])
