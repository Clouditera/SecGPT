from typing import List
import torch
from transformers import AutoTokenizer
import json
import random
import numpy as np


# <s>GPT4 Correct User: Hello, how are you?<|end_of_turn|>GPT4 Correct Assistant: I'm doing great. How can I help you today?<|end_of_turn|>GPT4 Correct User: I'd like to show off how chat templating works!<|end_of_turn|>
def make_context(context: List, tokenizer: AutoTokenizer):
    start_token = "<s>"
    stop_token = "<|end_of_turn|>"

    def _tokenizer(text):
        return tokenizer.encode(text, add_special_tokens=False)

    prompt = start_token
    stop_token_ids = _tokenizer(stop_token)
    start_token_ids = _tokenizer(start_token)

    input_ids = start_token_ids.copy()
    labels = start_token_ids.copy()

    for i in range(len(context)):
        if i % 2 == 0:
            pp = f"GPT4 Correct User: {context[i]['content']}"
            prompt += pp
            input_ids.extend(_tokenizer(pp))
            length = len(_tokenizer(pp))
            labels.extend(length * [-100])
        else:
            pp = f"GPT4 Correct Assistant: {context[i]['content']}"
            prompt += pp
            input_ids.extend(_tokenizer(pp))
            labels.extend(_tokenizer(pp))

        prompt += stop_token
        input_ids.extend(stop_token_ids)
        labels.extend(stop_token_ids)
    return prompt, input_ids, labels


class DataEngine():
    def __init__(self, tokenizer, micro_batch_size, max_length, checkpoint_step=0, data_path=""):
        self.micro_batch_size = micro_batch_size
        self.max_length = max_length
        with open(data_path, encoding="utf-8") as f:
            self.train_dataset = json.load(f)
        random.shuffle(self.train_dataset)
        self.tokenizer = tokenizer
        self.index = checkpoint_step
        self.data = []
        stop_token = "<|end_of_turn|>"
        self.stop_token_ids = tokenizer.encode(stop_token, add_special_tokens=False)

        for item in self.train_dataset:
            _, input_ids, labels = make_context(
                item,
                tokenizer
            )
            self.data.append({
                "input_ids": input_ids,
                "labels": labels,
            })

    def get_data(self):
        max_length = self.max_length * self.micro_batch_size
        g_input_ids = []
        g_labels_ids = []
        padding_token = self.stop_token_ids

        for item in self.data:
            input_ids = item["input_ids"]
            labels = item["labels"]
            if len(g_input_ids) + len(input_ids) < max_length:
                g_input_ids.extend(input_ids)
                g_labels_ids.extend(labels)
            else:
                g_input_ids = g_input_ids[:max_length] + (max_length - len(g_input_ids)) * padding_token
                g_labels_ids = g_labels_ids[:max_length] + (max_length - len(g_labels_ids)) * padding_token
                a1 = torch.LongTensor(np.asarray(g_input_ids).reshape(self.micro_batch_size, self.max_length))
                a2 = torch.LongTensor(np.asarray(g_labels_ids).reshape(self.micro_batch_size, self.max_length))
                yield dict(input_ids=a1, labels=a2)
                g_input_ids = input_ids
                g_labels_ids = labels
        if len(g_input_ids) > 0:
            g_input_ids = g_input_ids[:max_length] + (max_length - len(g_input_ids)) * padding_token
            g_labels_ids = g_labels_ids[:max_length] + (max_length - len(g_labels_ids)) * padding_token
            a1 = torch.LongTensor(np.asarray(g_input_ids).reshape(self.micro_batch_size, self.max_length))
            a2 = torch.LongTensor(np.asarray(g_labels_ids).reshape(self.micro_batch_size, self.max_length))
            yield dict(input_ids=a1, labels=a2)

    def __len__(self):
        # 只训练前xx条数据
        return len(self.data)


if __name__ == '__main__':
    chat = [
        {
            "content": "你好"
        },
        {
            "content": "11"
        },
        {
            "content": "你好22"
        },
        {
            "content": "222"
        },
    ]
    # print(make_context(chat, ))
    tokenizer = AutoTokenizer.from_pretrained("FuseAI/FuseChat-7B-VaRM")
    engine = DataEngine(tokenizer, 1, 8192, 0, "D:\数据集\data-index\jibei-清洗\sft-ai\cc.json")
    for item in engine.get_data():
        print(item)
