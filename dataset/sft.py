import numpy as np
import torch
from datasets import load_dataset


def reformat_sft(instruction, input):
    if input:
        prefix = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        )
    else:
        prefix = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n"
            f"### Instruction:\n{input}\n\n### Response:"
        )
    return prefix


def preprocess2(sources, tokenizer, max_length):
    EOS_TOKEN_ID = tokenizer.eos_token_id or 151643
    tokenizer.pad_token_id = tokenizer.pad_token_id or EOS_TOKEN_ID
    IGNORE_TOKEN_ID = -100
    g_input_ids = []
    g_labels = []
    for item in sources:
        instruction = item["instruction"]
        input2 = item["input"]
        output = item["output"]

        prefix = reformat_sft(instruction, input2)

        tokens2 = tokenizer.tokenize(prefix)
        target_ids = tokenizer.convert_tokens_to_ids(tokens2)

        tokens3 = tokenizer.tokenize(output)
        output_ids = tokenizer.convert_tokens_to_ids(tokens3)

        input_ids = target_ids + output_ids + [EOS_TOKEN_ID]
        labels = [IGNORE_TOKEN_ID] * (len(target_ids)) + output_ids + [EOS_TOKEN_ID]
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            yield dict(
                input_ids=input_ids,
                labels=labels,
            )
            continue
        if len(g_input_ids) + len(input_ids) > max_length:
            padding_length = max_length - len(g_input_ids)
            g_input_ids = g_input_ids + [tokenizer.pad_token_id] * padding_length
            g_labels = g_labels + [tokenizer.pad_token_id] * padding_length
            assert len(g_input_ids) == len(g_labels) == max_length
            yield dict(
                input_ids=g_input_ids,
                labels=g_labels,
            )
            g_input_ids, g_labels = [], []
        g_input_ids.extend(input_ids)
        g_labels.extend(labels)

    if len(g_input_ids) > 0:
        input_ids = g_input_ids[:max_length]
        labels = g_labels[:max_length]

        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        labels = labels + [tokenizer.pad_token_id] * padding_length
        assert len(input_ids) == len(labels) == max_length
        yield dict(
            input_ids=input_ids,
            labels=labels,
        )


class DataEngine():
    def __init__(self, tokenizer, micro_batch_size, max_length, checkpoint_step=0, data_path=""):
        self.micro_batch_size = micro_batch_size
        self.max_length = max_length
        self.train_dataset = \
            load_dataset("json", data_files=data_path)
        self.train_dataset = self.train_dataset.shuffle()["train"]
        print(self.train_dataset)
        self.tokenizer = tokenizer
        self.index = checkpoint_step
        self.data = []
        for item in preprocess2(self.train_dataset, self.tokenizer, max_length * micro_batch_size):
            self.data.append(item)
        self.cache_index = 0  # 缓存的数据索引

    def get_data(self):
        for item in self.data:
            input_ids = item["input_ids"]
            labels = item["labels"]
            input_ids = torch.LongTensor(np.asarray(input_ids).reshape(self.micro_batch_size, self.max_length))
            labels = torch.LongTensor(np.asarray(labels).reshape(self.micro_batch_size, self.max_length))
            yield dict(input_ids=input_ids, labels=labels)

    def __len__(self):
        # 只训练前xx条数据
        return len(self.data)
