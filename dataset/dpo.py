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
            f"### Instruction:\n{instruction}\n\n### Response:"
        )
    return prefix


def preprocess2(sources, tokenizer, max_length):
    EOS_TOKEN_ID = tokenizer.eos_token_id or tokenizer.eod_id  # for qwen
    tokenizer.pad_token_id = tokenizer.pad_token_id or EOS_TOKEN_ID
    IGNORE_TOKEN_ID = -100
    g_input_ids = []
    g_labels = []
    for item in sources:
        instruction = item["instruction"]
        input2 = item["input"]
        choose = item["output"][0]
        rejected = item["output"][1]

        prefix = reformat_sft(instruction, input2)

        tokens2 = tokenizer.tokenize(prefix)
        target_ids = tokenizer.convert_tokens_to_ids(tokens2)

        tokens3 = tokenizer.tokenize(choose)
        choose_ids = tokenizer.convert_tokens_to_ids(tokens3)

        tokens4 = tokenizer.tokenize(rejected)
        reject_ids = tokenizer.convert_tokens_to_ids(tokens4)

        choose_inputs = target_ids + choose_ids + [EOS_TOKEN_ID]
        choose_labels = [IGNORE_TOKEN_ID] * (len(target_ids)) + choose_ids + [EOS_TOKEN_ID]
        choose_inputs = choose_inputs[:max_length]
        choose_labels = choose_labels[:max_length]
        padding_length = max_length - len(choose_inputs)
        if padding_length > 0:
            choose_inputs = choose_inputs + [tokenizer.pad_token_id] * padding_length
            choose_labels = choose_labels + [IGNORE_TOKEN_ID] * padding_length

        reject_inputs = target_ids + reject_ids + [EOS_TOKEN_ID]
        reject_labels = [IGNORE_TOKEN_ID] * (len(target_ids)) + reject_ids + [EOS_TOKEN_ID]
        reject_inputs = reject_inputs[:max_length]
        reject_labels = reject_labels[:max_length]
        padding_length = max_length - len(reject_inputs)
        if padding_length > 0:
            reject_inputs = reject_inputs + [tokenizer.pad_token_id] * padding_length
            reject_labels = reject_labels + [IGNORE_TOKEN_ID] * padding_length

        assert len(choose_inputs) == len(choose_labels) == max_length
        assert len(reject_inputs) == len(reject_labels) == max_length
        yield dict(
            chosen_input_ids=choose_inputs,
            choose_labels_ids=choose_labels,
            rejected_input_ids=reject_inputs,
            reject_labels_ids=reject_labels,
        )


class DataEngine():
    def __init__(self, tokenizer, micro_batch_size, max_length, checkpoint_step=0, data_path=""):
        self.max_length = max_length
        self.train_dataset = \
            load_dataset("json", data_files=data_path)
        self.train_dataset = self.train_dataset.shuffle()["train"]
        print(self.train_dataset)
        self.tokenizer = tokenizer
        self.index = checkpoint_step
        self.data = []
        for item in preprocess2(self.train_dataset, self.tokenizer, max_length):
            self.data.append(item)

    def get_data(self):
        for item in self.data:
            for k, v in item.items():
                item[k] = torch.LongTensor(v).reshape(1, self.max_length)
            yield item

    def __len__(self):
        # 只训练前xx条数据
        return len(self.data)
