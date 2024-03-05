import json
import random
from typing import Tuple, List

import numpy as np
import torch
from transformers import PreTrainedTokenizer, AutoTokenizer


def make_context(
        tokenizer: PreTrainedTokenizer,
        history: List[Tuple[str, str]] = None,
        system: str = "",
        max_window_size: int = 6144,
):
    if history is None:
        history = []

    im_start, im_end = "<|im_start|>", "<|im_end|>"
    im_start_tokens = [tokenizer.im_start_id]
    im_end_tokens = [tokenizer.im_end_id]
    nl_tokens = tokenizer.encode("\n")

    def _tokenize_str(role, content):
        return f"{role}\n{content}", tokenizer.encode(
            role, allowed_special=set()
        ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

    system_text, system_tokens_part = _tokenize_str("system", system)
    system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

    raw_text = ""
    context_tokens = []
    labels = []

    for turn_query, turn_response in reversed(history):
        query_text, query_tokens_part = _tokenize_str("user", turn_query)
        query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
        response_text, response_tokens_part = _tokenize_str(
            "assistant", turn_response
        )
        response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

        next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
        label_context_tokens = nl_tokens + im_start_tokens + len(query_tokens_part) * [
            -100] + im_end_tokens + nl_tokens + response_tokens
        prev_chat = (
            f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
        )

        current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
        )
        if current_context_size < max_window_size:
            context_tokens = next_context_tokens + context_tokens
            labels = label_context_tokens + labels
            raw_text = prev_chat + raw_text
        else:
            break

    context_tokens = system_tokens + context_tokens
    labels = im_start_tokens + len(system_tokens_part) * [-100] + im_end_tokens + labels

    assert len(context_tokens) == len(labels)
    if len(context_tokens) < max_window_size:
        padding_size = max_window_size - len(context_tokens)
        context_tokens = context_tokens + (im_end_tokens * padding_size)
        labels = labels + (im_end_tokens * padding_size)

    raw_text = f"{im_start}{system_text}{im_end}" + raw_text
    return raw_text, context_tokens, labels


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
        for item in self.train_dataset:
            _, input_ids, labels = make_context(
                tokenizer,
                history=item,
                system="You are a helpful assistant",
                max_window_size=max_length,
            )
            self.data.append({
                "input_ids": input_ids,
                "labels": labels,
            })

    def get_data(self):
        for item in self.data:
            input_ids = item["input_ids"]
            labels = item["labels"]
            input_ids = torch.LongTensor(np.asarray(input_ids).reshape(1, self.max_length))
            labels = torch.LongTensor(np.asarray(labels).reshape(1, self.max_length))
            yield dict(input_ids=input_ids, labels=labels)

    def __len__(self):
        # 只训练前xx条数据
        return len(self.data)
