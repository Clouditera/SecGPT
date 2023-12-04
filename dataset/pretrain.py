import numpy as np
import torch
from datasets import load_dataset


class DataEngine():
    def __init__(self, tokenizer, micro_batch_size, max_length, checkpoint_step=0, data_path="", fieldName="text"):
        self.MIN_TEXT_LEN = 20

        self.EOS_TOKEN_ID = tokenizer.eos_token_id or tokenizer.eod_id  # for qwen
        self.PAD_TOKEN_ID = tokenizer.pad_token_id or self.EOS_TOKEN_ID
        assert self.EOS_TOKEN_ID

        self.micro_batch_size = micro_batch_size
        self.max_length = max_length
        self.train_dataset = \
            load_dataset("json", data_files=data_path)
        self.train_dataset = self.train_dataset.shuffle()["train"]
        print(self.train_dataset)
        self.tokenizer = tokenizer
        self.index = checkpoint_step
        self.data = []
        self.cache_index = 0  # 缓存的数据索引
        self.fieldName = fieldName
        self.dd = []
        for item in self.collect():
            self.dd.append(item)
        print("数据加载完毕")

    def collect(self):
        index = self.micro_batch_size * self.max_length
        while self.index < len(self.train_dataset):
            if len(self.data) >= index:
                data = self.data[:index]
                self.data = self.data[index:]
                # 判断EOS是否在data
                if self.EOS_TOKEN_ID in data and data.index(self.EOS_TOKEN_ID) < self.MIN_TEXT_LEN:
                    data = data[data.index(self.EOS_TOKEN_ID):]
                    data = data + [self.PAD_TOKEN_ID] * (index - len(data))
                seq = np.asarray(data).reshape(self.micro_batch_size, self.max_length)
                data = torch.LongTensor(seq)
                yield dict(
                    input_ids=data,
                    labels=data,
                )
            else:
                line = self.train_dataset[self.index][self.fieldName]
                self.index += 1
                tokens = self.tokenizer.tokenize(line)
                ids = self.tokenizer.convert_tokens_to_ids(tokens)
                cc = ids + [self.EOS_TOKEN_ID]
                if len(cc) < self.MIN_TEXT_LEN:
                    cc = []
                self.data.extend(cc)
        return

    def get_data(self):
        for item in self.dd:
            yield item

    def __len__(self):
        return len(self.dd)
