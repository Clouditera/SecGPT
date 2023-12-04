import json
import torch


# 加载缓存中的预训练数据集

class DataEngine():
    def __init__(self, tokenizer, micro_batch_size, max_length, data_path="", fieldName="text"):
        self.train_dataset = []
        with open(data_path, "r") as f:
            for item in f.readlines():
                item = item.strip()
                if not item:
                    continue
                self.train_dataset.append(json.loads(item))
        self.tokenizer = tokenizer
        self.data = []
        self.fieldName = fieldName
        self.dd = []
        print("数据加载完毕")

    def get_data(self):
        for item in self.train_dataset:
            data = item["text"]
            data = torch.LongTensor(data)
            yield dict(
                input_ids=data,
                labels=data,
            )

    def __len__(self):
        return len(self.train_dataset)
