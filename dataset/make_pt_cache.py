import json

import transformers
from tqdm import tqdm
from datasets import load_dataset

model_name_or_path = "models/Qwen-1_8B"
data_path = "datasets/security-paper-datasets/secpaper_1.26g.jsonl"
max_len = 4096


class DataEngine():
    def __init__(self, data_path, tokenizer, max_length=1024, micro_batch_size=1, fieldName="text"):
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
        self.index = 0
        self.data = []
        self.cache_index = 0  # 缓存的数据索引
        self.fieldName = fieldName
        print("数据加载完毕")

    def collect(self):
        index = self.micro_batch_size * self.max_length
        with tqdm(total=len(self.train_dataset)) as pbar:
            while self.index < len(self.train_dataset):
                if len(self.data) >= index:
                    data = self.data[:index]
                    self.data = self.data[index:]
                    # 判断EOS是否在data
                    if self.EOS_TOKEN_ID in data and data.index(self.EOS_TOKEN_ID) < self.MIN_TEXT_LEN:
                        data = data[data.index(self.EOS_TOKEN_ID):]
                        data = data + [self.PAD_TOKEN_ID] * (index - len(data))
                    # seq = np.asarray(data).reshape(self.micro_batch_size, self.max_length)
                    # data = torch.LongTensor(seq)
                    yield dict(
                        text=data,
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
                    pbar.update(1)
        return


tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path,
    model_max_length=max_len,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True,
)
tokenizer.pad_token_id = tokenizer.eod_id
print("准备中")
train_dataset = DataEngine(data_path, tokenizer=tokenizer, max_length=max_len)
print("formatting dataset...")
with open(data_path + "_cache.jsonl", "a+") as f:
    for item in train_dataset.collect():
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print("数据加载完毕...")
