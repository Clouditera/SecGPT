from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("YeungNLP/bloom-820m-zh", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("pretrain_bloom_820m", trust_remote_code=True)
print("加载完毕")


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


question = "中国的首都是北京，美国的首都是"
_, question = reformat_sft(question)
generation_kwargs = {
    "top_p": 0.8,
    "temperature": 0.8,
    "max_length": 200,
    "no_repeat_ngram_size": 4,
    "do_sample": True,
}
for i in range(5):
    inputs = tokenizer.encode(question, return_tensors='pt', truncation=True)
    output = model.generate(input_ids=inputs, **generation_kwargs)[0]
    # output = output[inputs.shape[1]:]
    print(f"第{i + 1}个输出")
    print(tokenizer.decode(output))
