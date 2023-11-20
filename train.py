import os.path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, AutoModelForCausalLM

from dataset import pretrain
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
import transformers
import matplotlib.pyplot as plt

# 最大token长度
max_position_embeddings = 2048
# batch size大小
batch_size = 4
# 梯度累积
accumulation_steps = 8
# 训练多少个epoch
num_train_epochs = 10
# 每隔多少步保存一次模型
save_steps = 400
# 每隔多少步打印一次日志
logging_steps = 50
# 学习率
lr = 1e-4
# 预训练地址
pre_train_path = "models/Baichuan-13B-Base"
# 训练数据json地址
dataset_paper = "data.json"
# lora
use_lora = True
pre_lora_train_path = ""  # 如果要继续上一个lora训练，这里填上上一个lora训练的地址
lora_rank = 8
lora_alpha = 32

global_pic = {
    "step": [],
    "loss": []
}
global_step = 0


def save_loss_pic():
    x = global_pic["step"]
    k1 = global_pic["loss"]
    # print(x,k1)

    plt.plot(x, k1, 'o-', color='b', label="loss")  # s-:方形

    plt.xlabel("step")  # 横坐标名字
    plt.ylabel("loss")  # 纵坐标名字

    # plt.legend(loc = "best")

    # plt.show()
    plt.savefig('foo.png')


def prepare_data():
    # security paper
    data_engine = pretrain.DataEngine(
        tokenizer, batch_size, max_position_embeddings,
        data_path=dataset_paper)
    return data_engine


def find_all_linear_names(peft_model):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def prepare_model():
    # 加载模型
    config = transformers.AutoConfig.from_pretrained(
        pre_train_path,
        trust_remote_code=True,
    )
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(pre_train_path, trust_remote_code=True, device_map="auto")
    print("模型加载完毕")
    # 加载lora模型
    if use_lora:
        if pre_lora_train_path:
            model = PeftModel.from_pretrained(model, pre_lora_train_path, is_trainable=True)
            for name, param in model.named_parameters():
                if 'lora' in name or 'Lora' in name:
                    param.requires_grad = True
        else:
            trainable = find_all_linear_names(model)
            print(trainable)
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,
                target_modules=trainable
            )
            model = get_peft_model(model, peft_config)
        print("lora加载完毕")
        model.print_trainable_parameters()  # 打印可训练参数
    else:
        print_model_parameters(model)
    model.supports_gradient_checkpointing = True  # 节约cuda
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model


def save_model(model, path):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(path)


# 打印模型参数
def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params / 1000000}M total:{total_params}')


def train(model, epoch):
    global global_pic, global_step
    data_engine = prepare_data()
    model.train()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    length = len(data_engine)
    pbar = tqdm(range(length))
    step = 0
    running_loss = 0
    epoch_loss = 0
    for item in data_engine.get_data():
        input_ids = item["input_ids"].cuda()
        labels = item["labels"].cuda()
        loss = model.forward(input_ids=input_ids, labels=labels)[0]
        show_loss = loss.mean().item()
        running_loss += show_loss
        epoch_loss += show_loss
        loss = loss.mean() / accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_parameters, 1.0)
        # update model parameters
        if step > 0 and step % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        if step % logging_steps == 0:
            print(f"step: {step}, loss: {running_loss / logging_steps}")
            global_pic["step"].append(global_step)
            global_pic["loss"].append(running_loss / logging_steps)
            running_loss = 0
            save_loss_pic()
        if step != 0 and step % save_steps == 0:
            save_model(model, f"{output_dir}/epoch-{epoch}-step-{step}")
        pbar.set_postfix({
            "step": step,
            "loss": show_loss
        })
        pbar.update(1)
        step += 1
        global_step += 1
    print(f"epoch:{epoch} loss:{epoch_loss / step}")
    global_pic["step"].append(global_step)
    global_pic["loss"].append(epoch_loss / step)
    save_loss_pic()
    pbar.close()
    save_model(model_engine, f"{output_dir}/secgpt-base-epoch-{i + 1}")


if __name__ == "__main__":
    # output
    output_dir = "output"

    tokenizer = AutoTokenizer.from_pretrained(pre_train_path, trust_remote_code=True)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    data_engine = prepare_data()
    model_engine = prepare_model()
    print_model_parameters(model_engine)

    optimizer = AdamW(model_engine.parameters(), lr=lr, correct_bias=True)

    for i in range(num_train_epochs):
        train(model_engine, i)
