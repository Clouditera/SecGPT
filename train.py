import json
import os.path

import matplotlib.pyplot as plt
import torch
import transformers
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, AutoModelForCausalLM

from dataset import pretrain, sft, chatml

global_pic = {
    "step": [],
    "loss": []
}
global_step = 0


def save_loss_pic():
    output_dir = config["output_dir"]
    x = global_pic["step"]
    k1 = global_pic["loss"]
    # print(x,k1)
    plt.plot(x, k1, 'o-', color='b', label="loss")  # s-:方形
    plt.xlabel("step")  # 横坐标名字
    plt.ylabel("loss")  # 纵坐标名字
    # plt.legend(loc = "best")

    # plt.show()
    plt.savefig(output_dir + '/foo.png')


def prepare_data():
    # 预训练
    train_option = config["train_option"]
    batch_size = config["batch_size"]
    max_position_embeddings = config["max_position_embeddings"]
    dataset_path = config["dataset_path"]
    if train_option == "pretrain":
        data_engine = pretrain.DataEngine(
            tokenizer, batch_size, max_position_embeddings,
            data_path=dataset_path)
    elif train_option == "sft":
        data_engine = sft.DataEngine(tokenizer, batch_size, max_position_embeddings,
                                     data_path=dataset_path)
    elif train_option == "chatml":
        data_engine = chatml.DataEngine(tokenizer, batch_size, max_position_embeddings,
                                        data_path=dataset_path)
    else:
        raise ValueError("train_option must be one of pretrain, sft, pretrain_cache")
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
    pre_train_path = config["pre_train_path"]
    # 加载模型
    mode_config = transformers.AutoConfig.from_pretrained(
        pre_train_path,
        trust_remote_code=True,
    )
    mode_config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(pre_train_path, trust_remote_code=True, config=mode_config,
                                                 device_map="auto")
    print("模型加载完毕")
    # 加载lora模型
    if config["use_lora"]:
        if config["pre_lora_train_path"]:
            model = PeftModel.from_pretrained(model, config["pre_lora_train_path"], is_trainable=True)
            for name, param in model.named_parameters():
                if 'lora' in name or 'Lora' in name:
                    param.requires_grad = True
        else:
            trainable = find_all_linear_names(model)
            print(trainable)
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config["lora_rank"],
                lora_alpha=config["lora_alpha"],
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
    accumulation_steps = int(config["accumulation_steps"])
    logging_steps = int(config["logging_steps"])
    save_steps = int(config["save_steps"])
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
        if step != 0 and step % logging_steps == 0:
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train.json",
                        help="预训练模型路径")
    configparser = parser.parse_args()

    with open(configparser.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config["pre_tokenizer_path"], trust_remote_code=True)
    output_dir = config["output_dir"]
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    model_engine = prepare_model()
    lr = config["learning_rate"]
    optimizer = AdamW(model_engine.parameters(), lr=lr, correct_bias=True)
    for i in range(int(config["num_train_epochs"])):
        train(model_engine, i)
