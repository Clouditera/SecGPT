import os.path
from typing import Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, AutoModelForCausalLM

from dataset import dpo
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
import transformers
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
dataset_paper = "w8ay/secgpt"
# 训练方式
dpo_beta = 0.3
output_dir = "output"
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
    data_engine = dpo.DataEngine(tokenizer, batch_size, max_position_embeddings,
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


def prepare_ref_model():
    # 加载模型
    config = transformers.AutoConfig.from_pretrained(
        pre_train_path,
        trust_remote_code=True,
    )
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(pre_train_path, trust_remote_code=True, device_map="cuda:2",
                                                 config=config)
    model.eval()
    return model


def prepare_model():
    # 加载模型
    config = transformers.AutoConfig.from_pretrained(
        pre_train_path,
        trust_remote_code=True,
    )
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(pre_train_path, trust_remote_code=True, device_map="auto",
                                                 config=config)
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


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False,
                     ) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]

    loss_mask = labels != -100

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def dpo_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    if reference_free:
        ref_logratios = 0
    else:
        ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = pi_logratios - ref_logratios

    # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
    # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
    # calculates a conservative DPO loss.
    label_smoothing = 0
    losses = (
            -F.logsigmoid(dpo_beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-dpo_beta * logits) * label_smoothing
    )

    chosen_rewards = dpo_beta * (policy_chosen_logps).detach()
    rejected_rewards = dpo_beta * (policy_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def train(model, reference_model, epoch):
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
        chosen_input_ids = item["chosen_input_ids"]
        choose_labels_ids = item["choose_labels_ids"]
        rejected_input_ids = item["rejected_input_ids"]
        reject_labels_ids = item["reject_labels_ids"]
        # with torch.no_grad():
        #     chosen_input_ids = chosen_input_ids.cuda(2)
        #     choose_labels_ids = choose_labels_ids.cuda(2)
        #     rejected_input_ids = rejected_input_ids.cuda(2)
        #     reject_labels_ids = reject_labels_ids.cuda(2)
        #     reference_chosen_logits = reference_model.forward(input_ids=chosen_input_ids,
        #                                                       labels=choose_labels_ids).logits
        #     reference_rejected_logits = reference_model.forward(input_ids=rejected_input_ids,
        #                                                         labels=reject_labels_ids).logits

        chosen_input_ids = chosen_input_ids.cuda()
        choose_labels_ids = choose_labels_ids.cuda()
        rejected_input_ids = rejected_input_ids.cuda()
        reject_labels_ids = reject_labels_ids.cuda()

        policy_chosen_logits = model(input_ids=chosen_input_ids, labels=choose_labels_ids).logits.to(torch.float32)
        policy_chosen_logps = _get_batch_logps(policy_chosen_logits, choose_labels_ids, average_log_prob=False)

        policy_rejected_logits = model(input_ids=rejected_input_ids, labels=reject_labels_ids).logits.to(torch.float32)
        policy_rejected_logps = _get_batch_logps(policy_rejected_logits, reject_labels_ids, average_log_prob=False)
        # reference_chosen_logits = reference_chosen_logits.to(choose_labels_ids.device)
        # reference_rejected_logits = reference_rejected_logits.to(reject_labels_ids.device)
        # reference_chosen_logps = _get_batch_logps(reference_chosen_logits, choose_labels_ids, average_log_prob=False,
        #                                           tokenizer=tokenizer)
        # reference_rejected_logps = _get_batch_logps(reference_rejected_logits, reject_labels_ids,
        #                                             average_log_prob=False, tokenizer=tokenizer)

        loss, chosen_rewards, rejected_rewards = dpo_loss(
            policy_chosen_logps, policy_rejected_logps, torch.FloatTensor(0), torch.FloatTensor(0), reference_free=True)
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
            "loss": show_loss,
            "chosen_rewards": chosen_rewards.item(),
            "rejected_rewards": rejected_rewards.item()
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

    tokenizer = AutoTokenizer.from_pretrained(pre_train_path, trust_remote_code=True)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    model_engine = prepare_model()
    # ref_model = prepare_ref_model()

    optimizer = AdamW(model_engine.parameters(), lr=lr, correct_bias=True)

    for i in range(num_train_epochs):
        train(model_engine, None, i)
