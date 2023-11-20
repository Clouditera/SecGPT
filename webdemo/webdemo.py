import json
import sys
from threading import Thread
from queue import Queue

import gradio as gr
import torch
from peft import PeftModel
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
import time

if torch.cuda.is_available():
    device = "auto"
else:
    device = "cpu"


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


class TextIterStreamer:
    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=False):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.tokens = []
        self.text_queue = Queue()
        # self.text_queue = []
        self.next_tokens_are_prompt = True

    def put(self, value):
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
        else:
            if len(value.shape) > 1:
                value = value[0]
            self.tokens.extend(value.tolist())
            word = self.tokenizer.decode(self.tokens, skip_special_tokens=self.skip_special_tokens)
            # self.text_queue.append(word)
            self.text_queue.put(word)

    def end(self):
        # self.text_queue.append(None)
        self.text_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get()
        if value is None:
            raise StopIteration()
        else:
            return value


def main(
        base_model: str = "",
        lora_weights: str = "",
        share_gradio: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    if lora_weights:
        model = PeftModel.from_pretrained(
            model,
            lora_weights
        )

    model.eval()

    def evaluate(
            instruction,
            temperature=0.1,
            top_p=0.75,
            max_new_tokens=128,
            repetition_penalty=1.1,
            **kwargs,
    ):
        print(instruction,
              temperature,
              top_p,
              max_new_tokens,
              repetition_penalty,
              **kwargs)
        if not instruction:
            return
        prompt = reformat_sft(instruction, "")

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()

        if not (1 > temperature > 0):
            temperature = 1
        if not (1 > top_p > 0):
            top_p = 1
        if not (2000 > max_new_tokens > 0):
            max_new_tokens = 200
        if not (5 > repetition_penalty > 0):
            repetition_penalty = 1.1

        output = ['', '', '']
        for i in range(3):
            if i > 0:
                time.sleep(0.5)
            streamer = TextIterStreamer(tokenizer)
            generation_config = dict(
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                streamer=streamer,
            )
            c = Thread(target=lambda: model.generate(input_ids=input_ids, **generation_config))
            c.start()
            for text in streamer:
                output[i] = text
                yield output[0], output[1], output[2]
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(output)

    def fk_select(select_option):
        def inner(context, answer1, answer2, answer3, fankui):
            print("反馈", select_option, context, answer1, answer2, answer3, fankui)
            gr.Info("反馈成功")
            data = {
                "context": context,
                "answer": [answer1, answer2, answer3],
                "choose": ""
            }
            if select_option == 1:
                data["choose"] = answer1
            elif select_option == 2:
                data["choose"] = answer2
            elif select_option == 3:
                data["choose"] = answer3
            elif select_option == 4:
                data["choose"] = fankui
            with open("fankui.jsonl", 'a+', encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        return inner

    with gr.Blocks() as demo:
        gr.Markdown(
            "# 云起无垠SecGPT模型RLHF测试\n\nHuggingface: https://huggingface.co/w8ay/secgpt\nGithub: https://github.com/Clouditera/secgpt")
        with gr.Row():
            with gr.Column():  # 列排列
                context = gr.Textbox(
                    lines=3,
                    label="Instruction",
                    placeholder="Tell me ..",
                )
                temperature = gr.Slider(
                    minimum=0, maximum=1, value=0.3, label="Temperature"
                )
                topp = gr.Slider(
                    minimum=0, maximum=1, value=0.7, label="Top p"
                )
                max_tokens = gr.Slider(
                    minimum=1, maximum=2000, step=1, value=300, label="Max tokens"
                )
                repetion = gr.Slider(
                    minimum=0, maximum=10, value=1.1, label="repetition_penalty"
                )
            with gr.Column():
                answer1 = gr.Textbox(
                    lines=4,
                    label="回答1",
                )
                fk1 = gr.Button("选这个")
                answer2 = gr.Textbox(
                    lines=4,
                    label="回答2",
                )
                fk2 = gr.Button("选这个")
                answer3 = gr.Textbox(
                    lines=4,
                    label="回答3",
                )
                fk3 = gr.Button("选这个")
                fankui = gr.Textbox(
                    lines=4,
                    label="反馈回答",
                )
                fk4 = gr.Button("都不好，反馈")
        with gr.Row():
            submit = gr.Button("submit", variant="primary")
            gr.ClearButton([context, answer1, answer2, answer3, fankui])
        submit.click(fn=evaluate, inputs=[context, temperature, topp, max_tokens, repetion],
                     outputs=[answer1, answer2, answer3])
        fk1.click(fn=fk_select(1), inputs=[context, answer1, answer2, answer3, fankui])
        fk2.click(fn=fk_select(2), inputs=[context, answer1, answer2, answer3, fankui])
        fk3.click(fn=fk_select(3), inputs=[context, answer1, answer2, answer3, fankui])
        fk4.click(fn=fk_select(4), inputs=[context, answer1, answer2, answer3, fankui])

    demo.queue().launch(server_name="0.0.0.0", share=share_gradio)
    # Old testing code follows.


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='云起无垠SecGPT模型RLHF测试')
    parser.add_argument("--base_model", type=str, required=True, help="基础模型")
    parser.add_argument("--lora", type=str, help="lora模型")
    parser.add_argument("--share_gradio", type=bool, default=False, help="开放外网访问")
    args = parser.parse_args()
    main(args.base_model, args.lora, args.share_gradio)
