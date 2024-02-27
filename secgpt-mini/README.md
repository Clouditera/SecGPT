CPU就能跑！

开源地址：https://huggingface.co/clouditera/secgpt-mini

## Docker一键运行
```dockerfile
FROM python:slim

WORKDIR /secgpt-mini

COPY . /secgpt-mini

RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple -r requirements.txt

EXPOSE 7860

CMD ["python", "webdemo.py", "--base_model", "/secgpt-mini/models"]
```
## 手动运行

模型使用方法：

1. 下载模型和源码

2. 安装python3.7 和依赖 pip install -r requirements.txt

3. 运行 python3 webdemo.py —base_model models

4. 输入指令就可以了