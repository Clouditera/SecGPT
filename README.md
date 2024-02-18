
# SecGPT 网络安全大模型
探索使用网络安全知识训练大模型，能达到怎样的能力边界。

**模型下载**
- [huggingface](https://huggingface.co/w8ay/secgpt)
- [wisemodel](https://wisemodel.cn/models/w8ay/secgpt)


## 模型介绍
SecGPT的愿景是将人工智能技术引入网络安全领域，以提高网络防御的效率和效果。其使命是推动网络安全智能化，为社会提供更安全的数字生活环境。


SecGPT可以作为基座安全模型，用于探索各种网络安全任务。以下是对SecGPT在网络安全任务中可能应用的进一步探讨：
### 1. 漏洞分析：
SecGPT可以与安全研究人员或开发团队进行多轮对话，共同分析和审查应用程序或系统中的潜在漏洞。它可以帮助识别和理解漏洞的性质、潜在影响以及可能的修复方法。
### 2. 溯源分析：
在网络入侵事件调查中，SecGPT可以与调查员合作，协助分析网络流量、日志和事件记录，以追踪攻击者的活动路径，从而支持安全溯源分析。
### 3. 流量分析：
SecGPT可以分析网络流量数据，识别异常流量模式，帮助检测潜在的网络攻击或入侵行为，并提供应对这些行为的建议。
### 4. 攻击研判：
在面对未知攻击时，SecGPT可以与安全团队一起进行多轮对话，共同分析攻击的特征、模式和可能的来源，以便更好地理解和应对威胁。
### 5. 命令解释：
SecGPT可以解释和分析安全相关的命令、脚本或代码，帮助安全专家理解其功能、潜在风险和执行结果。
### 6. 网安知识问答：
SecGPT可以用于回答网络安全领域的问题，为安全从业者提供支持和解释，包括解释安全概念、提供最佳实践建议和分析最新的威胁趋势。


相较于其他的开源模型，SecGPT有以下几个重要的特色：
### 1. 自写训练代码，节省显存： 
SecGPT提供了自己的训练代码，包括预训练、有监督学习和DPO强化对齐等功能。这使得即使使用小型显卡，也能够训练大规模的网络安全模型，为更多的开发者和研究者提供了参与网络安全研究的机会。
### 2. 高质量网络安全训练集： 
SecGPT提供了精选的高质量网络安全类无监督训练集，使模型能够学习网络安全问题。这有助于提高模型对于网络安全领域的理解。
### 3. DPO强化学习： 
通过DPO强化学习方法，SecGPT引入了安全专家的见解和答案，与模型原始输出一起作为数据进行训练。这种方法有助于模型更好地模拟安全专家的思维方式，提高了模型输出的实用性和准确性，使其更符合安全领域的需求。
### 4. 无道德限制：
在网络安全领域，无道德限制的GPT模型具有优势。比如模型可以深入分析和解释恶意软件的代码、行为和影响，而不受道德限制的束缚。当然，需要注意的是，无道德限制并不意味着可以不受限制地使用这些模型。在执行敏感网络安全任务时，仍然需要谨慎考虑法律、道德和隐私问题，并遵循适用的法律法规和道德准则。


## 效果展示
### 1. 代码审计&解释&修复建议
![image](https://github.com/lixiaofangyun/secgpt/assets/160388899/926a0ade-402a-40da-81bd-377656be5f47)
![image](https://github.com/lixiaofangyun/secgpt/assets/160388899/0ad934a6-28a0-4f50-a9f0-85160911ff67)
### 2. 反汇编伪代码分析
![image](https://github.com/lixiaofangyun/secgpt/assets/160388899/5fb5a846-2a4b-4248-babf-b83e2f39c047)
![image](https://github.com/lixiaofangyun/secgpt/assets/160388899/3f4430bb-3e3e-4bd4-8bd6-235c981be3f8)
### 3. 钓鱼邮件编写
![image](https://github.com/lixiaofangyun/secgpt/assets/160388899/1c94368c-a4e0-45d2-8a9c-9b6e91465040)
### 4. 命令解释
![image](https://github.com/lixiaofangyun/secgpt/assets/160388899/d68c8a6e-1961-4f52-b142-2168ba46bbfb)
### 5. 日志分析
![image](https://github.com/lixiaofangyun/secgpt/assets/160388899/ebfa7266-c80b-432f-a9d3-e47385ff8606)


## 模型训练
目前，SecGPT不仅开源了基座模型，还开源了所有的训练代码以及数据（包括预训练，有监督训练，DPO训练等大模型全流程训练代码）和网络安全训练数据。我们鼓励任何人都使用这些代码和数据去训练自己的网络安全大模型！


该模型的训练过程大致如下图所示：
![image](https://github.com/lixiaofangyun/secgpt/assets/160388899/0ba32fca-5694-4d96-a4cd-0f8942449635)
### 基座模型
(1) 选用Baichuan-13B模型作为基座模型进行训练(无道德限制，较好中文支持，显存资源占用小)


(2) 训练环境与配置:


① webdemo推理：2*4090(24G)


② lora训练：3*4090(24G)


### 微调技术


(1) 基于Lora做预训练和SFT训练


(2) 优化后的训练代码可展示训练的底层知识


(3) 大幅减少训练的显存占用，可在3*4090显卡上训练


### 训练数据


(1) 预训练数据


① 收集安全书籍，安全知识库，安全论文，安全社区文章，漏洞库等等安全内容。


② 数据集开源地址：


https://huggingface.co/datasets/w8ay/security-paper-datasets


![image](https://github.com/lixiaofangyun/secgpt/assets/160388899/4cc1f583-7954-4c20-a4e2-1fa27bb2b285)
(2) 有监督训练数据


① 利用ChatGPT+人工构造各类有监督安全能力数据集，让模型能了解各类安全指令。


![image](https://github.com/lixiaofangyun/secgpt/assets/160388899/9d08fb01-4499-4b91-99db-2bb0ea001c90)
② 思维链：基于思维链方式构造有监督数据集让模型能够根据问题逐步推理到最终答案，展现推理过程。


③ 知乎回答：加入了部分高质量知乎数据集，在一些开放性问题上模型能通过讲故事举例子等方式回答答案和观点，更易读懂。


④ 为防止灾难性遗忘，在有监督数据中，通用能力数据：安全能力数据，约为5:1。


### 模型训练


(1) 超参数信息配置：


![image](https://github.com/lixiaofangyun/secgpt/assets/160388899/f7a26738-bc15-4722-9008-990b36e6334c)


(2) 预训练配置：


![image](https://github.com/lixiaofangyun/secgpt/assets/160388899/5c16dd12-6abf-45cd-b98e-dde681fb465d)


(3) SFT训练配置：


![image](https://github.com/lixiaofangyun/secgpt/assets/160388899/4636dbe0-1ddf-4c03-8087-831588f4ba06)


(4) RLHF强化学习校准：


为了改善模型的输出质量，有以下两种策略可供选择。首先，可以通过使用更明确的提示词来引导模型，使其更专注于用户需求，从而生成更有针对性的回答。其次，可以采用RLHF强化学习方法，模型将生成多个答案，用户可以手动选择最优的答案，在后面进行强化学习，从而不断提高模型对齐的能力。


SecGPT模型具备内置的RLHF选择器，每次会输出三个不同的答案。用户可以根据需求选择最佳答案，同时这也为后续的RLHF微调提供了有价值的数据，有助于模型进一步提升输出的质量。这些方法共同作用，有助于确保模型输出满足用户的预期，并在网络安全等领域中发挥更大的作用。


![image](https://github.com/lixiaofangyun/secgpt/assets/160388899/49c52e9b-9f79-42b6-9f43-9e2668408c76)


关于SecGPT网络安全大模型，这只是我们分享的一部分内容。云起无垠致力于将SecGPT开源大模型分享给更多的安全从业者，我们鼓励大家积极参与共同的研讨和交流，以不断改进和增强其功能。通过开源的方式，我们也期望加强网络安全从业者对这一大模型的理解和应用。


如果您有任何关于该模型的问题或建议，请随时在GitHub上提交issue或PR。我们热切欢迎各位共同参与建设，让SecGPT网络安全大模型能够惠及网络安全领域的每一位需求者！共同的努力将有助于提高网络安全的水平，确保数字生活的安全性。


① SecGPT开源地址：


https://github.com/Clouditera/secgpt


② 模型地址：


https://huggingface.co/w8ay/secgpt


③ 数据集地址：


https://huggingface.co/datasets/w8ay/security-paper-datasets



#### 开放问题
![img.png](images/image-2.png)
![Alt text](images/image-3.png)
![Alt text](images/image-4.png)
