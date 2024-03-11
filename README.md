## Mini-Chinese-GPT
Recreate a mini GPT model from 0 to 1, 能够在医学领域具备较强的领域性
## 🤖预训练
一个好的预训练基座模型要具备**续写**的能力。
1. **分词器（Tokenizer）**：LLM分词器的构建方式有两种：一种是自己构造词表并训练一个分词器[custom tokenizers](https://github.com/karpathy/llama2.c)，另一种是选择开源模型训练好的分词器，例如ChatGLM2-6B，Llama2等。

   由于llama官方所提供的词表中，中文的部分只有700个，这也是llama中文能力聊胜于无的原因。因此，为了方便使用，本项目选择[ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)的分词器，该词表大小为64793，值得注意的是：这是一个很妙的数字，因为它刚好在uint16的表示范围（0～65535的无符号整数），每一个token只需要两个字节即可表示，当我们的语料较大时候，相比常用的int32可以节省一半的存储空间。

2. **预训练语料（Corpus for pre-training ）**：从LLM技术革命以来，开源中文预训练语料越来越多。本项目本着拾人牙慧的精神，收集并处理了以下几个经典数据集：
      
   | 中文预训练语料                                                                                                                                                                                                                    | 描述                                                            |
   |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
   | Wiki中文百科：[wikipedia-cn-20230720-filtered](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)                                                                                              | 中文Wikipedia的数据                                                |
   | BaiduBaiKe：[百度网盘](https://pan.baidu.com/s/1jIpCHnWLTNYabftavo3DVw?pwd=bwvb) 提取码: bwvb                                                                                                                                      | 中文BaiduBaiKe的数据                                               |
   | C4_zh：[百度网盘 part1](https://pan.baidu.com/s/18O2Tj_PPB718K8gnaWrWUQ) 提取码：zv4r；[百度网盘 part2](https://pan.baidu.com/s/11PTgtUfFXvpNkOige9Iw4w) 提取码：sb83；[百度网盘 part3](https://pan.baidu.com/s/1248QfTS8QHPojYW-0fd5jQ) 提取码：l89d | C4是可用的最大语言数据集之一，收集了来自互联网上超过3.65亿个域的超过1560亿个token。C4_zh是其中的一部分 |
   | WuDaoCorpora：[智源研究院BAAI：WuDaoCorpora Text文本预训练数据集](https://data.baai.ac.cn/details/WuDaoCorporaText)                                                                                                                       | 中文悟道开源的200G数据                                                 |
   | shibing624/medical：[shibing624/medical](https://huggingface.co/datasets/shibing624/medical/tree/main)                                                                                                          | 源自shibing624的一部分医学领域的预训练数据                                    |
   | MNBVC [MNBVC ](https://huggingface.co/datasets/liwu/MNBVC/tree/main)         | MNBVC数据集不但包括主流文化，也包括各个小众文化甚至火星文的数据。MNBVC数据集包括新闻、作文、小说、书籍、杂志、论文、台词、帖子、wiki、古诗、歌词、商品介绍、笑话、糗事、聊天记录等一切形式的纯文本中文数据。数据均来源于互联网收集。|

### 预训练语料预处理
数据预处理采取GPT的通用做法，对语料进行提前分词，对一个样本做完分词后在末尾加上一个结束符号`<eos>`（如果是采用开源模型架构，需要根据具体的模型结束符号比如qwen采用`<|im_end|>`），与下一个样本区分开。然后将所有的训练语料拼接成一个数组（np.uint16）以.bin二进制格式存储到磁盘上。如果语料过大，避免内存溢出，可以选择mmap格式。
```bash
#脚本里面每一个函数对应一个语料库的预处理，搭建新加语料可以自行扩展。
python data_process.py
#运行结束后，会在./data目录下产生pretrain_data.bin文件
```
### 预训练
```bash
#考虑到预训练的运行时间非常久，需要采用程序后台运行的措施，本项目提供一种常用的程序后台运行的操作：
screen -S ambrose    #(创建新的名称为ambrose的screen)
screen -r ambrose    #(进入名称为ambrose的screen)
#如需更新配置文件，请在对应prtrain.sh文件中进行修改
sh script/pretrain.sh
```
   
## 💡SFT指令微调
LLM微调的目的是将预训练模型中的知识引导出来的一种手段，通俗的讲就是教会模型说人话。
1. **微调方法**：自然语言处理目前存在一个重要的范式：一般领域数据的大规模预训练，对特定任务或领域的适应。因此，为了让预训练模型在特定任务或领域有不错的表现，需要对模型进行微调。目前主流的四种微调方法如下：

   ### LLM微调方法
   - **全面微调（Full Fine-tuning）**：使用任务特定数据调整LLM的所有参数。
   - **参数高效精细调整（Parameter Efficient Fine-tuning）**：修改选定参数以实现更高效的适应。例如：LoRA、Adapter、Prefix-tuning、P-tuning以及P-tuning v2。
   - **提示工程（Prompt Engineering）**：改进模型输入以指导模型输出理想结果。
   - **检索增强生成（Retrieval Augmented Generation）**：将提示工程与数据库查询结合，以获得丰富的上下文答案。

   其中Full Fine-tuning和Parameter Efficient Fine-tuning是需要基于特定任务或者垂直领域数据对模型（全部 or 部分）参数进行微调；
   Prompt Engineering和Retrieval Augmented Generation是通过设计模型输入的template，引导模型输出我们想要的内容，不需要对模型参数进行微调。其中RAG是通过外挂数据库的方式，为模型提供领域知识输入。

   由于本项目模型参数（仅有218M左右，与bert-large-340M参数量差不多）并不大，因此选择Full Fine-tuning对特定任务或领域数据进行微调。后续有更大的预训练模型会补充其他微调方法。
2. **SFT微调数据**：LLM在垂直领域的适应已经是2023年的主格调，因此各个领域的SFT语料和微调模型层出不穷。目前已经有大佬整理并持续更新这方面的[最新进展](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)，大家有需要可以自己访问。
   
   本项目主要针对两类SFT语料进行模型微调，如下：
      
   **日常问答SFT数据**：

   | SFT语料                                                                       | 描述                                                                  |
   |-----------------------------------------------------------------------------|---------------------------------------------------------------------|
   | alpaca-zh：[alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh) | 源自shibing624的一部分SFT数据。该数据集是参考Alpaca方法基于GPT4得到的self-instruct数据，约5万条。 |
   | bell：[bell](https://huggingface.co/datasets/BelleGroup/train_1M_CN)         | 源自BelleGroup的一部分SFT数据。包含约100万条由BELLE项目生成的中文指令数据。|
   | huatuo：[
Huatuo26M-Lite](https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite/viewer)         | Huatuo-26M 是目前为止最大的中文医疗问答数据集。此数据集包含了超过2600万个高质量的医疗问答对，涵盖了各种疾病、症状、治疗方式、药品信息等多个方面|

   **医学垂直领域SFT数据**：
         
   | SFT语料                                                                                                                    | 描述                                                                                                                        |
   |--------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
   | shibing624/medical：[shibing624/medical](https://huggingface.co/datasets/shibing624/medical/tree/main)        | 源自shibing624。该数据集不仅包含了预训练语料如上文所述，还包含一部分SFT数据。                                                                             |
   | HuatuoGPT-sft-data-v1：[HuatuoGPT-sft-data-v1](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1) | 源自HuatuoGPT的SFT数据                                                                                                         |
   | DISC-Med-SFT：[HuatuoGPT-sft-data-v1](https://huggingface.co/datasets/Flmc/DISC-Med-SFT) | DISC-Med-SFT Dataset的子集                                                                                                   |
   | ChatMed_Consult-v0.3：[michaelwzhu/ChatMed_Consult-v0.3](https://huggingface.co/datasets/michaelwzhu/ChatMed_Consult_Dataset) | 本数据集, ChatMed-Dataset, 中的query(或者是prompt)来自于互联网上的医疗问诊问题(549,326)，反映了真实世界的不同用户/患者的医疗问诊需求。目前response都是由OpenAI GPT-3.5引擎回答的。 |

### SFT样本构建
因为SFT语料一般较小，我们没必要提前分词，而是在构建Dataloader的时候进行分词构建batch送给模型。所以自行参考dataset_sft.py即可！

基本逻辑如下：
- prompt和answer之间一定要有一个开始符`<bos>`隔开，然后answer后需要一个结束符`<eos>`。
- 计算loss的时候，对prompt部分的loss进行mask，只计算answer部分的loss即可。

```bash
#脚本里面针对alpaca-zh和bell两个SFT语料进行处理，搭建新加SFT语料可以自行扩展。
python sft_data_process.py
#运行结束后，会在./sft_data目录下产生sft_data.csv文件
```
### 全面微调（Full Fine-tuning）
```bash
#微调所需时间一般较短，如需要后台运行，本项目提供一种常用的程序后台运行的操作：
screen -S ambrose    #(创建新的名称为ambrose的screen)
screen -r ambrose    #(进入名称为ambrose的screen)
python script/sft.py
```