# 多模态大模型说明

本次竞赛选用HuggingFace平台上的以下预训练模型作为白盒模型，具体如下：
- **llava-hf/llava-1.5-7b-hf**
- **Qwen/Qwen2.5-VL-7B-Instruct**
- **Salesforce/instructblip-vicuna-7b**

这些模型均经过安全对齐训练，在主办方提供的测试数据集上能够生成安全合规的响应内容，可满足竞赛相关要求。

## 官方下载地址

- [HuggingFace 官方平台](https://huggingface.co/)
  - [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
  - [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
  - [Salesforce/instructblip-vicuna-7b](https://huggingface.co/Salesforce/instructblip-vicuna-7b)

## 国内镜像地址

- [HuggingFace 国内镜像平台](https://hf-mirror.com)
  - [llava-hf/llava-1.5-7b-hf](https://hf-mirror.com/llava-hf/llava-1.5-7b-hf)
  - [Qwen/Qwen2.5-VL-7B-Instruct](https://hf-mirror.com/Qwen/Qwen2.5-VL-7B-Instruct)
  - [Salesforce/instructblip-vicuna-7b](https://hf-mirror.com/Salesforce/instructblip-vicuna-7b)

## 示例代码

本次竞赛公开的白盒模型均提供了推理示例代码，具体文件如下：
- `llava_infer.py`
- `qwen_infer.py`
- `instruct_blip_infer.py`

参赛者可参考上述示例代码进行模型推理等相关操作。
