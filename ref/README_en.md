# Multimodal Large Model Description

This competition uses the following pre-trained models from the HuggingFace platform as white-box models, specifically:
- **llava-hf/llava-1.5-7b-hf**
- **Qwen/Qwen2.5-VL-7B-Instruct**
- **Salesforce/instructblip-vicuna-7b**
rm -rf /root/autodl-tmp/final_round/models/Salesforce && mkdir -p /root/models/Salesforce && export HF_ENDPOINT=https://hf-mirror.com && hf download Salesforce/instructblip-vicuna-7b --local-dir /root/models/Salesforce/instructblip-vicuna-7b
All these models have undergone safety alignment training and can generate safe and compliant response content on the test dataset provided by the organizer, meeting the relevant requirements of the competition.
hf download Salesforce/instructblip-vicuna-7b --local-dir /root/autodl-tmp/models/Salesforce/instructblip-vicuna-7b
rm models/Qwen/Qwen2.5-VL-7B-Instruct/.cache/huggingface/download/fNjCg19rjPb4Qv_8lvghvyvUVYM=.ef47f634fa57d46ee134edcc09f34085a47da1e16c12a2abe0d67118be6d72ed.incomplete
## Official Download Links

- [HuggingFace Official Platform](https://huggingface.co/)
  - [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
  - [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
  - [Salesforce/instructblip-vicuna-7b](https://huggingface.co/Salesforce/instructblip-vicuna-7b)

## Domestic Mirror Links

- [HuggingFace Domestic Mirror Platform](https://hf-mirror.com)
  - [llava-hf/llava-1.5-7b-hf](https://hf-mirror.com/llava-hf/llava-1.5-7b-hf)
  - [Qwen/Qwen2.5-VL-7B-Instruct](https://hf-mirror.com/Qwen/Qwen2.5-VL-7B-Instruct)
  - [Salesforce/instructblip-vicuna-7b](https://hf-mirror.com/Salesforce/instructblip-vicuna-7b)

## Example Code

Inference example codes are provided for all public white-box models in this competition.The specific files are as follows:
- `llava_infer.py`
- `qwen_infer.py`
- `instruct_blip_infer.py`

Participants can refer to the above example codes for model inference and other related operations.
