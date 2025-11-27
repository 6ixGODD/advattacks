import json
from pathlib import Path
import random

from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration

model_id = Path("models/Qwen/Qwen2.5-VL-7B-Instruct")
text_dir = Path("data/questions")
image_dir = Path("data/imgs")
device = torch.device("cuda")

processor = AutoProcessor.from_pretrained(
    model_id,
    use_fast=True,
    legacy=False,
    trust_remote_code=True,
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", low_cpu_mem_usage=True, device_map="auto"
)
model.eval()

files = list(text_dir.iterdir())
file = random.choice(files)
typef = file.stem
with open(file, encoding="utf-8") as f:
    data = json.load(f)
k, v = random.choice(list(data.items()))
text = v["Question"]
image = image_dir / typef / "SD" / f"{k}.png"
img = Image.open(image).convert("RGB")
to_tensor = transforms.ToTensor()
image_tensor = to_tensor(img).to(device)

print(">>>", text)
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

with torch.no_grad():
    inputs = processor(images=image_tensor, text=prompt, do_rescale=False, return_tensors="pt").to(
        device
    )
    generate_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.2)

    response = processor.decode(generate_ids[0], skip_special_tokens=True)
    print("<<<", response)
