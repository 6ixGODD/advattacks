import json
from pathlib import Path
import random

from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import InstructBlipForConditionalGeneration
from transformers import InstructBlipProcessor

model_id = Path("models/Salesforce/instructblip-vicuna-7b")
text_dir = Path("data/questions")
image_dir = Path("data/imgs")
device = torch.device("cuda")

processor = InstructBlipProcessor.from_pretrained(model_id, use_fast=True, padding_side="left")
model = InstructBlipForConditionalGeneration.from_pretrained(
    model_id, low_cpu_mem_usage=True, device_map="auto"
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

with torch.no_grad():
    inputs = processor(
        images=image_tensor, text=text, padding=True, padding_side="left", return_tensors="pt"
    ).to(device)

    generate_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.2)

    response = processor.decode(generate_ids[0], skip_special_tokens=True)
    print("<<<", response)
