import os
import glob
import json

from PIL import Image

import torch
from transformers import AutoImageProcessor
from .modeling_dinov2 import Dinov2Model
from .utils import visualize_feature

data_root = "/scratch1/tandt/truck/"
input_folder = os.path.join(data_root, "images")
output_folder = os.path.join(data_root, "dinov2")
embed_folder = os.path.join(output_folder, "embeds")
json_pth = os.path.join(data_root, "dinov2_info.json")
device = "cuda"

if not os.path.exists(embed_folder):
    os.makedirs(embed_folder)

image_pths = sorted(glob.glob(os.path.join(input_folder, "*.jpg")))

device = "cuda" if torch.cuda.is_available() else "cpu"
_height, _width = 224, 224
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-giant')
model = Dinov2Model.from_pretrained('facebook/dinov2-giant').to(device)
patch_size = model.config.patch_size

info = {"information": '''
        pixel_embeds: pixel features of last hidden states, in (h, w, c). \n
        image_embeds: image features of last hidden states, in (1, c). \n
        Note: \n
        1. All the features are before normalization. \n
        ''',
        "images": []}

for image_pth in image_pths:
    with torch.no_grad():
        image_ori = Image.open(image_pth).convert('RGB')
        width, height = image_ori.size

        inputs = processor(images=image_ori, return_tensors="pt").to(device)
        outputs = model(**inputs)

        embeddings = outputs.last_hidden_state
        emb_height, emb_width = _height//patch_size, _width//patch_size
        cls_, fea_ = embeddings[:, 0], embeddings[:, 1:].reshape(1, emb_height, emb_width, -1)
        
        output = {
                "pixel_embeds": fea_[0].cpu(), 
                "image_embeds": cls_[0].cpu()
                }
        emb_pth = os.path.join(embed_folder, image_pth.split("/")[-1].replace(".jpg", ".emb"))
        torch.save(output, emb_pth)
        
        image_info = {
            "file_name": image_pth.split("/")[-1],
            "image_id": image_pth.split("/")[-1].split(".")[0],
            "height": height,
            "width": width,
            "emb_height": emb_height,
            "emb_width": emb_width,
            "emb_pth": emb_pth,
        }
        info["images"].append(image_info)
        visualize_feature(fea_.detach().cpu(), size=(emb_height, emb_width), filename="dinov2_logits.png")

json.dump(info, open(json_pth, "w"))