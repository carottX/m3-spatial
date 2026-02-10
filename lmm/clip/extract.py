import os
import glob
import json

from .load import load
from ..siglip.utils import visualize_mask_logits
from ..dinov2.utils import visualize_feature

from PIL import Image
import clip

import torch
import torchvision.transforms as T
import torch.nn.functional as F


data_root = "/scratch1/tandt/truck/"
input_folder = os.path.join(data_root, "images")
output_folder = os.path.join(data_root, "clip")
embed_folder = os.path.join(output_folder, "embeds")
json_pth = os.path.join(data_root, "clip_info.json")
device = "cuda"

if not os.path.exists(embed_folder):
    os.makedirs(embed_folder)

image_pths = sorted(glob.glob(os.path.join(input_folder, "*.jpg")))

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load("ViT-L/14@336px", device=device)
text = clip.tokenize(["713"]).to(device)

patch_size = model.visual.patch_size

input_size = 336
transform = T.Compose([
    T.Resize(input_size),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

info = {"information": '''
        pixel_embeds: pixel features after the vision head, in (h, w, c). \n
        image_embeds: image features after the vision head, in (1, c). \n
        Note: \n
        1. All the features are before normalization. \n
        ''',
        "images": []}

for idx, image_pth in enumerate(image_pths):
    print(idx, len(image_pths), image_pth)
    with torch.no_grad():
        image_ori = Image.open(image_pth).convert('RGB')
        width, height = image_ori.size
        # image = preprocess(image_ori).unsqueeze(0).to(device)
        image = transform(image_ori).unsqueeze(0).cuda()
        _height, _width = image.shape[-2:]

        image_features, pixel_features = model.encode_image(image)
        text_features = model.encode_text(text)

        emb_height, emb_width = _height//patch_size, _width//patch_size
        output = {
            "pixel_embeds": pixel_features[0].view(emb_height, emb_width, -1).cpu(),
            "image_embeds": image_features[0].cpu(),
        }        
        emb_pth = os.path.join(embed_folder, os.path.basename(image_pth).replace(".jpg", ".emb"))
        torch.save(output, emb_pth)

        # visualize_feature(pixel_features.float().cpu(), size=(emb_height, emb_width), filename="clip_pca.png")

        pixel_features = pixel_features.view(-1, pixel_features.size(-1))
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        pixel_features = pixel_features / pixel_features.norm(dim=1, keepdim=True)
        
        logits_per_pixel = pixel_features @ text_features.T
        mask_logits = logits_per_pixel.view(emb_height, emb_width, -1).cpu()
        
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
        visualize_mask_logits(mask_logits, image_ori, size=(_height//patch_size, _width//patch_size), filename="clip_logits.png")

json.dump(info, open(json_pth, "w"))