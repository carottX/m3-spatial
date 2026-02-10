import os
import glob
import json
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor
import torch

from .modeling_siglip import SiglipModel
from .utils import visualize_feature, visualize_feature_kmeans, visualize_feature_gmm, visualize_mask_logits
from ..dinov2.utils import visualize_feature

data_root = "/scratch1/tandt/truck/"
input_folder = os.path.join(data_root, "images")
output_folder = os.path.join(data_root, "siglip")
embed_folder = os.path.join(output_folder, "embeds")
json_pth = os.path.join(data_root, "siglip_info.json")
device = "cuda"

if not os.path.exists(embed_folder):
    os.makedirs(embed_folder)

image_pths = sorted(glob.glob(os.path.join(input_folder, "*.[jJ][pP][gG]")))

# Initialize model and processor
print("Loading model and processor...")
model = SiglipModel.from_pretrained("google/siglip-so400m-patch14-384",
                                    device_map=device,)
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
patch_size = model.config.vision_config.patch_size

info = {"information": '''
        pixel_embeds: pixel features after the vision head, in (h, w, c). \n
        last_hidden_state: pixel features before the vision head, in (h, w, c). \n
        Note: \n
        1. All the features are before normalization. \n
        ''',
        "images": []}

print(f"Processing {len(image_pths)} images...")
for image_pth in tqdm(image_pths, desc="Processing images", unit="image"):
    with torch.no_grad():
        image = Image.open(image_pth).convert('RGB')
        width, height = image.size
        inputs = processor(text=["713"], images=image, padding="max_length", return_tensors="pt")
        inputs.to(device)
        _height, _width = inputs.pixel_values.shape[-2:]
        emb_height, emb_width = _height // patch_size, _width // patch_size

        with torch.autocast(device_type='cuda'):
            outputs = model(**inputs, interpolate_pos_encoding=True)

        last_hidden_state = outputs.vision_model_output.last_hidden_state
        pooler_output = outputs.vision_model_output.pooler_output
        image_embeds = outputs.image_embeds
        
        mask_logits = outputs.logits_per_image
        mask_logits = mask_logits.view(_height//patch_size, _width//patch_size, -1)
                
        image_embeds = image_embeds.view(emb_height, emb_width, -1)
        pooler_output = pooler_output.view(emb_height, emb_width, -1)
        last_hidden_state = last_hidden_state[0].view(emb_height, emb_width, -1)
        
        output = {
            "pixel_embeds": pooler_output.cpu(),
            "pixel_embeds_normed": image_embeds.cpu(),
            "last_hidden_state": last_hidden_state.cpu(),
        }
        emb_pth = os.path.join(embed_folder, os.path.basename(image_pth).replace(".jpg", ".emb")).replace(".JPG", ".emb")
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
        
        visualize_mask_logits(mask_logits, image, size=(_height//patch_size, _width//patch_size), filename="siglip_logits.png", alpha=0.5)

print("Saving metadata to JSON...")
json.dump(info, open(json_pth, "w"))
print("Processing complete!")