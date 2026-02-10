import os
import json
import torch
from tqdm import tqdm

def load_dinov2_embeddings(data_root, device):
    json_path = os.path.join(data_root, "dinov2_info.json")
    with open(json_path, "r") as f:
        info = json.load(f)
    
    all_embeddings = []
    for image_info in tqdm(info["images"], desc="Loading DINOv2 embeddings"):
        emb_path = os.path.join(data_root, "/".join(image_info["emb_pth"].split('/')[-3:]))
        embeddings = torch.load(emb_path, map_location=device)
        pixel_embeds = embeddings["pixel_embeds"]
        pixel_embeds = pixel_embeds.view(-1, pixel_embeds.shape[-1])
        all_embeddings.append(pixel_embeds)
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings

def filter_embeddings_efficient(embeddings, threshold=0.9, chunk_size=1000):
    num_embeddings = embeddings.shape[0]
    device = embeddings.device
    
    # Normalize the embeddings
    normalized_embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    
    filtered_indices = []
    used_mask = torch.zeros(num_embeddings, dtype=torch.bool, device=device)

    for i in tqdm(range(0, num_embeddings, chunk_size), desc="Filtering embeddings"):
        chunk = normalized_embeddings[i:i+chunk_size]
        
        # Compute similarity of this chunk with all embeddings
        similarity_chunk = torch.mm(chunk, normalized_embeddings.t())
        
        for j in range(chunk.shape[0]):
            if used_mask[i+j]:
                continue
            
            similar_indices = torch.where(similarity_chunk[j] >= threshold)[0]
            if not used_mask[similar_indices].any():
                filtered_indices.append(i+j)
                used_mask[similar_indices] = True

    filtered_embeddings = embeddings[filtered_indices]
    return filtered_embeddings

if __name__ == "__main__":
    data_root = "/scratch1/tandt/truck/"  # Update this path as needed
    
    # Set up CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading DINOv2 embeddings...")
    all_embeddings = load_dinov2_embeddings(data_root, device)
    
    print(f"Loaded embeddings shape: {all_embeddings.shape}")
    
    print("Filtering embeddings...")
    # 0.7, 0.75, 0.80, 0.85, 0.90
    threshold = 0.85
    mem_embeddings = filter_embeddings_efficient(all_embeddings, threshold=threshold, chunk_size=5000)
    print(f"Filtered embeddings shape: {mem_embeddings.shape}")
    
    # Save the filtered embeddings
    output_path = os.path.join(data_root, "dinov2", f"mem{int(threshold*100)}.emb")
    # torch.save(mem_embeddings.cpu(), output_path)
    print(f"Saved filtered embeddings to {output_path}")