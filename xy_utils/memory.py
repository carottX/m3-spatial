import torch
import torch.nn.functional as F

def softmax_wt(logits, temperature=1, dim=-1):
    scaled_logits = logits / temperature
    sftmax = F.softmax(scaled_logits, dim=dim)
    # print(sftmax.max())
    return sftmax

# Define a function to apply softmax in chunks along the flattened n dimension
def softmax_in_chunks(x, chunk_size=1000, dim=-1, temperature=1.0):
    # Flatten the tensor from [h, w, d] to [n, d] where n = h * w
    h,w,d = x.shape
    x = x.view(h*w, d)
    
    # Split along the n dimension into smaller chunks
    n_chunks = torch.split(x, chunk_size, dim=0)

    # del x
    torch.cuda.empty_cache()
    
    # Apply softmax to each chunk along the last dimension (d)
    softmax_chunks = [softmax_wt(chunk, temperature=temperature, dim=dim) for chunk in n_chunks]
    
    del n_chunks
    torch.cuda.empty_cache()
    
    # Concatenate the chunks back together along the n dimension
    start_idx = 0
    for chunk in softmax_chunks:
        end_idx = min(start_idx + len(chunk), len(x))
        x[start_idx:end_idx,:] = chunk

    return x.view(h, w, d)

def emb_to_memory(embeddings, memory, _eval=False, _temp=1.0, _return_similarity=False):
    norm_emb = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-6)
    norm_mem = memory / (memory.norm(dim=-1, keepdim=True) + 1e-6)
    similarity = norm_emb @ norm_mem.t()
    
    if _eval:
        del embeddings
        torch.cuda.empty_cache()

    # Try regular softmax, and if OOM occurs, use chunked softmax
    try:
        similarity = softmax_wt(similarity, temperature=_temp, dim=-1)
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e) and _eval:
            print("OOM error occurred. Using chunked softmax.")
            similarity = softmax_in_chunks(similarity, chunk_size=1000, dim=-1, temperature=_temp)
        else:
            raise  # Re-raise the error if it's not OOM

    raw_feature = similarity @ memory

    if _return_similarity:
        return raw_feature, similarity

    return raw_feature

# def emb_to_memory(embeddings, memory, _eval=False, _temp=1.0, _chunk=False):
#     norm_emb = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-6)
#     norm_mem = memory / (memory.norm(dim=-1, keepdim=True) + 1e-6)
#     similarity = norm_emb @ norm_mem.t()
    
#     if _eval:
#         del embeddings
#         torch.cuda.empty_cache()

#     # Try regular softmax, and if OOM occurs, use chunked softmax
#     if not _chunk:
#         similarity = softmax_wt(similarity, temperature=_temp, dim=-1)
#     else:
#         similarity = softmax_in_chunks(similarity, chunk_size=1, dim=-1, temperature=_temp)

#     raw_feature = similarity @ memory
#     return raw_feature    

def index_to_raw(index, projection, memory, _eval=False, _temp=1.0, _return_similarity=False):
    '''
    index: [c,h,w]
    projection: [c,d]
    memory: [n,d]
    '''
    index = index.half()
    projection = projection.half()
    memory = memory.half()

    embeddings = index.permute(1,2,0) @ projection # NOTE XYZ: Learnable embeddings
    return emb_to_memory(embeddings, memory, _eval=_eval, _temp=_temp, _return_similarity=_return_similarity)

def points_index_to_raw(index, projection, memory, _eval=False, _temp=1.0):
    '''
    index: [c,l]
    projection: [c,d]
    memory: [n,d]
    '''
    memory = memory.type_as(index)
    projection = projection.type_as(index)
    embeddings = index.t() @ projection # NOTE XYZ: Learnable embeddings
    return emb_to_memory(embeddings, memory, _eval=_eval, _temp=_temp)