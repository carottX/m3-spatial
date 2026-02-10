import os
import json
from argparse import ArgumentParser

import torch
import torch.nn.functional as F

import utils.general_utils as utils
from arguments import (
    AuxiliaryParams,
    ModelParams,
    PipelineParams,
    OptimizationParams,
    DistributionParams,
    BenchmarkParams,
    DebugParams,
    get_combined_args,
    init_args,
    print_all_args,
)
from gaussian_renderer import GaussianModel
from scene import Scene
from utils.general_utils import init_distributed, safe_state, set_args, set_log_file


SUPPORTED_MODELS = ["clip", "siglip", "dinov2", "seem", "llama3", "llamav"]


@torch.no_grad()
def compute_mem_xyz_from_gaussians(
    xyz: torch.Tensor,
    index_features: torch.Tensor,
    projection: torch.Tensor,
    memory: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 2048,
    topk: int = 0,
    return_topk: bool = False,
) -> torch.Tensor:
    """Compute a 3D coordinate for each memory entry via weighted average.

    Given gaussian primitives i with 3D coordinates x_i and a soft assignment w_ij
    to memory entry j, compute:
        c_j = sum_i w_ij * x_i / (sum_i w_ij + eps)

    Args:
        xyz: [N, 3] gaussian coordinates.
        index_features: [N, C] gaussian per-primitive index/code features.
        projection: [C, D] projection matrix.
        memory: [M, D] memory bank entries.
        temperature: softmax temperature.
        chunk_size: kept for backward compatibility; no longer used.

    Returns:
        If return_topk is False:
            mem_xyz: [M, 3] coordinate per memory entry.
        If return_topk is True:
            (mem_xyz, topk_info) where topk_info contains:
                topk_weights: [M, K]
                topk_indices: [M, K] (gaussian indices)
                topk_xyz: [M, K, 3]
                topk_valid: [M, K] (bool mask)
    """

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected xyz as [N,3], got {tuple(xyz.shape)}")
    if index_features.ndim != 2:
        raise ValueError(
            f"Expected index_features as [N,C], got {tuple(index_features.shape)}"
        )
    if projection.ndim != 2:
        raise ValueError(f"Expected projection as [C,D], got {tuple(projection.shape)}")
    if memory.ndim != 2:
        raise ValueError(f"Expected memory as [M,D], got {tuple(memory.shape)}")

    device = xyz.device
    xyz_f = xyz.float()

    # Normalize memory once.
    mem_f = memory.float()
    mem_norm = mem_f / (mem_f.norm(dim=-1, keepdim=True) + 1e-6)

    m = memory.shape[0]

    topk = int(topk) if topk is not None else 0
    if topk < 0:
        raise ValueError(f"topk must be >= 0, got {topk}")

    # Compute full logits and weights without chunking.
    proj_f = projection.float()
    emb = index_features.float() @ proj_f  # [N, D]
    emb_norm = emb / (emb.norm(dim=-1, keepdim=True) + 1e-6)

    logits = emb_norm @ mem_norm.t()  # [N, M]
    weights = F.softmax(logits / float(temperature), dim=-1).to(torch.float32)  # [N, M]
    print(weights.max())
    denom = weights.sum(dim=0)  # [M]
    print(denom)
    numerator = weights.t() @ xyz_f  # [M, 3]

    mem_xyz = numerator / (denom[:, None] + 1e-8)

    if topk > 0:
        # Per-memory-entry top-k weights across all gaussians.
        # weights.topk(dim=0) returns [K, M].
        topk_w_km, topk_i_km = weights.topk(topk, dim=0)
        topk_weights = topk_w_km.t().contiguous()  # [M, K]
        topk_indices = topk_i_km.t().contiguous()  # [M, K]
        flat_idx = topk_indices.reshape(-1)
        topk_xyz = xyz_f[flat_idx].reshape(m, topk, 3).contiguous()  # [M, K, 3]
        topk_valid = torch.isfinite(topk_weights)
        topk_info = {
            "topk_weights": topk_weights,
            "topk_indices": topk_indices,
            "topk_xyz": topk_xyz,
            "topk_valid": topk_valid,
        }
    else:
        topk_info = None

    if return_topk:
        return mem_xyz, topk_info
    return mem_xyz


def _select_models(args, requested: str):
    if requested == "all":
        return [m for m in SUPPORTED_MODELS if getattr(args, f"use_{m}", False)]
    if requested not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model '{requested}', supported: {SUPPORTED_MODELS}")
    if not getattr(args, f"use_{requested}", False):
        raise ValueError(
            f"Model '{requested}' not enabled (args.use_{requested} is False)."
        )
    return [requested]


def main():
    parser = ArgumentParser(description="Compute memory-entry 3D coordinates via weighted average")

    # Reuse the project's argument system to locate checkpoints / dataset / flags.
    AuxiliaryParams(parser)
    ModelParams(parser, sentinel=True)
    OptimizationParams(parser)
    PipelineParams(parser)
    DistributionParams(parser)
    BenchmarkParams(parser)
    DebugParams(parser)

    parser.add_argument("--iteration", default=-1, type=int, help="Which iteration to load (-1 = latest)")
    parser.add_argument(
        "--model",
        default="all",
        choices=["all"] + SUPPORTED_MODELS,
        help="Which memory bank to process (default: all enabled models)",
    )
    parser.add_argument("--chunk_size", default=2048, type=int, help="Gaussian chunk size")
    parser.add_argument(
        "--topk",
        default=5,
        type=int,
        help="For each memory entry, keep top-k gaussian weights/xyz (0 disables)",
    )
    parser.add_argument(
        "--dump_topk",
        default="jsonl",
        choices=["none", "jsonl"],
        help="Dump per-memory-entry topk xyz/weights and final mem_xyz (default: jsonl)",
    )
    parser.add_argument(
        "--temperature",
        default=None,
        type=float,
        help="Softmax temperature (default: args.softmax_temp if present, else 1.0)",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        type=str,
        help="Output directory (default: <load_path>/mem_xyz)",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Path to load the model from",
    )

    args = get_combined_args(parser)

    init_distributed(args)

    log_path = os.path.join(
        args.model_path,
        f"compute_mem_xyz_ws={utils.DEFAULT_GROUP.size()}_rk_{utils.DEFAULT_GROUP.rank()}.log",
    )
    os.makedirs(args.model_path, exist_ok=True)
    log_file = open(log_path, "w")
    set_log_file(log_file)

    init_args(args)
    set_args(args)
    print_all_args(args, log_file)

    safe_state(getattr(args, "quiet", False))

    temperature = 0.05
    out_dir = args.out_dir or os.path.join(args.model_path, "mem_xyz")
    os.makedirs(out_dir, exist_ok=True)

    gaussians = GaussianModel(args.sh_degree, args.emb_degree, getattr(args, "use_embed", True))
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False, _eval=True)

    print('Loading model weights from checkpoint...')

    scene.load_weights(args.model_path)

    xyz = scene.gaussians.get_xyz.detach()
    emb = scene.gaussians.get_embeddings.detach()
    if emb.ndim == 3 and emb.shape[1] == 1:
        emb = emb[:, 0, :]
    if emb.ndim != 2:
        raise RuntimeError(f"Unexpected gaussian embedding shape: {tuple(emb.shape)}")

    models = _select_models(args, args.model)

    rank = utils.DEFAULT_GROUP.rank()
    world = utils.DEFAULT_GROUP.size()
    rank_suffix = "" if int(world) == 1 else f"_rk{int(rank)}"

    for model in models:
        bit_l, bit_r = getattr(args, f"{model}_bit")
        projection = scene.emb_proj_ops[model]
        memory = scene.emb_mem_ops[model]

        if projection is None or memory is None:
            raise RuntimeError(f"Missing projection/memory for model '{model}'")

        index_features = emb[:, bit_l:bit_r].contiguous()

        mem_xyz = compute_mem_xyz_from_gaussians(
            xyz=xyz,
            index_features=index_features,
            projection=projection,
            memory=memory,
            temperature=temperature,
            chunk_size=int(args.chunk_size),
            topk=int(args.topk) if args.dump_topk != "none" else 0,
            return_topk=(args.dump_topk != "none"),
        )

        topk_info = None
        if args.dump_topk != "none":
            mem_xyz, topk_info = mem_xyz

        save_path = os.path.join(out_dir, f"mem_xyz_{model}{rank_suffix}.pt")
        payload = {
            "model": model,
            "temperature": float(temperature),
            "chunk_size": int(args.chunk_size),
            "bit": (int(bit_l), int(bit_r)),
            "mem_xyz": mem_xyz.cpu(),
        }
        if topk_info is not None:
            payload.update(
                {
                    "topk": int(args.topk),
                    "topk_weights": topk_info["topk_weights"].cpu(),
                    "topk_indices": topk_info["topk_indices"].cpu(),
                    "topk_xyz": topk_info["topk_xyz"].cpu(),
                    "topk_valid": topk_info["topk_valid"].cpu(),
                }
            )
        torch.save(payload, save_path)

        if args.dump_topk == "jsonl" and topk_info is not None:
            jsonl_path = os.path.join(out_dir, f"mem_xyz_{model}{rank_suffix}_topk.jsonl")
            mem_xyz_cpu = mem_xyz.detach().cpu()
            topk_w = topk_info["topk_weights"].detach().cpu()
            topk_i = topk_info["topk_indices"].detach().cpu()
            topk_xyz = topk_info["topk_xyz"].detach().cpu()
            topk_valid = topk_info["topk_valid"].detach().cpu()

            with open(jsonl_path, "w") as f:
                m_entries = mem_xyz_cpu.shape[0]
                k_entries = topk_w.shape[1]
                for j in range(m_entries):
                    top_list = []
                    for t in range(k_entries):
                        if not bool(topk_valid[j, t].item()):
                            continue
                        top_list.append(
                            {
                                "gaussian_index": int(topk_i[j, t].item()),
                                "weight": float(topk_w[j, t].item()),
                                "xyz": [
                                    float(topk_xyz[j, t, 0].item()),
                                    float(topk_xyz[j, t, 1].item()),
                                    float(topk_xyz[j, t, 2].item()),
                                ],
                            }
                        )
                    rec = {
                        "mem_index": int(j),
                        "mem_xyz": [
                            float(mem_xyz_cpu[j, 0].item()),
                            float(mem_xyz_cpu[j, 1].item()),
                            float(mem_xyz_cpu[j, 2].item()),
                        ],
                        "topk": top_list,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if utils.DEFAULT_GROUP.rank() == 0:
            print(f"[{model}] saved -> {save_path} (shape={tuple(mem_xyz.shape)})")
            if args.dump_topk == "jsonl" and topk_info is not None:
                print(f"[{model}] topk jsonl -> {jsonl_path}")

    if utils.DEFAULT_GROUP.rank() == 0:
        print(f"Done. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
