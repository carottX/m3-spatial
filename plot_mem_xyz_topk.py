#!/usr/bin/env python3
import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass
class Entry:
    mem_index: int
    mem_xyz: Tuple[float, float, float]
    topk_xyz: List[Tuple[float, float, float]]
    topk_w: List[float]


def _iter_entries(jsonl_path: str) -> Iterable[Entry]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON decode error at line {line_no}: {e}") from e

            mem_index = int(rec["mem_index"])
            mem_xyz_list = rec["mem_xyz"]
            mem_xyz = (float(mem_xyz_list[0]), float(mem_xyz_list[1]), float(mem_xyz_list[2]))

            topk = rec.get("topk", [])
            topk_xyz: List[Tuple[float, float, float]] = []
            topk_w: List[float] = []
            for t in topk:
                xyz = t.get("xyz", None)
                if xyz is None:
                    continue
                w = t.get("weight", None)
                if w is None:
                    continue
                topk_xyz.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
                topk_w.append(float(w))

            yield Entry(mem_index=mem_index, mem_xyz=mem_xyz, topk_xyz=topk_xyz, topk_w=topk_w)


def _weights_to_sizes(weights: Sequence[float], s_min: float, s_max: float) -> List[float]:
    if len(weights) == 0:
        return []
    w_min = min(weights)
    w_max = max(weights)
    denom = (w_max - w_min) if (w_max - w_min) != 0.0 else 1.0
    sizes: List[float] = []
    for w in weights:
        t = (w - w_min) / denom
        sizes.append(float(s_min + t * (s_max - s_min)))
    return sizes


def _weights_to_alphas(weights: Sequence[float], a_min: float, a_max: float) -> List[float]:
    if len(weights) == 0:
        return []
    w_min = min(weights)
    w_max = max(weights)
    denom = (w_max - w_min) if (w_max - w_min) != 0.0 else 1.0
    alphas: List[float] = []
    for w in weights:
        t = (w - w_min) / denom
        alphas.append(float(a_min + t * (a_max - a_min)))
    return alphas


def _reservoir_sample(entries: Iterable[Entry], k: int, rng: random.Random) -> List[Entry]:
    sample: List[Entry] = []
    seen = 0
    for e in entries:
        if not e.topk_xyz:
            continue
        seen += 1
        if len(sample) < k:
            sample.append(e)
            continue
        j = rng.randrange(seen)
        if j < k:
            sample[j] = e
    return sample


def _set_axes_equal(ax):
    # Matplotlib 3D equal aspect workaround
    import numpy as np

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_entries(
    selected: Sequence[Entry],
    out_path: str,
    title: Optional[str] = None,
    connect: bool = True,
    scale_mode: str = "per_entry",
    topk_size_min: float = 20.0,
    topk_size_max: float = 160.0,
    mem_size: float = 220.0,
    line_alpha_min: float = 0.15,
    line_alpha_max: float = 0.6,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("tab10")

    global_w: List[float] = []
    if scale_mode == "global":
        for e in selected:
            global_w.extend(e.topk_w)

    for i, e in enumerate(selected):
        color = cmap(i % 10)

        mx, my, mz = e.mem_xyz
        ax.scatter([mx], [my], [mz], c=[color], s=float(mem_size), marker="X", edgecolors="k", linewidths=0.6)

        tx = [p[0] for p in e.topk_xyz]
        ty = [p[1] for p in e.topk_xyz]
        tz = [p[2] for p in e.topk_xyz]

        if scale_mode == "global":
            sizes = _weights_to_sizes(global_w, float(topk_size_min), float(topk_size_max))
            # Map each weight to a size by ranking within global weights.
            # This is deterministic and avoids per-weight floating matching issues.
            order = sorted(range(len(global_w)), key=lambda idx: global_w[idx])
            rank = {idx: r for r, idx in enumerate(order)}
            # For the current entry, approximate sizes by weight rank position in the global list.
            # If duplicates exist, this still yields consistent relative sizing.
            idxs = sorted(range(len(e.topk_w)), key=lambda j: e.topk_w[j])
            e_sizes = _weights_to_sizes(e.topk_w, float(topk_size_min), float(topk_size_max))
            sizes_use = e_sizes
            alphas_use = _weights_to_alphas(e.topk_w, float(line_alpha_min), float(line_alpha_max))
        else:
            sizes_use = _weights_to_sizes(e.topk_w, float(topk_size_min), float(topk_size_max))
            alphas_use = _weights_to_alphas(e.topk_w, float(line_alpha_min), float(line_alpha_max))

        ax.scatter(tx, ty, tz, c=[color], s=sizes_use, marker="o", alpha=0.9)

        if connect:
            for (x, y, z), a in zip(e.topk_xyz, alphas_use):
                ax.plot([mx, x], [my, y], [mz, z], c=color, alpha=float(a), linewidth=1.0)

        # ax.text(mx, my, mz, f"mem {e.mem_index}", color="black", fontsize=9)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title:
        ax.set_title(title)

    _set_axes_equal(ax)
    ax.grid(True, alpha=0.25)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description="Randomly sample memory entries from mem_xyz_*_topk.jsonl and plot mem_xyz + topk xyz in 3D."
    )
    p.add_argument("--jsonl", required=True, type=str, help="Path to mem_xyz_*_topk.jsonl")
    p.add_argument("--num_entries", default=5, type=int, help="How many memory entries to sample")
    p.add_argument("--seed", default=0, type=int, help="Random seed")
    p.add_argument(
        "--out",
        default=None,
        type=str,
        help="Output PNG path (default: <jsonl_dir>/sample_topk_3d.png)",
    )
    p.add_argument("--no_connect", action="store_true", help="Do not draw lines between mem_xyz and topk points")
    p.add_argument(
        "--scale_mode",
        default="per_entry",
        choices=["per_entry", "global"],
        help="Scale topk point size by weights per entry or globally",
    )
    p.add_argument("--topk_size_min", default=20.0, type=float, help="Min marker size for topk points")
    p.add_argument("--topk_size_max", default=160.0, type=float, help="Max marker size for topk points")
    p.add_argument("--mem_size", default=220.0, type=float, help="Marker size for mem_xyz points")
    p.add_argument("--line_alpha_min", default=0.15, type=float, help="Min alpha for connecting lines")
    p.add_argument("--line_alpha_max", default=0.6, type=float, help="Max alpha for connecting lines")
    args = p.parse_args()

    jsonl_path = args.jsonl
    out_path = args.out
    if out_path is None:
        out_path = os.path.join(os.path.dirname(jsonl_path), "sample_topk_3d.png")

    rng = random.Random(int(args.seed))
    selected = _reservoir_sample(_iter_entries(jsonl_path), int(args.num_entries), rng)
    if len(selected) < int(args.num_entries):
        raise RuntimeError(
            f"Only sampled {len(selected)} entries with non-empty topk; requested {int(args.num_entries)}"
        )

    title = os.path.basename(jsonl_path)
    plot_entries(
        selected,
        out_path=out_path,
        title=title,
        connect=(not args.no_connect),
        scale_mode=str(args.scale_mode),
        topk_size_min=float(args.topk_size_min),
        topk_size_max=float(args.topk_size_max),
        mem_size=float(args.mem_size),
        line_alpha_min=float(args.line_alpha_min),
        line_alpha_max=float(args.line_alpha_max),
    )
    # print(max(global_W))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
