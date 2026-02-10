#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from tqdm import tqdm
from os import makedirs

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from scene import Scene, SceneDataset
from gaussian_renderer import (
    # preprocess3dgs_and_all2all,
    # render
    distributed_preprocess3dgs_and_all2all_final,
    render_final,
)
import torchvision
from utils.general_utils import (
    safe_state,
    set_args,
    init_distributed,
    set_log_file,
    set_cur_iter,
)
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from gaussian_renderer.loss_distribution import load_camera_from_cpu_to_all_gpu_for_eval
from gaussian_renderer.workload_division import (
    start_strategy_final,
    DivisionStrategyHistoryFinal,
)
from arguments import (
    AuxiliaryParams,
    ModelParams,
    PipelineParams,
    OptimizationParams,
    DistributionParams,
    BenchmarkParams,
    DebugParams,
    print_all_args,
    init_args,
)
import utils.general_utils as utils
from utils.loss_utils import l2_loss, cosine_loss
from xy_utils.memory import index_to_raw
from xy_utils.visual import vpca_embeddings
from utils.graphics_utils import geom_transform_points

import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import io

from lmm.llamav.fea2mem import load_llamav_embeddings
from lmm.llama.fea2mem import load_llama3_embeddings
from lmm.clip.fea2mem import load_clip_embeddings
from lmm.dinov2.fea2mem import load_dinov2_embeddings
from lmm.seem.fea2mem import load_seem_embeddings
from lmm.siglip.fea2mem import load_siglip_embeddings


def create_2x2_grid(ret_images):
    """
    Create a 2x2 grid from four images in (h,w,c) format.
    
    Args:
        ret_images (list): List of 4 numpy arrays of shape (h,w,c)
        
    Returns:
        numpy.ndarray: Combined image in 2x2 grid format with shape (2h,2w,c)
    """
    if len(ret_images) != 4:
        raise ValueError("Exactly 4 images are required")
    
    # Get dimensions
    h, w, c = ret_images[0].shape
    
    # Create blank canvas
    grid = np.zeros((h*2, w*2, c), dtype=np.uint8)
    
    # Place images in 2x2 grid
    grid[0:h, 0:w, :] = ret_images[0]  # Top-left
    grid[0:h, w:w*2, :] = ret_images[1]  # Top-right
    grid[h:h*2, 0:w, :] = ret_images[2]  # Bottom-left
    grid[h:h*2, w:w*2, :] = ret_images[3]  # Bottom-right
    
    return grid

def sample_points(similarity, points):
    """
    Sample points from a similarity tensor at specified relative coordinates.
    
    Args:
        similarity: torch tensor of shape [H, W, C]
        points: list of [h, w] coordinates in relative [0-1] space
        
    Returns:
        samples: torch tensor of shape [N, C] where N is number of points
    """
    height, width, channels = similarity.shape
    
    # Convert relative coordinates to absolute indices
    absolute_indices = []
    for h, w in points:
        # Convert relative [0-1] coordinates to absolute indices
        h_idx = int(h * (height - 1))
        w_idx = int(w * (width - 1))
        absolute_indices.append((h_idx, w_idx))
    
    # Sample points from similarity tensor
    samples = []
    for h_idx, w_idx in absolute_indices:
        sample = similarity[h_idx, w_idx, :]
        samples.append(sample)
    
    return torch.stack(samples, dim=0)

def visualize_embeddings_3d(embeddings_list,
                           color_schemes=['light_gray', 'light_purple', 'red', 'green', 'blue', 'yellow'],
                           point_size=1,
                           alpha=0.1,
                           figsize=(8, 8),
                           dpi=300):
    """
    Visualize multiple sets of high-dimensional embeddings in 3D space using UMAP.
    First 2 embedding sets use colormaps, subsequent sets use solid colors with larger points.
    
    Args:
        embeddings_list: list of numpy arrays or torch tensors, each of shape (n_samples, n_features)
        color_schemes: list of color schemes for each embedding set
        point_size: float, size of scatter points for first 2 sets
        alpha: float, transparency of points
        figsize: tuple, figure size
        dpi: int, dots per inch for the output image
    
    Returns:
        tuple: (image_array, fig, ax)
        - image_array: numpy array containing the RGB image
        - fig: matplotlib figure object
        - ax: matplotlib axes object
    """
    # Define color schemes for gradient colormaps
    color_mappings = {
        'light_gray': [
            "#404040",  # Darker gray
            "#808080",  # Medium gray
            "#D3D3D3"   # Light gray
        ],
        'light_purple': [
            "#4B0082",  # Indigo
            "#9370DB",  # Medium purple
            "#E6E6FA"   # Lavender
        ],
    }
    
    # Define solid colors for later sets
    solid_colors = {
        'red': '#FF0000',
        'green': '#00FF00',
        'blue': '#0000FF',
        'yellow': '#FFD700'
    }
    
    # Ensure we have enough color schemes for all embedding sets
    if len(color_schemes) < len(embeddings_list):
        raise ValueError(f"Not enough color schemes provided. Expected {len(embeddings_list)}, got {len(color_schemes)}")
    
    # Convert embeddings to numpy if they're torch tensors
    embeddings_list = [
        embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        for embeddings in embeddings_list
    ]
    
    # Concatenate all embeddings for joint UMAP
    all_embeddings = np.vstack(embeddings_list)
    
    # Create array to track which embedding set each point belongs to
    set_indices = np.concatenate([
        np.full(len(embeddings), i) for i, embeddings in enumerate(embeddings_list)
    ])
    
    print("Reducing dimensionality with UMAP...")
    # Perform UMAP dimensionality reduction on all embeddings together
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    embeddings_3d = reducer.fit_transform(all_embeddings)
    
    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each embedding set
    for i, scheme in enumerate(color_schemes[:len(embeddings_list)]):
        # Get points belonging to this embedding set
        mask = (set_indices == i)
        set_points = embeddings_3d[mask]
        
        if i < 2:  # First two sets use colormaps
            # Calculate distances from origin for this embedding set
            distances = np.linalg.norm(set_points, axis=1)
            colors = (distances - distances.min()) / (distances.max() - distances.min())
            
            # Create custom colormap for this embedding set
            colormap = LinearSegmentedColormap.from_list(
                f"custom_{scheme}",
                color_mappings[scheme]
            )
            
            # Create scatter plot with colormap
            scatter = ax.scatter(
                set_points[:, 0],
                set_points[:, 1],
                set_points[:, 2],
                c=colors,
                cmap=colormap,
                s=point_size,
                alpha=alpha,
            )
        else:  # Later sets use solid colors and larger points
            # Use solid color and larger point size
            scatter = ax.scatter(
                set_points[:, 0],
                set_points[:, 1],
                set_points[:, 2],
                c=solid_colors[scheme],
                s=point_size * 20,  # Larger points
                alpha=1.,   # More opaque
            )
    
    # Add grid and set background color
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert figure to numpy array
    fig.canvas.draw()
    
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    image_array = buffer.reshape((h, w, 4))[:, :, 1:]  # Convert ARGB to RGB
    
    return image_array, fig, ax

def draw_colored_circles(image, relative_positions, index=None, radius=20):
    """
    Draw circles with transparency using RGBA format.
    
    Args:
        image: numpy array of shape (H, W, 3) or (H, W, 4) for RGBA input.
        relative_positions: list of [x, y] pairs with values between 0 and 1.
        radius: radius of circles in pixels.
    
    Returns:
        image: numpy array with drawn circles in RGBA format.
    """
    # Ensure the image has 4 channels (RGBA)
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    height, width = image.shape[:2]
    
    # Convert relative positions to absolute pixel coordinates
    positions = [(int(width * y), int(height * x)) for x, y in relative_positions]
    
    # Define colors: (B, G, R, A) format
    colors = [
        (0, 0, 255, 128),    # Red with 50% transparency
        (0, 255, 0, 128),    # Green with 50% transparency
        (255, 0, 0, 128),    # Blue with 50% transparency
        (0, 255, 255, 128)   # Yellow with 50% transparency
    ]
    
    if index is not None:
        colors = colors[index:index+1]
    
    # Create an overlay image with transparent circles
    overlay = np.zeros_like(image, dtype=np.uint8)
    
    for (x, y), (b, g, r, a) in zip(positions, colors):
        # Draw filled circle on the overlay
        cv2.circle(overlay, (x, y), radius, (b, g, r, a), -1)
        
        # Draw black boundary on the original image
        cv2.circle(overlay, (x, y), radius, (1, 1, 1, 255), 2)
    
    mask = (overlay[:,:,:3].sum(axis=-1) > 0) * 1.0
    # Combine the original image with the overlay using alpha blending
    alpha_channel = overlay[:, :, 3] / 255.0
    for c in range(3):  # Blend only the RGB channels
        image[:, :, c] = ((overlay[:, :, c] * alpha_channel + image[:, :, c] * (1 - alpha_channel)) * mask + image[:, :, c] * (1 - mask)).astype(np.uint8)
    return image

def overlay_features(gt_image, raw_feature_pca, alpha=0.6):
    """
    Overlay pre-processed PCA features on ground truth image with transparency
    
    Args:
        gt_image (torch.Tensor): Ground truth image [C, H, W]
        raw_feature_pca (np.array): Pre-processed PCA features [H, W, C]
        alpha (float): Transparency value (0-1)
        
    Returns:
        np.array: Overlaid image [H, W, C]
    """
    h,w,c = gt_image.shape
    raw_feature_pca = cv2.resize(raw_feature_pca, (w, h))
    # Blend images
    overlaid = alpha * gt_image + (1 - alpha) * raw_feature_pca
    # Ensure output is in valid range [0,1]
    overlaid = np.clip(overlaid, 0, 1)
    return overlaid


def _resolve_mem_xyz_path(mem_xyz_dir: str, model: str) -> str:
    if mem_xyz_dir is None:
        raise FileNotFoundError("mem_xyz_dir is None")
    if os.path.isfile(mem_xyz_dir):
        return mem_xyz_dir
    candidates = [
        os.path.join(mem_xyz_dir, f"mem_xyz_{model}.pt"),
        os.path.join(mem_xyz_dir, f"mem_xyz_{model}.pth"),
        os.path.join(mem_xyz_dir, f"mem_xyz{model}.pt"),
        os.path.join(mem_xyz_dir, f"mem_xyz{model}.pth"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Cannot find mem_xyz for model '{model}' in '{mem_xyz_dir}'. Tried e.g. '{candidates[0]}'."
    )


def _load_mem_xyz(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    mem_xyz = obj["mem_xyz"] if isinstance(obj, dict) and "mem_xyz" in obj else obj
    if not isinstance(mem_xyz, torch.Tensor):
        raise TypeError(f"mem_xyz must be a torch.Tensor, got {type(mem_xyz)}")
    if mem_xyz.ndim != 2 or mem_xyz.shape[1] != 3:
        raise ValueError(f"Expected mem_xyz [M,3], got {tuple(mem_xyz.shape)}")
    return mem_xyz.float()


def _build_camera_map(scene: Scene):
    cams = []
    try:
        tc = scene.getTrainCameras()
        if tc is not None:
            cams.extend(list(tc))
        print(tc)
    except Exception:
        pass
    try:
        vc = scene.getTestCameras()
        if vc is not None:
            cams.extend(list(vc))
        print(vc)
    except Exception:
        pass
    cam_map = {}
    for cam in cams:
        name = getattr(cam, "image_name", None)
        if name is None:
            continue
        # Index by full name and basename to be robust to paths.
        if name not in cam_map:
            cam_map[name] = cam
        base = os.path.basename(name)
        if base not in cam_map:
            cam_map[base] = cam
    return cam_map


def _project_world_to_pixel(cam, xyz_world: np.ndarray, image_wh):
    xyz_world = np.asarray(xyz_world, dtype=np.float32).reshape(3)
    w_img, h_img = int(image_wh[0]), int(image_wh[1])
    if w_img <= 1 or h_img <= 1:
        return None
    if not hasattr(cam, "full_proj_transform"):
        return None

    device = cam.full_proj_transform.device
    p = torch.tensor(xyz_world[None, :], device=device, dtype=torch.float32)
    ndc = geom_transform_points(p, cam.full_proj_transform)[0]
    ndc_x = float(ndc[0].item())
    ndc_y = float(ndc[1].item())
    ndc_z = float(ndc[2].item())
    if not np.isfinite(ndc_x) or not np.isfinite(ndc_y) or not np.isfinite(ndc_z):
        return None

    x = int(round((ndc_x + 1.0) * 0.5 * (w_img - 1)))
    y = int(round((1.0 - (ndc_y + 1.0) * 0.5) * (h_img - 1)))
    if x < 0 or x >= w_img or y < 0 or y >= h_img:
        return None
    return (x, y)


def _draw_abs_marker(image: np.ndarray, xy, radius: int = 18):
    if xy is None:
        return image
    x, y = int(xy[0]), int(xy[1])
    out = image.copy()
    if out.shape[-1] == 4:
        color = (255, 0, 0, 255)  # Blue in BGRA
        white = (255, 255, 255, 255)
    else:
        color = (255, 0, 0)  # Blue in BGR
        white = (255, 255, 255)
    cv2.circle(out, (x, y), radius, color, -1)
    cv2.circle(out, (x, y), radius, white, 2)
    return out

def visualize_gma(
    source_path,
    gt_image,
    raw_feature,
    ren_feature,
    similarity,
    all_embeddings,
    memory,
    mem2fea,
    mem_xyz=None,
    cam_map=None,
):
    # [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]
    _,h,w = gt_image.shape
    gt_image = gt_image.permute(1,2,0).cpu().numpy()[:,:,::-1]

    # Principal Query
    raw_feature_pca = vpca_embeddings(raw_feature)
    raw_feature_pca = raw_feature_pca / 255.0
    overlay_gt_pq = overlay_features(gt_image, raw_feature_pca)
    overlay_gt_pq = (overlay_gt_pq * 255).astype(np.uint8)
    overlay_gt_pq = draw_colored_circles(overlay_gt_pq, [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)])
    # cv2.imwrite("overlay_gt_pq.png", overlay_gt_pq)

    # Principal Scene Component
    points = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    sim2mem = sample_points(similarity, points)
    indices = sim2mem.topk(5, dim=-1).indices.cpu()
    memory = memory.cpu()

    psc_visual,_,_ = visualize_embeddings_3d([all_embeddings[:50000], memory, memory[indices[0]], memory[indices[1]], memory[indices[2]], memory[indices[3]]])
    # cv2.imwrite("psc_visual.png", psc_visual[:,:,::-1])
    
    # Raw Feature
    images_root = os.path.join(source_path, "images")
    ret_images = []
    for idx, index in enumerate(indices[:,0]):
        info = mem2fea[index.item()][0]
        ret_image = os.path.join(images_root, info['image_name'])
        ret_image = cv2.imread(ret_image)
        if ret_image is None:
            continue
        ret_image = draw_colored_circles(ret_image, [(info['point_height'], info['point_width'])], index=idx)

        # Also draw the projected mem_xyz point for this memory entry, if available.
        if mem_xyz is not None and cam_map is not None:
            try:
                xyz_world = mem_xyz[index.item()].detach().cpu().numpy()
                img_name = info.get('image_name', None)
                if img_name is None:
                    raise KeyError("mem2fea occurrence missing image_name")

                # Try exact match, then basename.
                cam = cam_map.get(img_name, None)
                # print(img_name)
                # print(os.path.basename(img_name))
                if cam is None:
                    cam = cam_map.get(os.path.splitext(img_name)[0], None)
                if cam is not None:
                    print(f"Projecting mem_xyz for index {index.item()} onto image '{img_name}'")
                    h_img, w_img = ret_image.shape[:2]
                    xy = _project_world_to_pixel(cam, xyz_world, (w_img, h_img))
                    print('Pixel is at:', xy)
                    ret_image = _draw_abs_marker(ret_image, xy, radius=20)
            except Exception as e:
                print(f"{e}")
                pass
        ret_images += [ret_image]
    point_image = create_2x2_grid(ret_images)
    # cv2.imwrite("point_image.png", point_image)
    
    # Render Feature
    ren_feature_pca = vpca_embeddings(ren_feature.permute(2,0,1))
    # cv2.imwrite("ren_feature.png", ren_feature_pca)
    
    return overlay_gt_pq, psc_visual[:,:,::-1], point_image, ren_feature_pca
    
def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, scene, background, cam_map=None, mem_xyz_map=None):
    pq_path = os.path.join(model_path, name, "ours_{}".format(iteration), "principal_query")
    psc_path = os.path.join(model_path, name, "ours_{}".format(iteration), "principal_scene_component")
    raw_path = os.path.join(model_path, name, "ours_{}".format(iteration), "raw_feature")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_feature")
    
    makedirs(pq_path, exist_ok=True)
    makedirs(psc_path, exist_ok=True)
    makedirs(raw_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)

    dataset = SceneDataset(views)

    set_cur_iter(iteration)
    generated_cnt = 0

    num_cameras = len(views)
    strategy_history = DivisionStrategyHistoryFinal(
        dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank()
    )
    progress_bar = tqdm(
        range(1, num_cameras + 1),
        desc="Rendering progress",
        disable=(utils.LOCAL_RANK != 0),
    )

    mem2fea_path = os.path.join(model_path, name, "mem2fea")

    # process embeddings
    # model_names = ["clip", "siglip", "dinov2", "seem", "llama3", "llamav"]
    model_names = ["siglip", "dinov2", "llamav", "clip"]
    model2mem2fea = {
        "clip": "mem80.index",
        "siglip": "mem85.index",
        "dinov2": "mem75.index",
        "seem": "mem100.index",
        "llama3": "mem100.index",
        "llamav": "mem60.index",
    }
    model2load = {
        "clip": load_clip_embeddings,
        "siglip": load_siglip_embeddings,
        "dinov2": load_dinov2_embeddings,
        "seem": load_seem_embeddings,
        "llama3": load_llama3_embeddings,
        "llamav": load_llamav_embeddings,
    }
    mem_load = {}
    for idx in range(1, num_cameras + 1, args.bsz):
        print(idx)
        progress_bar.update(args.bsz)

        num_camera_to_load = min(args.bsz, num_cameras - idx + 1)
        batched_cameras = dataset.get_batched_cameras(num_camera_to_load)
        batched_strategies, gpuid2tasks = start_strategy_final(
            batched_cameras, strategy_history
        )
        load_camera_from_cpu_to_all_gpu_for_eval(
            batched_cameras, batched_strategies, gpuid2tasks
        )

        batched_screenspace_pkg = distributed_preprocess3dgs_and_all2all_final(
            batched_cameras,
            gaussians,
            pipeline,
            background,
            batched_strategies=batched_strategies,
            mode="test",
        )
        batched_image, batched_embeddings, _ = render_final(batched_screenspace_pkg, batched_strategies, use_embed=args.use_embed)

        for camera_id, (image, gt_camera, embedding) in enumerate(
            zip(batched_image, batched_cameras, batched_embeddings)
        ):
            actual_idx = idx + camera_id
            if args.sample_freq != -1 and actual_idx % args.sample_freq != 0:
                continue
            if generated_cnt == args.generate_num:
                break
            # Uncomment for not regenerating images.
            # if os.path.exists(
            #     os.path.join(render_path, "{0:05d}".format(actual_idx) + ".png")
            # ):
            #     continue
            if args.l != -1 and args.r != -1:
                if actual_idx < args.l or actual_idx >= args.r:
                    continue

            generated_cnt += 1

            if (
                image is None or len(image.shape) == 0
            ):  # The image is not rendered locally.
                image = torch.zeros(
                    gt_camera.original_image.shape, device="cuda", dtype=torch.float32
                )

            # process image
            image = torch.clamp(image, 0.0, 1.0)
            gt_image = torch.clamp(gt_camera.original_image / 255.0, 0.0, 1.0)
            for model in model_names:
                if not getattr(args, "use_{}".format(model)):
                    continue
                emb_proj = scene.emb_proj_ops[model]
                emb_mem = scene.emb_mem_ops[model]
                gt_feature = gt_camera.original_embeddings_backup[model].to('cuda') # [h,w,c]
                pred = embedding[getattr(args, f'{model}_bit')[0]:getattr(args, f'{model}_bit')[1],:,:]
                
                c,h,w = pred.shape
                embedding_resized = F.interpolate(
                    pred[None,], (h//3, w//3), mode='bilinear', align_corners=True
                )[0]
                raw_feature, similarity = index_to_raw(embedding_resized, emb_proj, emb_mem, _eval=True, _temp=args.softmax_temp, _return_similarity=True)
                
                mem2fea_path = os.path.join(source_path, model, model2mem2fea[model])
                mem2fea = torch.load(mem2fea_path)
                
                if model not in mem_load:
                    all_embeddings = model2load[model](source_path, 'cuda')
                    mem_load[model] = all_embeddings
                else:
                    all_embeddings = mem_load[model]

                mem_xyz = None
                if mem_xyz_map is not None:
                    mem_xyz = mem_xyz_map.get(model, None)
                pq_visual, psc_visual, point_visual, ren_visual = visualize_gma(
                    source_path,
                    gt_image,
                    embedding_resized,
                    raw_feature,
                    similarity,
                    all_embeddings,
                    emb_mem,
                    mem2fea,
                    mem_xyz=mem_xyz,
                    cam_map=cam_map,
                )
                
                pq_image_path = os.path.join(pq_path, model, "{0:05d}".format(actual_idx) + ".png")
                psc_image_path = os.path.join(psc_path, model, "{0:05d}".format(actual_idx) + ".png")
                raw_image_path = os.path.join(raw_path, model, "{0:05d}".format(actual_idx) + ".png")
                ren_image_path = os.path.join(render_path, model, "{0:05d}".format(actual_idx) + ".png")
                os.makedirs(os.path.dirname(pq_image_path), exist_ok=True)
                os.makedirs(os.path.dirname(psc_image_path), exist_ok=True)
                os.makedirs(os.path.dirname(raw_image_path), exist_ok=True)
                os.makedirs(os.path.dirname(ren_image_path), exist_ok=True)
                
                cv2.imwrite(pq_image_path, pq_visual)
                cv2.imwrite(psc_image_path, psc_visual)
                cv2.imwrite(raw_image_path, point_visual)
                cv2.imwrite(ren_image_path, ren_visual)
                torch.cuda.empty_cache()

            # release memory usage
            gt_camera.original_image = None
            gt_camera.original_embeddings_backup = None

        if generated_cnt == args.generate_num:
            break


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
):
    with torch.no_grad():
        args = utils.get_args()
        gaussians = GaussianModel(dataset.sh_degree, dataset.emb_degree, args.use_embed)
        scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False, _eval=True)
        scene.load_weights(args.load_path)

        cam_map = _build_camera_map(scene)
        print(cam_map.keys())

        # Optional mem_xyz loading (used to project 3D coords onto images)
        mem_xyz_map = {}
        mem_xyz_dir = getattr(args, "mem_xyz_dir", None)
        if mem_xyz_dir is None:
            mem_xyz_dir = os.path.join(args.load_path, "mem_xyz")
        for m in ["clip", "siglip", "dinov2", "seem", "llama3", "llamav"]:
            if not getattr(args, f"use_{m}", False):
                continue
            try:
                p = _resolve_mem_xyz_path(mem_xyz_dir, m)
                mem_xyz_map[m] = _load_mem_xyz(p)
            except Exception:
                pass

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_train:
            render_set(
                args.load_path,
                args.source_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                scene,
                background,
                cam_map=cam_map,
                mem_xyz_map=mem_xyz_map,
            )

        if not skip_test:
            render_set(
                args.load_path,
                args.source_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                scene,
                background,
                cam_map=cam_map,
                mem_xyz_map=mem_xyz_map,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dist_p = DistributionParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--generate_num", default=-1, type=int)
    parser.add_argument("--sample_freq", default=-1, type=int)
    parser.add_argument("--distributed_load", action="store_true")  # TODO: delete this.
    parser.add_argument("--l", default=-1, type=int)
    parser.add_argument("--r", default=-1, type=int)
    parser.add_argument(
        "--mem_xyz_dir",
        default=None,
        type=str,
        help="Directory containing mem_xyz_<model>.pt (default: <load_path>/mem_xyz)",
    )
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    init_distributed(args)
    # This script only supports single-gpu rendering.
    # I need to put the flags here because the render() function need it.
    # However, disable them during render.py because they are only needed during training.

    log_file = open(
        args.model_path
        + f"/render_ws={utils.DEFAULT_GROUP.size()}_rk_{utils.DEFAULT_GROUP.rank()}.log",
        "w",
    )
    set_log_file(log_file)
    
    ## Prepare arguments.
    # Check arguments
    init_args(args)
    # NOTE: Do NOT zero out num_train_cameras/num_test_cameras when skipping rendering.
    # We still want all camera poses available to build `cam_map` so mem_xyz can be
    # projected onto any occurrence image referenced by mem2fea (which may come from
    # train or test views).
    # Set up global args
    set_args(args)

    print_all_args(args, log_file)

    if utils.WORLD_SIZE > 1:
        torch.distributed.barrier(group=utils.DEFAULT_GROUP)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    render_sets(
        lp.extract(args),
        args.iteration,
        pp.extract(args),
        args.skip_train,
        args.skip_test,
    )