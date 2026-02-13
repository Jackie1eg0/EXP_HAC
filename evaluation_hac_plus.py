#!/usr/bin/env python3
"""
evaluation_hac_plus.py
======================
Comprehensive comparison and evaluation of HAC++ vs Scaffold-GS baseline.

Based on: "HAC++: Towards 100X Compression of 3D Gaussian Splatting"

Modules
-------
1. Rate-Distortion & Metrics  (PSNR, SSIM, LPIPS)
2. Storage Decomposition Analysis
3. Pruning & Masking Efficiency
4. Bit Allocation Visualization  (3-D scatter)
5. Adaptive Quantization Analysis

Usage
-----
    python evaluation_hac_plus.py \
        --source_path /path/to/dataset \
        --scaffold_model output/scaffold_gs \
        --hac_models output/hac_l0005 output/hac_l001 output/hac_l002 output/hac_l004 \
        --hac_lambdas 0.0005 0.001 0.002 0.004 \
        --output_dir evaluation_results
"""

# ================================================================
#  Imports
# ================================================================
import os
import json
import numpy as np
import torch
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")                       # headless backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # noqa: F401 – registers 3-D proj
import seaborn as sns

import lpips as lpips_lib

# ---- project imports (run from repo root) ----
from scene import Scene, GaussianModel
from scene.hac_model import BinaryHashGrid, HACContextModel
from gaussian_renderer import prefilter_voxel, render
from utils.image_utils import psnr
from utils.loss_utils import ssim

# ================================================================
#  Global State & Style
# ================================================================
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "font.family":      "serif",
    "axes.titlesize":   14,
    "axes.labelsize":   12,
})

_LPIPS_FN = None

def _get_lpips_fn():
    global _LPIPS_FN
    if _LPIPS_FN is None:
        _LPIPS_FN = lpips_lib.LPIPS(net="vgg").cuda().eval()
    return _LPIPS_FN


# ================================================================
#  Argument Parsing
# ================================================================
def parse_args():
    p = ArgumentParser(description="HAC++ Comprehensive Evaluation")
    p.add_argument("--source_path",     type=str, required=True,
                   help="Path to the dataset (e.g. data/nerf_synthetic/lego)")
    p.add_argument("--scaffold_model",  type=str, required=True,
                   help="Path to the trained Scaffold-GS output directory")
    p.add_argument("--hac_models",      nargs="+", type=str, required=True,
                   help="Paths to HAC++ output directories (one per lambda)")
    p.add_argument("--hac_lambdas",     nargs="+", type=float, required=True,
                   help="Lambda values that match --hac_models 1-to-1")
    p.add_argument("--iteration",       type=int, default=-1,
                   help="Checkpoint iteration to load (-1 = latest)")
    p.add_argument("--output_dir",      type=str, default="evaluation_results",
                   help="Where to save all plots and reports")
    p.add_argument("--white_background", action="store_true")
    p.add_argument("--num_mask_views",  type=int, default=0,
                   help="Views for mask analysis (0 = all test views)")
    p.add_argument("--batch_size",      type=int, default=4096,
                   help="Batch size for batched entropy / quant-step computation")
    return p.parse_args()


# ================================================================
#  Model Loading
# ================================================================
def load_model(model_path: str, source_path: str, iteration: int = -1):
    """
    Load a trained Scaffold-GS or HAC++ model.

    Returns
    -------
    gaussians : GaussianModel   (on GPU, eval mode)
    scene     : Scene
    cfg       : Namespace       (original training config)
    """
    # ---- reconstruct config from saved cfg_args ----
    cfg_file = os.path.join(model_path, "cfg_args")
    with open(cfg_file) as f:
        cfg = eval(f.read())

    # override paths (might differ from training machine)
    cfg.source_path = os.path.abspath(source_path)
    cfg.model_path  = model_path

    # ensure all expected keys exist with safe defaults
    _defaults = dict(
        feat_dim=32, n_offsets=10, voxel_size=0.001,
        update_depth=3, update_init_factor=16, update_hierachy_factor=4,
        use_feat_bank=False, appearance_dim=32, ratio=1,
        add_opacity_dist=False, add_cov_dist=False, add_color_dist=False,
        white_background=False, eval=False, data_device="cuda",
        images="images", resolution=-1, sh_degree=3,
    )
    for k, v in _defaults.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)

    # ---- build & load ----
    gaussians = GaussianModel(
        cfg.feat_dim, cfg.n_offsets, cfg.voxel_size,
        cfg.update_depth, cfg.update_init_factor, cfg.update_hierachy_factor,
        cfg.use_feat_bank, cfg.appearance_dim, cfg.ratio,
        cfg.add_opacity_dist, cfg.add_cov_dist, cfg.add_color_dist,
    )
    scene = Scene(cfg, gaussians, load_iteration=iteration, shuffle=False)

    # auto-detect HAC++ checkpoint
    ckpt_dir  = os.path.join(model_path,
                             f"point_cloud/iteration_{scene.loaded_iter}")
    ckpt_file = os.path.join(ckpt_dir, "hac_checkpoints.pth")
    if os.path.exists(ckpt_file):
        gaussians.load_hac_checkpoints(ckpt_dir)
        print(f"    [HAC++] Loaded compression model  "
              f"(mode={gaussians.hac_mode})")

    gaussians.eval()
    return gaussians, scene, cfg


# helper: default pipeline args for rendering
def _default_pipe():
    return Namespace(debug=False,
                     compute_cov3D_python=False,
                     convert_SHs_python=False)


# ================================================================
#  Module 1 – Test Metrics (PSNR / SSIM / LPIPS)
# ================================================================
@torch.no_grad()
def compute_test_metrics(gaussians, scene, white_bg=False):
    """Render every test view and return mean + per-view metrics."""
    bg   = torch.tensor([1, 1, 1] if white_bg else [0, 0, 0],
                        dtype=torch.float32, device="cuda")
    pipe = _default_pipe()
    lpips_fn = _get_lpips_fn()

    psnrs_l, ssims_l, lpipss_l = [], [], []

    for view in tqdm(scene.getTestCameras(), desc="    Rendering"):
        vis = prefilter_voxel(view, gaussians, pipe, bg)
        img = torch.clamp(
            render(view, gaussians, pipe, bg, visible_mask=vis)["render"],
            0.0, 1.0)
        gt = view.original_image[:3].cuda()

        psnrs_l.append(psnr(img.unsqueeze(0), gt.unsqueeze(0)).item())
        ssims_l.append(ssim(img.unsqueeze(0), gt.unsqueeze(0)).item())
        lpipss_l.append(lpips_fn(img.unsqueeze(0), gt.unsqueeze(0)).item())

    return dict(
        psnr=float(np.mean(psnrs_l)),  ssim=float(np.mean(ssims_l)),
        lpips=float(np.mean(lpipss_l)),
        psnr_std=float(np.std(psnrs_l)),
        ssim_std=float(np.std(ssims_l)),
        lpips_std=float(np.std(lpipss_l)),
        per_view_psnr=psnrs_l, per_view_ssim=ssims_l,
        per_view_lpips=lpipss_l,
    )


# ================================================================
#  Module 2 – Storage Decomposition
# ================================================================
def _count_module_bytes(module, dtype_bytes=4):
    """Sum parameter bytes for an nn.Module (default float32 = 4B)."""
    return sum(p.numel() for p in module.parameters()) * dtype_bytes


def compute_storage_scaffold(gaussians):
    """
    Analytical storage for raw Scaffold-GS (all float32).
    """
    N    = gaussians.get_anchor.shape[0]
    fd   = gaussians._anchor_feat.shape[1]
    sd   = gaussians._scaling.shape[1]
    rd   = gaussians._rotation.shape[1]
    noff = gaussians.n_offsets

    b = dict(
        Features  = N * fd * 4,
        Scaling   = N * sd * 4,
        Offsets   = N * noff * 3 * 4,
        Positions = N * 3 * 4,
        Rotation  = N * rd * 4,
        Opacity   = N * 1 * 4,
    )
    # MLPs
    mlp_bytes = 0
    mlp_bytes += _count_module_bytes(gaussians.mlp_opacity)
    mlp_bytes += _count_module_bytes(gaussians.mlp_cov)
    mlp_bytes += _count_module_bytes(gaussians.mlp_color)
    if gaussians.use_feat_bank:
        mlp_bytes += _count_module_bytes(gaussians.mlp_feature_bank)
    if gaussians.appearance_dim > 0 and gaussians.embedding_appearance is not None:
        mlp_bytes += _count_module_bytes(gaussians.embedding_appearance)
    b["Networks"] = mlp_bytes

    total = sum(b.values())
    return dict(
        total_bytes=total,
        total_mb=total / (1024 ** 2),
        breakdown=b,
        breakdown_mb={k: v / (1024 ** 2) for k, v in b.items()},
        num_anchors=N,
    )


@torch.no_grad()
def compute_storage_hac(gaussians, anchor_mask=None, batch_size=4096):
    """
    Entropy-estimated compressed storage for an HAC++ model.

    Parameters
    ----------
    anchor_mask : np.ndarray[bool] of shape [N], optional
        If given, only count entropy bits for valid anchors.
    """
    N = gaussians.get_anchor.shape[0]

    # decide which anchors contribute to the bitstream
    if anchor_mask is not None:
        valid_idx = torch.from_numpy(np.where(anchor_mask)[0]).cuda()
    else:
        valid_idx = torch.arange(N, device="cuda")
    N_valid = valid_idx.shape[0]

    # ---- 1. Entropy-coded attributes (features / scaling / offsets) ----
    total_bits_f = 0.0
    total_bits_s = 0.0
    total_bits_o = 0.0

    for start in tqdm(range(0, N_valid, batch_size),
                      desc="    Entropy estimation", leave=False):
        end = min(start + batch_size, N_valid)
        idx = valid_idx[start:end]
        bf, bs, bo, _, _, _ = gaussians.compute_entropy_loss(idx)
        total_bits_f += bf.sum().item()
        total_bits_s += bs.sum().item()
        total_bits_o += bo.sum().item()

    bytes_feat    = total_bits_f / 8
    bytes_scaling = total_bits_s / 8
    bytes_offsets = total_bits_o / 8

    # ---- 2. Hash Grid – binary entropy ----
    hash_bits = 0.0
    tables = [lv.hash_table for lv in gaussians.hash_grid.levels_3d]
    for lv in gaussians.hash_grid.levels_2d:
        tables.extend([lv.table_xy, lv.table_xz, lv.table_yz])
    for t in tables:
        p   = torch.sigmoid(t)
        ent = -p * torch.log2(p + 1e-10) - (1 - p) * torch.log2(1 - p + 1e-10)
        hash_bits += ent.sum().item()
    bytes_hash = hash_bits / 8

    # ---- 3. Anchor positions (float16 estimate) ----
    bytes_pos = N_valid * 3 * 2

    # ---- 4. Rotation + Opacity (float16, not entropy-coded) ----
    rd = gaussians._rotation.shape[1]
    bytes_rot_op = N_valid * (rd + 1) * 2

    # ---- 5. Rendering MLPs (shared overhead, float32) ----
    mlp_bytes = 0
    mlp_bytes += _count_module_bytes(gaussians.mlp_opacity)
    mlp_bytes += _count_module_bytes(gaussians.mlp_cov)
    mlp_bytes += _count_module_bytes(gaussians.mlp_color)
    if gaussians.use_feat_bank:
        mlp_bytes += _count_module_bytes(gaussians.mlp_feature_bank)
    if gaussians.appearance_dim > 0 and gaussians.embedding_appearance is not None:
        mlp_bytes += _count_module_bytes(gaussians.embedding_appearance)

    # ---- 6. Context model (decoding overhead, float32) ----
    ctx_bytes = _count_module_bytes(gaussians.context_model)

    breakdown = dict(
        Features        = bytes_feat,
        Scaling         = bytes_scaling,
        Offsets         = bytes_offsets,
        Positions       = bytes_pos,
        HashGrid        = bytes_hash,
        Networks        = mlp_bytes + ctx_bytes + bytes_rot_op,
    )
    total = sum(breakdown.values())
    return dict(
        total_bytes=total,
        total_mb=total / (1024 ** 2),
        breakdown=breakdown,
        breakdown_mb={k: v / (1024 ** 2) for k, v in breakdown.items()},
        num_anchors=N,
        num_valid_anchors=N_valid,
        bits_per_anchor=dict(
            features=total_bits_f / max(N_valid, 1),
            scaling=total_bits_s  / max(N_valid, 1),
            offsets=total_bits_o  / max(N_valid, 1),
        ),
    )


# ================================================================
#  Module 3 – Pruning & Masking Efficiency
# ================================================================
@torch.no_grad()
def compute_mask_stats(gaussians, scene, white_bg=False, num_views=0):
    """
    Aggregate neural-opacity statistics across test views.

    Returns valid-anchor / valid-Gaussian ratios and the raw opacity
    distribution (for histogram plotting).
    """
    bg   = torch.tensor([1, 1, 1] if white_bg else [0, 0, 0],
                        dtype=torch.float32, device="cuda")
    pipe = _default_pipe()

    cams = scene.getTestCameras()
    if 0 < num_views < len(cams):
        cams = cams[:num_views]

    N    = gaussians.get_anchor.shape[0]
    noff = gaussians.n_offsets

    # per-anchor / per-offset maximum opacity observed across views
    max_op = torch.full((N, noff), -999.0, device="cuda")
    all_op = []                                # for histogram

    for view in tqdm(cams, desc="    Mask analysis", leave=False):
        vis_mask = prefilter_voxel(view, gaussians, pipe, bg)
        vis_idx  = torch.where(vis_mask)[0]

        anchor = gaussians.get_anchor[vis_mask]
        feat, grid_scaling, grid_offsets = \
            gaussians.get_quantized_attributes(vis_mask)

        # view direction & distance
        ob_view = anchor - view.camera_center
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        ob_view = ob_view / ob_dist

        # feature-bank (if enabled)
        if gaussians.use_feat_bank:
            cat_vd = torch.cat([ob_view, ob_dist], dim=1)
            bw = gaussians.get_featurebank_mlp(cat_vd).unsqueeze(1)
            fb = feat.unsqueeze(-1)
            feat = (fb[:, ::4, :1].repeat(1, 4, 1) * bw[:, :, :1]
                    + fb[:, ::2, :1].repeat(1, 2, 1) * bw[:, :, 1:2]
                    + fb[:, ::1, :1] * bw[:, :, 2:]).squeeze(-1)

        # neural opacity  (same logic as renderer)
        if gaussians.add_opacity_dist:
            cat_in = torch.cat([feat, ob_view, ob_dist], dim=1)
        else:
            cat_in = torch.cat([feat, ob_view], dim=1)

        neural_op = gaussians.get_opacity_mlp(cat_in)          # [V, noff]

        max_op[vis_idx] = torch.max(max_op[vis_idx], neural_op)
        all_op.append(neural_op.cpu().flatten())

    # ---- aggregate ----
    was_visible = (max_op > -998.0).any(dim=1)
    anchor_valid  = (max_op > 0.0).any(dim=1)          # at least 1 active offset
    offset_valid  = (max_op > 0.0)

    va = anchor_valid[was_visible].float().mean().item() if was_visible.any() else 0.0
    vg = offset_valid[was_visible].float().mean().item() if was_visible.any() else 0.0

    return dict(
        valid_anchor_ratio=va,
        valid_gaussian_ratio=vg,
        anchor_mask=anchor_valid.cpu().numpy(),
        opacity_values=torch.cat(all_op).numpy() if all_op else np.array([]),
        num_anchors_total=N,
        num_anchors_visible=int(was_visible.sum().item()),
        num_anchors_valid=int(anchor_valid.sum().item()),
    )


# ================================================================
#  Module 4 – Bit Allocation  (per-anchor bits for 3-D scatter)
# ================================================================
@torch.no_grad()
def compute_bit_allocation(gaussians, batch_size=4096):
    """Return per-anchor bit consumption and anchor xyz positions."""
    if gaussians.hac_mode < 2 or gaussians.hash_grid is None:
        return None

    N = gaussians.get_anchor.shape[0]
    bits = torch.zeros(N, 3, device="cuda")        # feat / scaling / offset

    for start in tqdm(range(0, N, batch_size),
                      desc="    Bit allocation", leave=False):
        end = min(start + batch_size, N)
        idx = torch.arange(start, end, device="cuda")
        bf, bs, bo, _, _, _ = gaussians.compute_entropy_loss(idx)
        bits[start:end, 0] = bf
        bits[start:end, 1] = bs
        bits[start:end, 2] = bo

    pos   = gaussians.get_anchor.detach().cpu().numpy()
    total = bits.sum(dim=1).cpu().numpy()
    return dict(
        positions=pos,                            # [N, 3]
        total_bits=total,                         # [N]
        bits_feat=bits[:, 0].cpu().numpy(),
        bits_scaling=bits[:, 1].cpu().numpy(),
        bits_offset=bits[:, 2].cpu().numpy(),
    )


# ================================================================
#  Module 5 – Adaptive Quantization Steps
# ================================================================
@torch.no_grad()
def compute_quantization_steps(gaussians, batch_size=8192):
    """Extract the learned quantization step sizes q_i from the AQM."""
    if gaussians.hac_mode < 2 or gaussians.context_model is None:
        return None

    N = gaussians.get_anchor.shape[0]
    q_f_all, q_s_all, q_o_all = [], [], []

    for start in tqdm(range(0, N, batch_size),
                      desc="    Quant steps", leave=False):
        end = min(start + batch_size, N)
        pos = gaussians.get_anchor[start:end]
        hf  = gaussians.hash_grid(pos)
        qf, qs, qo = gaussians.context_model.compute_quantization_step(hf)
        q_f_all.append(qf.cpu())
        q_s_all.append(qs.cpu())
        q_o_all.append(qo.cpu())

    return dict(
        q_feat=torch.cat(q_f_all).numpy(),        # [N, feat_dim]
        q_scaling=torch.cat(q_s_all).numpy(),      # [N, scaling_dim]
        q_offset=torch.cat(q_o_all).numpy(),       # [N, offset_dim]
    )


# ================================================================
#  Plot 1 – Rate-Distortion Curve
# ================================================================
def plot_rate_distortion(results, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scaffold-GS baseline
    sg = results["scaffold"]
    ax.scatter(sg["storage"]["total_mb"], sg["metrics"]["psnr"],
               s=250, marker="*", c="red", zorder=5,
               edgecolors="black", linewidths=1.2,
               label="Scaffold-GS (baseline)")
    ax.annotate("Scaffold-GS",
                (sg["storage"]["total_mb"], sg["metrics"]["psnr"]),
                textcoords="offset points", xytext=(12, 8),
                fontsize=10, fontweight="bold", color="red")

    # HAC++ models (sorted by lambda → typically decreasing size)
    hac_keys = sorted(k for k in results if k.startswith("hac_"))
    sizes, psnrs, labels = [], [], []
    for k in hac_keys:
        r = results[k]
        sizes.append(r["storage"]["total_mb"])
        psnrs.append(r["metrics"]["psnr"])
        labels.append(r["label"])

    ax.plot(sizes, psnrs, "o-", color="#1976D2",
            markersize=10, linewidth=2, label="HAC++", zorder=4)
    for i, lb in enumerate(labels):
        ax.annotate(lb.replace("HAC++ ", ""),
                    (sizes[i], psnrs[i]),
                    textcoords="offset points", xytext=(8, -14),
                    fontsize=8, alpha=0.85)

    ax.set_xlabel("Model Size (MB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Rate-Distortion Curve: Scaffold-GS vs HAC++")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    _save(fig, output_dir, "plot1_rate_distortion")


# ================================================================
#  Plot 2 – Storage Decomposition  (stacked bar)
# ================================================================
def plot_storage_decomposition(results, output_dir):
    # ---- collect per-model component vectors ----
    model_labels = []
    components   = ["Features", "Offsets", "Scaling",
                    "Positions", "HashGrid", "Networks"]
    comp_data    = {c: [] for c in components}

    # Scaffold-GS (HashGrid = 0)
    sg_bd = results["scaffold"]["storage"]["breakdown_mb"]
    model_labels.append("Scaffold-GS")
    comp_data["Features"].append(sg_bd.get("Features", 0))
    comp_data["Offsets"].append(sg_bd.get("Offsets", 0))
    comp_data["Scaling"].append(sg_bd.get("Scaling", 0))
    comp_data["Positions"].append(sg_bd.get("Positions", 0))
    comp_data["HashGrid"].append(0.0)
    # group Rotation + Opacity + Networks into "Networks"
    net = (sg_bd.get("Networks", 0)
           + sg_bd.get("Rotation", 0)
           + sg_bd.get("Opacity", 0))
    comp_data["Networks"].append(net)

    for k in sorted(k for k in results if k.startswith("hac_")):
        r  = results[k]
        bd = r["storage"]["breakdown_mb"]
        model_labels.append(r["label"])
        for c in components:
            comp_data[c].append(bd.get(c, 0))

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(model_labels)), 6))
    x      = np.arange(len(model_labels))
    width  = 0.50
    colors = ["#EF5350", "#42A5F5", "#66BB6A",
              "#FFA726", "#AB47BC", "#78909C"]
    bottom = np.zeros(len(model_labels))

    for ci, comp in enumerate(components):
        vals = np.array(comp_data[comp])
        ax.bar(x, vals, width, bottom=bottom,
               label=comp, color=colors[ci % len(colors)],
               edgecolor="white", linewidth=0.5)
        bottom += vals

    for i, tot in enumerate(bottom):
        ax.text(i, tot + 0.2, f"{tot:.1f} MB",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=20, ha="right")
    ax.set_ylabel("Size (MB)")
    ax.set_title("Storage Decomposition by Component")
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, output_dir, "plot2_storage_decomposition")


# ================================================================
#  Plot 3 – Mask Value Distribution  (histogram)
# ================================================================
def plot_mask_distribution(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Scaffold-GS
    _plot_opacity_hist(axes[0],
                       results["scaffold"]["mask_stats"],
                       "Scaffold-GS", "#EF5350")

    # Right: HAC++ (pick highest lambda for strongest pruning effect)
    hac_keys = sorted(k for k in results if k.startswith("hac_"))
    if hac_keys:
        hk = hac_keys[-1]
        _plot_opacity_hist(axes[1],
                           results[hk]["mask_stats"],
                           results[hk]["label"], "#1976D2")

    fig.suptitle("Mask (Neural Opacity) Distribution: "
                 "Scaffold-GS vs HAC++", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, output_dir, "plot3_mask_distribution")


def _plot_opacity_hist(ax, ms, title, color):
    vals = ms["opacity_values"]
    if len(vals) == 0:
        ax.set_title(f"{title} (no data)")
        return
    ax.hist(vals, bins=120, density=True, alpha=0.85,
            color=color, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", ls="--", lw=1.2, alpha=0.6)
    vr = ms["valid_gaussian_ratio"]
    ar = ms["valid_anchor_ratio"]
    ax.set_title(title)
    ax.set_xlabel("Neural Opacity (mask value)")
    ax.set_ylabel("Density")
    ax.legend([f"Anchor valid: {ar:.1%}\n"
               f"Gauss. valid: {vr:.1%}"],
              fontsize=10, loc="upper right")


# ================================================================
#  Plot 4 – Bit Allocation 3-D Scatter
# ================================================================
def plot_bit_allocation_3d(results, output_dir):
    # pick first HAC++ model with bit data
    ba, label = None, ""
    mask = None
    for k in sorted(k for k in results if k.startswith("hac_")):
        r = results[k]
        if r.get("bit_allocation") is not None:
            ba    = r["bit_allocation"]
            label = r["label"]
            mask  = r["mask_stats"]["anchor_mask"]
            break
    if ba is None:
        print("    [SKIP] No HAC++ model with bit allocation data.")
        return

    pos  = ba["positions"]
    bits = ba["total_bits"]
    if mask is not None:
        pos, bits = pos[mask], bits[mask]

    # sub-sample for performance
    MAX_PTS = 50_000
    if len(pos) > MAX_PTS:
        idx = np.random.choice(len(pos), MAX_PTS, replace=False)
        pos, bits = pos[idx], bits[idx]

    vmin, vmax = np.percentile(bits, 2), np.percentile(bits, 98)

    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection="3d")
    sc  = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                     c=bits, cmap="viridis", s=0.8, alpha=0.6,
                     vmin=vmin, vmax=vmax, rasterized=True)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.12)
    cbar.set_label("Bits per Anchor ($b_i$)", fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"Bit Allocation – {label}", fontsize=13)
    _save(fig, output_dir, "plot4_bit_allocation_3d")


# ================================================================
#  Plot 5 – Adaptive Quantization Step Distribution
# ================================================================
def plot_quantization_steps(results, output_dir):
    qs, label = None, ""
    for k in sorted(k for k in results if k.startswith("hac_")):
        r = results[k]
        if r.get("quant_steps") is not None:
            qs    = r["quant_steps"]
            label = r["label"]
            break
    if qs is None:
        print("    [SKIP] No HAC++ model with quantization step data.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    _spec = [
        ("q_feat",    "Feature $q_i$",  HACContextModel.Q0_FEAT,    "#EF5350"),
        ("q_scaling", "Scaling $q_i$",  HACContextModel.Q0_SCALING, "#42A5F5"),
        ("q_offset",  "Offset $q_i$",   HACContextModel.Q0_OFFSET,  "#66BB6A"),
    ]
    for ax, (key, title, q0, color) in zip(axes, _spec):
        vals = qs[key].flatten()
        ax.hist(vals, bins=100, density=True, alpha=0.85,
                color=color, edgecolor="white", linewidth=0.3)
        ax.axvline(q0, color="black", ls="--", lw=2,
                   label=f"$Q_0$={q0}")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Quantization Step")
        ax.set_ylabel("Density")
        ax.legend(fontsize=10)

    fig.suptitle(f"Adaptive Quantization Step Distributions – {label}",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, output_dir, "plot5_quantization_steps")


# ================================================================
#  Helper – save figure as PNG + PDF
# ================================================================
def _save(fig, output_dir, stem):
    fig.savefig(os.path.join(output_dir, f"{stem}.png"))
    fig.savefig(os.path.join(output_dir, f"{stem}.pdf"))
    plt.close(fig)
    print(f"    Saved: {stem}.png / .pdf")


# ================================================================
#  Summary Report
# ================================================================
def print_summary(results, output_dir):
    lines = []
    sep = "=" * 92
    lines.append(sep)
    lines.append("  HAC++ vs Scaffold-GS  –  Evaluation Summary")
    lines.append(sep)

    # ---- metrics table ----
    hdr = f"\n{'Model':<28s} {'PSNR':>7s} {'SSIM':>8s} {'LPIPS':>8s}" \
          f" {'Size(MB)':>10s} {'Ratio':>8s}"
    lines.append(hdr)
    lines.append("-" * 78)
    sg_mb = results["scaffold"]["storage"]["total_mb"]

    all_keys = ["scaffold"] + sorted(k for k in results if k.startswith("hac_"))
    for key in all_keys:
        r  = results[key]
        m  = r["metrics"]
        mb = r["storage"]["total_mb"]
        ratio = sg_mb / mb if mb > 0 else 0
        rstr  = f"{ratio:.1f}x" if key != "scaffold" else "1.0x (ref)"
        lines.append(
            f"  {r['label']:<26s} {m['psnr']:>7.2f} {m['ssim']:>8.4f}"
            f" {m['lpips']:>8.4f} {mb:>10.2f} {rstr:>8s}")

    # ---- mask stats ----
    lines.append(f"\n{'':=<78s}")
    lines.append(f"  {'Model':<28s} {'Anchor Valid%':>14s}"
                 f" {'Gauss. Valid%':>14s} {'#Anchors':>10s}")
    lines.append("-" * 72)
    for key in all_keys:
        r  = results[key]
        ms = r["mask_stats"]
        lines.append(
            f"  {r['label']:<26s} {ms['valid_anchor_ratio']:>13.1%}"
            f" {ms['valid_gaussian_ratio']:>13.1%}"
            f" {ms['num_anchors_total']:>10d}")

    # ---- storage breakdown ----
    lines.append(f"\n{'':=<78s}")
    lines.append("  Storage Breakdown (MB)")
    lines.append("-" * 72)
    for key in all_keys:
        r  = results[key]
        bd = r["storage"]["breakdown_mb"]
        lines.append(f"\n  {r['label']}:")
        for comp, val in bd.items():
            lines.append(f"      {comp:<22s}: {val:>10.4f} MB")

    lines.append("\n" + sep)
    report = "\n".join(lines)
    print(report)

    # save to file
    with open(os.path.join(output_dir, "summary_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(report)

    # save JSON version (serialisable)
    jdata = {}
    for key in all_keys:
        r = results[key]
        jdata[key] = dict(
            label=r["label"],
            metrics={k: (v if not isinstance(v, list)
                         else [float(x) for x in v])
                     for k, v in r["metrics"].items()},
            storage_mb=r["storage"]["total_mb"],
            storage_breakdown_mb=r["storage"]["breakdown_mb"],
            mask_stats={k: v for k, v in r["mask_stats"].items()
                        if k != "opacity_values" and k != "anchor_mask"},
        )
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(jdata, f, indent=2)

    print(f"\n    Report  → {output_dir}/summary_report.txt")
    print(f"    JSON    → {output_dir}/results.json")


# ================================================================
#  Main
# ================================================================
def main():
    args = parse_args()
    assert len(args.hac_models) == len(args.hac_lambdas), \
        "--hac_models and --hac_lambdas must match in length"
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    # ═══════════════════════════════════════════════════════════
    #  Scaffold-GS baseline
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  Processing: Scaffold-GS (baseline)")
    print(f"{'=' * 60}")

    sg_g, sg_s, sg_c = load_model(
        args.scaffold_model, args.source_path, args.iteration)
    wbg = getattr(sg_c, "white_background", args.white_background)

    print("  [1/3] Test metrics …")
    sg_metrics = compute_test_metrics(sg_g, sg_s, wbg)
    print(f"        PSNR={sg_metrics['psnr']:.2f}  "
          f"SSIM={sg_metrics['ssim']:.4f}  "
          f"LPIPS={sg_metrics['lpips']:.4f}")

    print("  [2/3] Storage …")
    sg_storage = compute_storage_scaffold(sg_g)
    print(f"        {sg_storage['total_mb']:.2f} MB  "
          f"({sg_storage['num_anchors']} anchors)")

    print("  [3/3] Mask analysis …")
    sg_mask = compute_mask_stats(sg_g, sg_s, wbg, args.num_mask_views)
    print(f"        Anchor valid: {sg_mask['valid_anchor_ratio']:.1%}  "
          f"Gaussian valid: {sg_mask['valid_gaussian_ratio']:.1%}")

    results["scaffold"] = dict(
        label="Scaffold-GS", metrics=sg_metrics,
        storage=sg_storage, mask_stats=sg_mask,
        bit_allocation=None, quant_steps=None)

    del sg_g, sg_s
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════
    #  HAC++ models
    # ═══════════════════════════════════════════════════════════
    for mpath, lam in zip(args.hac_models, args.hac_lambdas):
        key   = f"hac_{lam}"
        label = f"HAC++ (λ={lam})"

        print(f"\n{'=' * 60}")
        print(f"  Processing: {label}")
        print(f"{'=' * 60}")

        try:
            g, s, c = load_model(mpath, args.source_path, args.iteration)
        except Exception as e:
            print(f"  [ERROR] Failed to load {mpath}: {e}")
            continue
        wbg    = getattr(c, "white_background", args.white_background)
        is_hac = (g.hac_mode >= 2 and g.hash_grid is not None)

        print("  [1/5] Test metrics …")
        metrics = compute_test_metrics(g, s, wbg)
        print(f"        PSNR={metrics['psnr']:.2f}  "
              f"SSIM={metrics['ssim']:.4f}  "
              f"LPIPS={metrics['lpips']:.4f}")

        print("  [2/5] Mask analysis …")
        mask_stats = compute_mask_stats(g, s, wbg, args.num_mask_views)
        print(f"        Anchor valid: {mask_stats['valid_anchor_ratio']:.1%}  "
              f"Gaussian valid: {mask_stats['valid_gaussian_ratio']:.1%}")

        print("  [3/5] Storage …")
        if is_hac:
            storage = compute_storage_hac(
                g, anchor_mask=mask_stats["anchor_mask"],
                batch_size=args.batch_size)
        else:
            print("        [WARN] No HAC++ ckpt – using raw Scaffold-GS size.")
            storage = compute_storage_scaffold(g)
        print(f"        {storage['total_mb']:.2f} MB  "
              f"({storage['num_anchors']} anchors)")

        bit_alloc, quant_steps = None, None
        if is_hac:
            print("  [4/5] Bit allocation …")
            bit_alloc = compute_bit_allocation(g, args.batch_size)
            print("  [5/5] Quantization steps …")
            quant_steps = compute_quantization_steps(g, args.batch_size)
        else:
            print("  [4/5] Bit allocation … SKIPPED (no HAC++)")
            print("  [5/5] Quantization steps … SKIPPED")

        results[key] = dict(
            label=label, metrics=metrics, storage=storage,
            mask_stats=mask_stats, bit_allocation=bit_alloc,
            quant_steps=quant_steps)

        del g, s
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════
    #  Generate Plots
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  Generating plots")
    print(f"{'=' * 60}")

    print("  Plot 1 – Rate-Distortion …")
    plot_rate_distortion(results, args.output_dir)

    print("  Plot 2 – Storage Decomposition …")
    plot_storage_decomposition(results, args.output_dir)

    print("  Plot 3 – Mask Distribution …")
    plot_mask_distribution(results, args.output_dir)

    print("  Plot 4 – Bit Allocation 3-D …")
    plot_bit_allocation_3d(results, args.output_dir)

    print("  Plot 5 – Quantization Steps …")
    plot_quantization_steps(results, args.output_dir)

    # ═══════════════════════════════════════════════════════════
    #  Summary
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  Summary Report")
    print(f"{'=' * 60}")
    print_summary(results, args.output_dir)

    print(f"\n  All evaluation complete.  Results → {args.output_dir}/\n")


if __name__ == "__main__":
    main()
