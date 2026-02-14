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
# Restructured to match HAC-plus renderer:
#   - Direct parameter access (no hash grid decode)
#   - 3-phase quantization schedule  
#   - Binary grid masks for per-offset pruning
#   - Entropy estimation during training (step > 10000)
#

import time

import torch
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.encodings import STE_binary, STE_multistep


def generate_neural_gaussians(viewpoint_camera, pc: GaussianModel, visible_mask=None, is_training=False, step=0):
    """Generate neural Gaussians from anchors, following HAC-plus architecture."""

    time_sub = 0

    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=pc.get_anchor.device)

    # Direct parameter access (NOT hash-grid-decoded)
    anchor = pc.get_anchor[visible_mask]
    feat = pc._anchor_feat[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    binary_grid_masks = pc.get_mask[visible_mask]  # [N_vis, n_offsets, 1]

    # ============================================================
    #  Quantization Schedule (aligned with HAC-plus)
    # ============================================================
    bit_per_param = None
    bit_per_feat_param = None
    bit_per_scaling_param = None
    bit_per_offsets_param = None
    Q_feat = 1
    Q_scaling = 0.001
    Q_offsets = 0.2

    if is_training:
        # Phase 1: [0, 3000] — no quantization, pure distortion training
        # Phase 2: (3000, 10000] — uniform noise injection
        if step > 3000 and step <= 10000:
            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets

        # Update anchor bounds at transition
        if step == 10000:
            pc.update_anchor_bound()

        # Phase 3: (10000, end) — adaptive quantization + entropy estimation
        if step > 10000:
            # For rendering: adaptive Q noise on visible anchors
            feat_context_orig = pc.calc_interp_feat(anchor)
            feat_context = pc.get_grid_mlp(feat_context_orig)
            mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(feat_context, split_size_or_sections=[
                    pc.feat_dim, pc.feat_dim, pc.feat_dim,
                    6, 6,
                    3 * pc.n_offsets, 3 * pc.n_offsets,
                    1, 1, 1
                ], dim=-1)

            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))
            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets.unsqueeze(1)

            # For entropy estimation: sample 5% of ALL anchors
            choose_idx = torch.rand_like(pc.get_anchor[:, 0]) <= 0.05
            anchor_chosen = pc.get_anchor[choose_idx]
            feat_chosen = pc._anchor_feat[choose_idx]
            grid_offsets_chosen = pc._offset[choose_idx]
            grid_scaling_chosen = pc.get_scaling[choose_idx]
            binary_grid_masks_chosen = pc.get_mask[choose_idx]
            mask_anchor_chosen = pc.get_mask_anchor[choose_idx]

            feat_context_orig = pc.calc_interp_feat(anchor_chosen)
            feat_context = pc.get_grid_mlp(feat_context_orig)
            mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(feat_context, split_size_or_sections=[
                    pc.feat_dim, pc.feat_dim, pc.feat_dim,
                    6, 6,
                    3 * pc.n_offsets, 3 * pc.n_offsets,
                    1, 1, 1
                ], dim=-1)

            Q_feat_e = 1
            Q_scaling_e = 0.001
            Q_offsets_e = 0.2
            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1])
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1])
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1])
            Q_feat_e = Q_feat_e * (1 + torch.tanh(Q_feat_adj))
            Q_scaling_e = Q_scaling_e * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets_e = Q_offsets_e * (1 + torch.tanh(Q_offsets_adj)).view(-1, pc.n_offsets, 3)

            feat_chosen = feat_chosen + torch.empty_like(feat_chosen).uniform_(-0.5, 0.5) * Q_feat_e
            mean_adj, scale_adj, prob_adj = pc.get_deform_mlp.forward(feat_chosen, torch.cat([mean, scale, prob], dim=-1))
            probs = torch.stack([prob, prob_adj], dim=-1)
            probs = torch.softmax(probs, dim=-1)

            grid_scaling_chosen = grid_scaling_chosen + torch.empty_like(grid_scaling_chosen).uniform_(-0.5, 0.5) * Q_scaling_e
            grid_offsets_chosen = grid_offsets_chosen + torch.empty_like(grid_offsets_chosen).uniform_(-0.5, 0.5) * Q_offsets_e
            grid_offsets_chosen = grid_offsets_chosen.view(-1, 3 * pc.n_offsets)
            binary_grid_masks_chosen = binary_grid_masks_chosen.repeat(1, 1, 3).view(-1, 3 * pc.n_offsets)

            bit_feat = pc.EG_mix_prob_2.forward(feat_chosen,
                                                mean, mean_adj,
                                                scale, scale_adj,
                                                probs[..., 0], probs[..., 1],
                                                Q=Q_feat_e, x_mean=pc._anchor_feat.mean())
            bit_feat = bit_feat * mask_anchor_chosen
            bit_scaling = pc.entropy_gaussian.forward(grid_scaling_chosen, mean_scaling, scale_scaling, Q_scaling_e, pc.get_scaling.mean())
            bit_scaling = bit_scaling * mask_anchor_chosen
            bit_offsets = pc.entropy_gaussian.forward(grid_offsets_chosen, mean_offsets, scale_offsets,
                                                     Q_offsets_e.view(-1, 3 * pc.n_offsets), pc._offset.mean())
            bit_offsets = bit_offsets * mask_anchor_chosen * binary_grid_masks_chosen

            bit_per_feat_param = torch.sum(bit_feat) / bit_feat.numel()
            bit_per_scaling_param = torch.sum(bit_scaling) / bit_scaling.numel()
            bit_per_offsets_param = torch.sum(bit_offsets) / bit_offsets.numel()
            bit_per_param = (torch.sum(bit_feat) + torch.sum(bit_scaling) + torch.sum(bit_offsets)) / \
                            (bit_feat.numel() + bit_scaling.numel() + bit_offsets.numel())

    elif not pc.decoded_version:
        # Inference: apply STE quantization
        torch.cuda.synchronize(); t1 = time.time()
        feat_context = pc.calc_interp_feat(anchor)
        mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            torch.split(pc.get_grid_mlp(feat_context), split_size_or_sections=[
                pc.feat_dim, pc.feat_dim, pc.feat_dim,
                6, 6,
                3 * pc.n_offsets, 3 * pc.n_offsets,
                1, 1, 1
            ], dim=-1)

        Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1])
        Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1])
        Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1])
        Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
        Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
        Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj)).view(-1, pc.n_offsets, 3)
        feat = (STE_multistep.apply(feat, Q_feat, pc._anchor_feat.mean())).detach()
        grid_scaling = (STE_multistep.apply(grid_scaling, Q_scaling, pc.get_scaling.mean())).detach()
        grid_offsets = (STE_multistep.apply(grid_offsets, Q_offsets, pc._offset.mean())).detach()
        torch.cuda.synchronize(); time_sub = time.time() - t1
    else:
        pass  # decoded_version: use raw values

    # ============================================================
    #  View-dependent Feature Processing
    # ============================================================
    ob_view = anchor - viewpoint_camera.camera_center
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist

    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)
        feat = feat.unsqueeze(dim=-1)
        feat = \
            feat[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1] + \
            feat[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2] + \
            feat[:, ::1, :1] * bank_weight[:, :, 2:]
        feat = feat.squeeze(dim=-1)

    # ============================================================
    #  MLP Inference (always include distance)
    # ============================================================
    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)

    neural_opacity = pc.get_opacity_mlp(cat_local_view)
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity > 0.0)
    mask = mask.view(-1)
    opacity = neural_opacity[mask]

    color = pc.get_color_mlp(cat_local_view)
    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])

    scale_rot = pc.get_cov_mlp(cat_local_view)
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])

    # ============================================================
    #  Post-processing
    # ============================================================
    offsets = grid_offsets.view([-1, 3])
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

    scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
    rot = pc.rotation_activation(scale_rot[:, 3:7])
    offsets = offsets * scaling_repeat[:, :3]
    xyz = repeat_anchor + offsets

    # ============================================================
    #  Binary Grid Mask Application (from HAC-plus)
    # ============================================================
    binary_grid_masks_pergaussian = binary_grid_masks.view(-1, 1)
    if is_training:
        opacity = opacity * binary_grid_masks_pergaussian[mask]
        scaling = scaling * binary_grid_masks_pergaussian[mask]
    else:
        the_mask = (binary_grid_masks_pergaussian[mask]).to(torch.bool)
        the_mask = the_mask[:, 0]
        xyz = xyz[the_mask]
        color = color[the_mask]
        opacity = opacity[the_mask]
        scaling = scaling[the_mask]
        rot = rot[the_mask]

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param
    else:
        return xyz, color, opacity, scaling, rot, time_sub


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           visible_mask=None, retain_grad=False, step=0):
    """Render the scene."""
    is_training = pc.get_color_mlp.training

    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param = \
            generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, step=step)
    else:
        xyz, color, opacity, scaling, rot, time_sub = \
            generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, step=step)

    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None)

    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "bit_per_param": bit_per_param,
                "bit_per_feat_param": bit_per_feat_param,
                "bit_per_scaling_param": bit_per_scaling_param,
                "bit_per_offsets_param": bit_per_offsets_param,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "time_sub": time_sub,
                }


def prefilter_voxel(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
                    override_color=None):
    """View frustum culling for voxels."""
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(
        means3D=means3D,
        scales=scales[:, :3],
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return radii_pure > 0
