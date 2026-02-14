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
# Restructured to match HAC-plus architecture:
#   - All attributes (_anchor_feat, _offset, _scaling) are DIRECTLY optimizable
#   - Hash grid (encoding_xyz) is used ONLY for entropy coding context
#   - Added learnable binary offset masks (_mask)
#   - Added mlp_grid + mlp_deform for entropy parameter prediction
#

import os
import numpy as np
import torch
from torch import nn
from functools import reduce
from torch_scatter import scatter_max
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from scipy.spatial import cKDTree

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, strip_symmetric, build_scaling_rotation
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p
from utils.encodings import STE_binary, STE_multistep, get_binary_vxl_size
from utils.entropy_models import Entropy_gaussian, Entropy_gaussian_mix_prob_2
from scene.hac_modules import MultiResolutionHashGrid, TCNN_AVAILABLE
if TCNN_AVAILABLE:
    from scene.hac_modules import TcnnHashGrid


# ================================================================
#  Channel-wise Autoregressive Context Model (from HAC-plus)
# ================================================================

class Channel_CTX_fea(nn.Module):
    """Channel-wise autoregressive context for feat_dim=50, split into 5 chunks of 10."""
    def __init__(self):
        super().__init__()
        self.MLP_d0 = nn.Sequential(
            nn.Linear(50 * 3 + 10 * 0, 20 * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20 * 2, 10 * 3),
        )
        self.MLP_d1 = nn.Sequential(
            nn.Linear(50 * 3 + 10 * 1, 20 * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20 * 2, 10 * 3),
        )
        self.MLP_d2 = nn.Sequential(
            nn.Linear(50 * 3 + 10 * 2, 20 * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20 * 2, 10 * 3),
        )
        self.MLP_d3 = nn.Sequential(
            nn.Linear(50 * 3 + 10 * 3, 20 * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20 * 2, 10 * 3),
        )
        self.MLP_d4 = nn.Sequential(
            nn.Linear(50 * 3 + 10 * 4, 20 * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20 * 2, 10 * 3),
        )

    def forward(self, fea_q, mean_scale, to_dec=-1):
        d0, d1, d2, d3, d4 = torch.split(fea_q, split_size_or_sections=[10, 10, 10, 10, 10], dim=-1)
        mean_d0, scale_d0, prob_d0 = torch.chunk(self.MLP_d0(torch.cat([mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d1(torch.cat([d0, mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d2(torch.cat([d0, d1, mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d3(torch.cat([d0, d1, d2, mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d4, scale_d4, prob_d4 = torch.chunk(self.MLP_d4(torch.cat([d0, d1, d2, d3, mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_adj = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3, mean_d4], dim=-1)
        scale_adj = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3, scale_d4], dim=-1)
        prob_adj = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3, prob_d4], dim=-1)

        if to_dec == 0: return mean_d0, scale_d0, prob_d0
        if to_dec == 1: return mean_d1, scale_d1, prob_d1
        if to_dec == 2: return mean_d2, scale_d2, prob_d2
        if to_dec == 3: return mean_d3, scale_d3, prob_d3
        if to_dec == 4: return mean_d4, scale_d4, prob_d4
        return mean_adj, scale_adj, prob_adj


class Channel_CTX_fea_tiny(nn.Module):
    """Simplified channel context for synthetic NeRF scenes."""
    def __init__(self):
        super().__init__()
        self.mean_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.scale_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.prob_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.MLP_d1 = nn.Sequential(nn.Linear(10 * 1, 10 * 3), nn.LeakyReLU(inplace=True), nn.Linear(10 * 3, 10 * 3))
        self.MLP_d2 = nn.Sequential(nn.Linear(10 * 2, 10 * 3), nn.LeakyReLU(inplace=True), nn.Linear(10 * 3, 10 * 3))
        self.MLP_d3 = nn.Sequential(nn.Linear(10 * 3, 10 * 3), nn.LeakyReLU(inplace=True), nn.Linear(10 * 3, 10 * 3))
        self.MLP_d4 = nn.Sequential(nn.Linear(10 * 4, 10 * 3), nn.LeakyReLU(inplace=True), nn.Linear(10 * 3, 10 * 3))

    def forward(self, fea_q, mean_scale, to_dec=-1):
        NN = fea_q.shape[0]
        d0, d1, d2, d3, d4 = torch.split(fea_q, split_size_or_sections=[10, 10, 10, 10, 10], dim=-1)
        mean_d0, scale_d0, prob_d0 = self.mean_d0.repeat(NN, 1), self.scale_d0.repeat(NN, 1), self.prob_d0.repeat(NN, 1)
        mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d1(torch.cat([d0], dim=-1)), chunks=3, dim=-1)
        mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d2(torch.cat([d0, d1], dim=-1)), chunks=3, dim=-1)
        mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d3(torch.cat([d0, d1, d2], dim=-1)), chunks=3, dim=-1)
        mean_d4, scale_d4, prob_d4 = torch.chunk(self.MLP_d4(torch.cat([d0, d1, d2, d3], dim=-1)), chunks=3, dim=-1)
        mean_adj = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3, mean_d4], dim=-1)
        scale_adj = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3, scale_d4], dim=-1)
        prob_adj = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3, prob_d4], dim=-1)

        if to_dec == 0: return mean_d0, scale_d0, prob_d0
        if to_dec == 1: return mean_d1, scale_d1, prob_d1
        if to_dec == 2: return mean_d2, scale_d2, prob_d2
        if to_dec == 3: return mean_d3, scale_d3, prob_d3
        if to_dec == 4: return mean_d4, scale_d4, prob_d4
        return mean_adj, scale_adj, prob_adj


# ================================================================
#  Gaussian Model (aligned with HAC-plus architecture)
# ================================================================

def safe_distCUDA2(points_tensor):
    """Memory-safe KNN distance calculation with CPU fallback."""
    try:
        dist2 = distCUDA2(points_tensor).float().cuda()
        return dist2
    except (RuntimeError, MemoryError) as e:
        if 'out of memory' in str(e).lower() or isinstance(e, MemoryError):
            print(f"[Memory] distCUDA2 OOM, falling back to CPU KDTree...")
            torch.cuda.empty_cache()
            points_np = points_tensor.detach().cpu().numpy().astype(np.float64)
            tree = cKDTree(points_np)
            dists, _ = tree.query(points_np, k=4)
            mean_dist_sq = np.mean(dists[:, 1:4] ** 2, axis=1).astype(np.float32)
            del points_np, tree, dists
            return torch.from_numpy(mean_dist_sq).float().cuda()
        else:
            raise


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self,
                 feat_dim: int = 50,
                 n_offsets: int = 10,
                 voxel_size: float = 0.001,
                 update_depth: int = 3,
                 update_init_factor: int = 16,
                 update_hierachy_factor: int = 4,
                 use_feat_bank: bool = False,
                 # Hash grid parameters for entropy context
                 hash_n_levels: int = 16,
                 hash_n_features_per_level: int = 2,
                 hash_log2_hashmap_size: int = 19,
                 hash_base_resolution: int = 16,
                 hash_max_resolution: int = 4096,
                 decoded_version: bool = False,
                 is_synthetic_nerf: bool = False,
                 ):
        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank
        self.decoded_version = decoded_version

        # Anchor bounds for hash grid normalization
        self.x_bound_min = torch.zeros(size=[1, 3], device='cuda')
        self.x_bound_max = torch.ones(size=[1, 3], device='cuda')

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._mask = torch.empty(0)       # NEW: learnable per-offset mask
        self._anchor_feat = torch.empty(0)
        self.opacity_accum = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)
        self.anchor_demon = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        # ============================================================
        #  Hash Grid Encoding (for entropy context ONLY, not rendering)
        #  Auto-select: tinycudann (if available) > PyTorch fallback
        # ============================================================
        _hash_args = dict(
            n_levels=hash_n_levels,
            n_features_per_level=hash_n_features_per_level,
            log2_hashmap_size=hash_log2_hashmap_size,
            base_resolution=hash_base_resolution,
            max_resolution=hash_max_resolution,
        )
        if TCNN_AVAILABLE:
            self.encoding_xyz = TcnnHashGrid(**_hash_args).cuda()
            print(f"[GaussianModel] Using tinycudann hash grid (CUDA-accelerated)")
        else:
            self.encoding_xyz = MultiResolutionHashGrid(**_hash_args).cuda()
            print(f"[GaussianModel] Using PyTorch hash grid (optimized)")

        # ============================================================
        #  Standard Scaffold-GS MLPs (all attributes directly optimized)
        # ============================================================
        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3 + 1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        # MLP input: feat_dim + 3 (view_dir) + 1 (distance) — always include distance
        mlp_input_dim = feat_dim + 3 + 1

        self.mlp_opacity = nn.Sequential(
            nn.Linear(mlp_input_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(mlp_input_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7 * n_offsets),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(mlp_input_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3 * n_offsets),
            nn.Sigmoid()
        ).cuda()

        # ============================================================
        #  Entropy Context MLPs (from HAC-plus)
        # ============================================================
        # mlp_grid: hash_grid_features -> entropy parameters (mean, scale, Q adjustments)
        hash_output_dim = self.encoding_xyz.output_dim
        self.mlp_grid = nn.Sequential(
            nn.Linear(hash_output_dim, feat_dim * 2),
            nn.ReLU(True),
            nn.Linear(feat_dim * 2, (feat_dim + 6 + 3 * n_offsets) * 2 + feat_dim + 1 + 1 + 1),
        ).cuda()

        # mlp_deform: channel-wise autoregressive context for feature entropy
        if not is_synthetic_nerf:
            self.mlp_deform = Channel_CTX_fea().cuda()
        else:
            print('find synthetic nerf, use Channel_CTX_fea_tiny')
            self.mlp_deform = Channel_CTX_fea_tiny().cuda()

        # ============================================================
        #  Entropy Models
        # ============================================================
        self.entropy_gaussian = Entropy_gaussian(Q=1).cuda()
        self.EG_mix_prob_2 = Entropy_gaussian_mix_prob_2(Q=1).cuda()

    # ================================================================
    #  Properties
    # ================================================================

    def get_encoding_params(self):
        """Get hash grid encoding parameters (for binary size estimation)."""
        params = self.encoding_xyz.hash_tables.view(-1, self.encoding_xyz.n_features_per_level)
        params = STE_binary.apply(params)
        return params

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        self.encoding_xyz.eval()
        self.mlp_grid.eval()
        self.mlp_deform.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        self.encoding_xyz.train()
        self.mlp_grid.train()
        self.mlp_deform.train()
        if self.use_feat_bank:
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._mask,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._anchor,
         self._offset,
         self._mask,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self.decoded_version:
            return self._scaling
        return 1.0 * self.scaling_activation(self._scaling)

    @property
    def get_mask(self):
        """Learnable per-offset binary mask with STE."""
        if self.decoded_version:
            return self._mask[:, :self.n_offsets, :]
        mask_sig = torch.sigmoid(self._mask[:, :self.n_offsets, :])
        return ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig

    @property
    def get_mask_anchor(self):
        """Per-anchor mask: anchor is active if any offset is active."""
        mask = self.get_mask  # [N, n_offsets, 1]
        mask_rate = torch.mean(mask, dim=1)  # [N, 1]
        mask_anchor = ((mask_rate > 0.0).float() - mask_rate).detach() + mask_rate
        return mask_anchor  # [N, 1]

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_grid_mlp(self):
        return self.mlp_grid

    @property
    def get_deform_mlp(self):
        return self.mlp_deform

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        """Anchor positions snapped to voxel grid with STE (like HAC-plus)."""
        if self.decoded_version:
            return self._anchor
        anchor = torch.round(self._anchor / self.voxel_size) * self.voxel_size
        anchor = anchor.detach() + (self._anchor - self._anchor.detach())
        return anchor

    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    # ================================================================
    #  Hash Grid Context Methods
    # ================================================================

    @torch.no_grad()
    def update_anchor_bound(self):
        """Update spatial bounds for hash grid normalization."""
        x_bound_min = (torch.min(self._anchor, dim=0, keepdim=True)[0]).detach()
        x_bound_max = (torch.max(self._anchor, dim=0, keepdim=True)[0]).detach()
        for c in range(x_bound_min.shape[-1]):
            x_bound_min[0, c] = x_bound_min[0, c] * 1.2 if x_bound_min[0, c] < 0 else x_bound_min[0, c] * 0.8
        for c in range(x_bound_max.shape[-1]):
            x_bound_max[0, c] = x_bound_max[0, c] * 1.2 if x_bound_max[0, c] > 0 else x_bound_max[0, c] * 0.8
        self.x_bound_min = x_bound_min
        self.x_bound_max = x_bound_max
        # Also update the hash grid bbox to match
        self.encoding_xyz.bbox_min.copy_(x_bound_min.squeeze(0))
        self.encoding_xyz.bbox_max.copy_(x_bound_max.squeeze(0))
        # Invalidate cached inverse range for fast normalization
        self.encoding_xyz._update_bbox_cache()
        print('anchor_bound_updated')

    def calc_interp_feat(self, x):
        """Compute hash grid features for entropy context.
        
        Args:
            x: [N, 3] world coordinates
        Returns:
            features: [N, hash_output_dim]
        """
        assert len(x.shape) == 2 and x.shape[1] == 3
        # The MultiResolutionHashGrid handles normalization internally via bbox
        features = self.encoding_xyz(x)
        return features

    # ================================================================
    #  Point Cloud Initialization
    # ================================================================

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data / voxel_size), axis=0) * voxel_size
        return data

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::1]  # ratio=1

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = safe_distCUDA2(init_points)
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0] * 0.5))
            self.voxel_size = median_dist.item()
            del init_dist, init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')

        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(safe_distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        masks = torch.ones((fused_point_cloud.shape[0], self.n_offsets + 1, 1)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

        # ALL attributes are directly optimizable (key difference from old code)
        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._mask = nn.Parameter(masks.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    # ================================================================
    #  Training Setup
    # ================================================================

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        l = [
            # Core parameters (directly optimizable)
            {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
            {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            # Rendering MLPs
            {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            # Hash grid encoding (entropy context)
            {'params': self.encoding_xyz.parameters(), 'lr': training_args.encoding_xyz_lr_init, "name": "encoding_xyz"},
            # Entropy context MLPs
            {'params': self.mlp_grid.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "mlp_grid"},
            {'params': self.mlp_deform.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},
        ]

        if self.use_feat_bank:
            l.append({'params': self.mlp_feature_bank.parameters(),
                      'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # Learning rate schedulers
        self.anchor_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(
            lr_init=training_args.offset_lr_init * self.spatial_lr_scale,
            lr_final=training_args.offset_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.offset_lr_delay_mult,
            max_steps=training_args.offset_lr_max_steps)
        self.mask_scheduler_args = get_expon_lr_func(
            lr_init=training_args.mask_lr_init * self.spatial_lr_scale,
            lr_final=training_args.mask_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.mask_lr_delay_mult,
            max_steps=training_args.mask_lr_max_steps)
        self.mlp_opacity_scheduler_args = get_expon_lr_func(
            lr_init=training_args.mlp_opacity_lr_init,
            lr_final=training_args.mlp_opacity_lr_final,
            lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
            max_steps=training_args.mlp_opacity_lr_max_steps)
        self.mlp_cov_scheduler_args = get_expon_lr_func(
            lr_init=training_args.mlp_cov_lr_init,
            lr_final=training_args.mlp_cov_lr_final,
            lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
            max_steps=training_args.mlp_cov_lr_max_steps)
        self.mlp_color_scheduler_args = get_expon_lr_func(
            lr_init=training_args.mlp_color_lr_init,
            lr_final=training_args.mlp_color_lr_final,
            lr_delay_mult=training_args.mlp_color_lr_delay_mult,
            max_steps=training_args.mlp_color_lr_max_steps)
        self.encoding_xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.encoding_xyz_lr_init,
            lr_final=training_args.encoding_xyz_lr_final,
            lr_delay_mult=training_args.encoding_xyz_lr_delay_mult,
            max_steps=training_args.encoding_xyz_lr_max_steps)
        self.mlp_grid_scheduler_args = get_expon_lr_func(
            lr_init=training_args.mlp_grid_lr_init,
            lr_final=training_args.mlp_grid_lr_final,
            lr_delay_mult=training_args.mlp_grid_lr_delay_mult,
            max_steps=training_args.mlp_grid_lr_max_steps)
        self.mlp_deform_scheduler_args = get_expon_lr_func(
            lr_init=training_args.mlp_deform_lr_init,
            lr_final=training_args.mlp_deform_lr_final,
            lr_delay_mult=training_args.mlp_deform_lr_delay_mult,
            max_steps=training_args.mlp_deform_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(
                lr_init=training_args.mlp_featurebank_lr_init,
                lr_final=training_args.mlp_featurebank_lr_final,
                lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                max_steps=training_args.mlp_featurebank_lr_max_steps)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            name = param_group["name"]
            if name == "anchor":
                param_group['lr'] = self.anchor_scheduler_args(iteration)
            elif name == "offset":
                param_group['lr'] = self.offset_scheduler_args(iteration)
            elif name == "mask":
                param_group['lr'] = self.mask_scheduler_args(iteration)
            elif name == "mlp_opacity":
                param_group['lr'] = self.mlp_opacity_scheduler_args(iteration)
            elif name == "mlp_cov":
                param_group['lr'] = self.mlp_cov_scheduler_args(iteration)
            elif name == "mlp_color":
                param_group['lr'] = self.mlp_color_scheduler_args(iteration)
            elif name == "encoding_xyz":
                param_group['lr'] = self.encoding_xyz_scheduler_args(iteration)
            elif name == "mlp_grid":
                param_group['lr'] = self.mlp_grid_scheduler_args(iteration)
            elif name == "mlp_deform":
                param_group['lr'] = self.mlp_deform_scheduler_args(iteration)
            elif name == "mlp_featurebank" and self.use_feat_bank:
                param_group['lr'] = self.mlp_featurebank_scheduler_args(iteration)

    # ================================================================
    #  PLY Save / Load
    # ================================================================

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1] * self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                           np.asarray(plydata.elements[0]["y"]),
                           np.asarray(plydata.elements[0]["z"])), axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")],
                             key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("rot")],
                           key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        anchor_feat_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")],
                                   key=lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")],
                              key=lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        # Initialize masks as all-ones
        N = anchor.shape[0]
        masks = torch.ones((N, self.n_offsets + 1, 1), dtype=torch.float, device="cuda")
        self._mask = nn.Parameter(masks.requires_grad_(True))

    # ================================================================
    #  Optimizer Helpers
    # ================================================================

    def _is_neural_param_group(self, name):
        skip_keywords = ['mlp', 'conv', 'feat_base', 'encoding']
        return any(kw in name for kw in skip_keywords)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if self._is_neural_param_group(group['name']):
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # ================================================================
    #  Densification / Pruning
    # ================================================================

    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity < 0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        self.anchor_demon[anchor_visible_mask] += 1

        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if self._is_neural_param_group(group['name']):
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:, 3:]
                    temp[temp > 0.05] = 0.05
                    group["params"][0][:, 3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:, 3:]
                    temp[temp > 0.05] = 0.05
                    group["params"][0][:, 3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_anchor(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)
        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._mask = optimizable_tensors["mask"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def anchor_growing(self, grads, threshold, offset_mask):
        init_length = self.get_anchor.shape[0] * self.n_offsets
        for i in range(self.update_depth):
            cur_threshold = threshold * ((self.update_hierachy_factor // 2) ** i)
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5 ** (i + 1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self.get_anchor.shape[0] * self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:, :3].unsqueeze(dim=1)

            size_factor = self.update_init_factor // (self.update_hierachy_factor ** i)
            cur_size = self.voxel_size * size_factor

            grid_coords = torch.round(self.get_anchor / cur_size).int()
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            chunk_size = 4096
            max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
            remove_duplicates_list = []
            for ci in range(max_iters):
                cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[ci * chunk_size:(ci + 1) * chunk_size, :]).all(-1).any(-1).view(-1)
                remove_duplicates_list.append(cur_remove_duplicates)
            remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:, 0] = 1.0
                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()
                new_masks = torch.ones_like(candidate_anchor[:, 0:1]).unsqueeze(dim=1).repeat([1, self.n_offsets + 1, 1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "mask": new_masks,
                    "opacity": new_opacities,
                }

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._mask = optimizable_tensors["mask"]
                self._opacity = optimizable_tensors["opacity"]

    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        grads = self.offset_gradient_accum / self.offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval * success_threshold * 0.5).squeeze(dim=1)

        self.anchor_growing(grads_norm, grad_threshold, offset_mask)

        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0] * self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0] * self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                                    dtype=torch.int32, device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        prune_mask = (self.opacity_accum < min_opacity * self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval * success_threshold).squeeze(dim=1)
        prune_mask = torch.logical_and(prune_mask, anchors_mask)

        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        if anchors_mask.sum() > 0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0] > 0:
            self.prune_anchor(prune_mask)

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    # ================================================================
    #  MLP Checkpoints Save / Load
    # ================================================================

    def save_mlp_checkpoints(self, path, mode='split'):
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim + 3 + 1).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim + 3 + 1).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim + 3 + 1).cuda()))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3 + 1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            # Also save entropy context modules (cannot be jit.traced easily, use state_dict)
            entropy_state = {
                'encoding_xyz': self.encoding_xyz.state_dict(),
                'mlp_grid': self.mlp_grid.state_dict(),
                'mlp_deform': self.mlp_deform.state_dict(),
            }
            torch.save(entropy_state, os.path.join(path, 'entropy_checkpoints.pth'))

        elif mode == 'unite':
            state = {
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict(),
                'encoding_xyz': self.encoding_xyz.state_dict(),
                'mlp_grid': self.mlp_grid.state_dict(),
                'mlp_deform': self.mlp_deform.state_dict(),
            }
            if self.use_feat_bank:
                state['feature_bank_mlp'] = self.mlp_feature_bank.state_dict()
            torch.save(state, os.path.join(path, 'checkpoints.pth'))

    def load_mlp_checkpoints(self, path, mode='split'):
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            # Load entropy context modules
            entropy_ckpt_path = os.path.join(path, 'entropy_checkpoints.pth')
            if os.path.exists(entropy_ckpt_path):
                entropy_ckpt = torch.load(entropy_ckpt_path)
                self.encoding_xyz.load_state_dict(entropy_ckpt['encoding_xyz'])
                self.mlp_grid.load_state_dict(entropy_ckpt['mlp_grid'])
                self.mlp_deform.load_state_dict(entropy_ckpt['mlp_deform'])
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            if 'encoding_xyz' in checkpoint:
                self.encoding_xyz.load_state_dict(checkpoint['encoding_xyz'])
            if 'mlp_grid' in checkpoint:
                self.mlp_grid.load_state_dict(checkpoint['mlp_grid'])
            if 'mlp_deform' in checkpoint:
                self.mlp_deform.load_state_dict(checkpoint['mlp_deform'])
            if self.use_feat_bank and 'feature_bank_mlp' in checkpoint:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])

    def set_appearance(self, num_cameras):
        # No appearance embedding in HAC-plus style architecture
        pass

    @property
    def get_appearance(self):
        return None
