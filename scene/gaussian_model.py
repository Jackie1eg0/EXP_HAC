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

import torch
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding
from scene.hac_model import BinaryHashGrid, HACContextModel
from scene.hac_modules import (
    MultiResolutionHashGrid,
    ContextMLP,
    AdaptiveQuantizationModule,
    FactorizedEntropyModel,
    RateDistortionLoss,
)

    
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
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 voxel_size: float=0.01,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 # ---- HAC++ 新增参数 ----
                 use_hash_grid: bool = False,
                 hash_n_levels: int = 16,
                 hash_n_features_per_level: int = 2,
                 hash_log2_hashmap_size: int = 19,
                 hash_base_resolution: int = 16,
                 hash_max_resolution: int = 4096,
                 context_hidden_dim: int = 128,
                 geometry_context_dim: int = 64,
                 aqm_init_step_size: float = 0.01,
                 lambda_rate: float = 1e-4,
                 # ---- HAC++ 渐进式训练调度 ----
                 aqm_start_iter: int = 10000,
                 rate_start_iter: int = 20000,
                 ):

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist

        # ---- HAC++ 模式标志 ----
        self.use_hash_grid = use_hash_grid
        self.lambda_rate = lambda_rate
        # ---- HAC++ 渐进式训练调度 ----
        self.aqm_start_iter = aqm_start_iter
        self.rate_start_iter = rate_start_iter
        self.current_iteration = 0  # 由训练循环更新

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
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

        # ================================================================
        #  HAC++ 模式: 使用 HashGrid + ContextMLP + AQM + 熵模型
        # ================================================================
        if self.use_hash_grid:
            hash_feat_dim = hash_n_levels * hash_n_features_per_level

            # --- 多分辨率哈希网格 ---
            self.hash_grid = MultiResolutionHashGrid(
                n_levels=hash_n_levels,
                n_features_per_level=hash_n_features_per_level,
                log2_hashmap_size=hash_log2_hashmap_size,
                base_resolution=hash_base_resolution,
                max_resolution=hash_max_resolution,
            ).cuda()

            # --- 上下文 MLP (含 Intra-anchor 依赖) ---
            self.context_mlp = ContextMLP(
                hash_feat_dim=hash_feat_dim,
                anchor_feat_dim=feat_dim,
                hidden_dim=context_hidden_dim,
                geometry_context_dim=geometry_context_dim,
                n_offsets=n_offsets,
                appearance_dim=appearance_dim,
                add_opacity_dist=add_opacity_dist,
                add_cov_dist=add_cov_dist,
                add_color_dist=add_color_dist,
            ).cuda()

            # --- 自适应量化模块 (AQM) ---
            # 对 Stage 1 解码出的各属性分别建立量化器
            # anchor_feat(feat_dim) + offset(n_offsets*3) + scaling(6) + rotation(4) + opacity(1)
            self.aqm_anchor_feat = AdaptiveQuantizationModule(feat_dim, aqm_init_step_size).cuda()
            self.aqm_offset      = AdaptiveQuantizationModule(n_offsets * 3, aqm_init_step_size).cuda()
            self.aqm_scaling     = AdaptiveQuantizationModule(6, aqm_init_step_size).cuda()
            self.aqm_rotation    = AdaptiveQuantizationModule(4, aqm_init_step_size).cuda()
            self.aqm_opacity     = AdaptiveQuantizationModule(1, aqm_init_step_size).cuda()

            # --- 因子化熵模型 (用于比特率估计) ---
            self.entropy_anchor_feat = FactorizedEntropyModel(feat_dim).cuda()
            self.entropy_offset      = FactorizedEntropyModel(n_offsets * 3).cuda()
            self.entropy_scaling     = FactorizedEntropyModel(6).cuda()
            self.entropy_rotation    = FactorizedEntropyModel(4).cuda()
            self.entropy_opacity     = FactorizedEntropyModel(1).cuda()

            # --- Rate-Distortion 损失 ---
            self.rd_loss = RateDistortionLoss(
                lambda_rate=lambda_rate,
                lambda_ssim=0.2,
                lambda_scaling=0.01,
            )

        # ================================================================
        #  原始 Scaffold-GS 模式 (向后兼容)
        # ================================================================
        if not self.use_hash_grid:
            if self.use_feat_bank:
                self.mlp_feature_bank = nn.Sequential(
                    nn.Linear(3+1, feat_dim),
                    nn.ReLU(True),
                    nn.Linear(feat_dim, 3),
                    nn.Softmax(dim=1)
                ).cuda()

            self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
            self.mlp_opacity = nn.Sequential(
                nn.Linear(feat_dim+3+self.opacity_dist_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, n_offsets),
                nn.Tanh()
            ).cuda()

            self.add_cov_dist = add_cov_dist
            self.cov_dist_dim = 1 if self.add_cov_dist else 0
            self.mlp_cov = nn.Sequential(
                nn.Linear(feat_dim+3+self.cov_dist_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 7*self.n_offsets),
            ).cuda()

            self.color_dist_dim = 1 if self.add_color_dist else 0
            self.mlp_color = nn.Sequential(
                nn.Linear(feat_dim+3+self.color_dist_dim+self.appearance_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3*self.n_offsets),
                nn.Sigmoid()
            ).cuda()
        else:
            # HAC++ 模式下, 用 ContextMLP 的 Stage 2 替代原始 MLPs
            # 为保持 property 访问不报错, 设置别名
            self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
            self.cov_dist_dim = 1 if self.add_cov_dist else 0
            self.color_dist_dim = 1 if self.add_color_dist else 0
            self.mlp_opacity = self.context_mlp.opacity_decoder
            self.mlp_cov = self.context_mlp.cov_decoder
            self.mlp_color = self.context_mlp.color_decoder

        # ---- HAC++ compression components ----
        # hac_mode: 0 = off, 1 = base Q0 noise, 2 = full HAC++
        self.hac_mode = 0
        # hash_grid 和 context_model 仅在未使用 hash_grid 时初始化为 None
        # (使用 hash_grid 时, 在上面的 if self.use_hash_grid 分支中已创建)
        if not self.use_hash_grid:
            self.hash_grid = None
        self.context_model = None


    def eval(self):
        if self.use_hash_grid:
            self.hash_grid.eval()
            self.context_mlp.eval()
        else:
            self.mlp_opacity.eval()
            self.mlp_cov.eval()
            self.mlp_color.eval()
            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()
        if self.hash_grid is not None:
            self.hash_grid.eval()
        if self.context_model is not None:
            self.context_model.eval()

    def train(self):
        if self.use_hash_grid:
            self.hash_grid.train()
            self.context_mlp.train()
        else:
            self.mlp_opacity.train()
            self.mlp_cov.train()
            self.mlp_color.train()
            if self.use_feat_bank:
                self.mlp_feature_bank.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()
        if self.hash_grid is not None:
            self.hash_grid.train()
        if self.context_model is not None:
            self.context_model.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._local,
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
        self._local,
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

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    
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
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    # ==============================================================
    #  HAC++ 核心方法
    # ==============================================================

    def decode_anchor_attributes(self, anchor_positions: torch.Tensor, is_training: bool = True):
        """
        [HAC++ 专用] 通过 HashGrid + ContextMLP Stage 1 解码锚点级属性。

        渐进式训练调度:
          Phase 1 [0, aqm_start):     纯解码, 无量化噪声, 无熵估计
          Phase 2 [aqm_start, rate_start): 启用 AQM 量化 (STE), 无 Rate Loss
          Phase 3 [rate_start, end):  完整 AQM + 熵估计 (供 Rate Loss 使用)

        Args:
            anchor_positions: [N, 3]  锚点世界坐标
            is_training:      bool    是否训练模式 (影响量化行为)

        Returns:
            decoded: dict  包含锚点属性
            rate_dict: dict  各属性的比特率估计 (Phase 3 才有内容)
        """
        assert self.use_hash_grid, "decode_anchor_attributes 仅在 HAC++ 模式下可用"

        N = anchor_positions.shape[0]
        it = self.current_iteration

        # 判断当前训练阶段
        apply_aqm = is_training and (it >= self.aqm_start_iter)
        compute_rate = is_training and (it >= self.rate_start_iter)

        # Step 1: 哈希网格查询 (已内置分块 + 梯度检查点)
        hash_features = self.hash_grid(anchor_positions)  # [N, hash_out_dim]

        # Step 2: Context MLP Stage 1 — 几何属性解码 (使用梯度检查点)
        if is_training and torch.is_grad_enabled():
            geom = grad_checkpoint(
                self.context_mlp.decode_geometry, hash_features,
                use_reentrant=False
            )
        else:
            geom = self.context_mlp.decode_geometry(hash_features)
        del hash_features  # 释放: 后续不再需要

        # 提取需要的张量, 立即释放 geom dict 中的冗余引用
        raw_anchor_feat = geom['anchor_feat']
        raw_offsets     = geom['offsets']
        raw_scaling     = geom['scaling']
        raw_rotation    = geom['rotation']
        raw_opacity     = geom['opacity']
        geometry_context = geom['geometry_context']
        del geom  # 释放 dict 引用

        # Step 3: 条件性 AQM 量化
        rate_dict = {}

        if apply_aqm:
            q_anchor_feat, step_af = self.aqm_anchor_feat(raw_anchor_feat, is_training)
            q_offsets, step_off    = self.aqm_offset(raw_offsets, is_training)
            q_scaling, step_sc     = self.aqm_scaling(raw_scaling, is_training)
            q_rotation, step_rot   = self.aqm_rotation(raw_rotation, is_training)
            q_opacity, step_opa    = self.aqm_opacity(raw_opacity, is_training)

            if compute_rate:
                rate_af, _ = self.entropy_anchor_feat(raw_anchor_feat, step_af)
                rate_dict['anchor_feat'] = (rate_af, N * self.feat_dim)
                rate_off, _ = self.entropy_offset(raw_offsets, step_off)
                rate_dict['offset'] = (rate_off, N * self.n_offsets * 3)
                rate_sc, _ = self.entropy_scaling(raw_scaling, step_sc)
                rate_dict['scaling'] = (rate_sc, N * 6)
                rate_rot, _ = self.entropy_rotation(raw_rotation, step_rot)
                rate_dict['rotation'] = (rate_rot, N * 4)
                rate_opa, _ = self.entropy_opacity(raw_opacity, step_opa)
                rate_dict['opacity'] = (rate_opa, N * 1)
            del raw_anchor_feat, raw_offsets, raw_scaling, raw_rotation, raw_opacity
        elif not is_training:
            q_anchor_feat, _ = self.aqm_anchor_feat(raw_anchor_feat, False)
            q_offsets, _     = self.aqm_offset(raw_offsets, False)
            q_scaling, _     = self.aqm_scaling(raw_scaling, False)
            q_rotation, _    = self.aqm_rotation(raw_rotation, False)
            q_opacity, _     = self.aqm_opacity(raw_opacity, False)
            del raw_anchor_feat, raw_offsets, raw_scaling, raw_rotation, raw_opacity
        else:
            # Phase 1: 直接使用解码值
            q_anchor_feat = raw_anchor_feat
            q_offsets     = raw_offsets
            q_scaling     = raw_scaling
            q_rotation    = raw_rotation
            q_opacity     = raw_opacity

        decoded = {
            'anchor_feat': q_anchor_feat,
            'offsets': q_offsets,
            'scaling': q_scaling,
            'rotation': q_rotation,
            'opacity': q_opacity,
            'geometry_context': geometry_context,
        }

        return decoded, rate_dict

    def decode_appearance_from_context(
        self,
        anchor_feat: torch.Tensor,
        geometry_context: torch.Tensor,
        view_dir: torch.Tensor,
        ob_dist: torch.Tensor = None,
        appearance_embed: torch.Tensor = None,
    ):
        """
        [HAC++ 专用] 通过 ContextMLP Stage 2 解码视角相关的外观属性。

        这是 Intra-anchor context 的关键: 外观解码依赖于几何阶段的上下文特征。

        Args:
            anchor_feat:      [N, feat_dim]  (来自 decode_anchor_attributes, 已量化)
            geometry_context: [N, ctx_dim]   (来自 decode_anchor_attributes)
            view_dir:         [N, 3]
            ob_dist:          [N, 1]  (可选)
            appearance_embed: [N, appearance_dim]  (可选)

        Returns:
            dict 包含 neural_opacity, color, scale_rot
        """
        assert self.use_hash_grid, "decode_appearance_from_context 仅在 HAC++ 模式下可用"

        return self.context_mlp.decode_appearance(
            anchor_feat=anchor_feat,
            geometry_context=geometry_context,
            view_dir=view_dir,
            ob_dist=ob_dist,
            appearance_embed=appearance_embed,
        )

    def _sync_decoded_placeholders(self):
        """
        [HAC++] 将 HashGrid+ContextMLP 解码出的真实属性值同步到占位符张量。
        
        在 anchor_growing/prefilter_voxel 等需要读取 offset/scaling/rotation 的地方调用。
        HAC++ 模式下这些属性不再是独立可优化参数，而是由网络解码得到，
        但 growing/pruning 等操作仍需读取其当前值。
        
        注意: 此操作应在 torch.no_grad() 下运行，且不会太频繁调用。
        """
        if not self.use_hash_grid:
            return
        
        with torch.no_grad():
            decoded, _ = self.decode_anchor_attributes(self._anchor, is_training=False)
            self._anchor_feat.data.copy_(decoded['anchor_feat'])
            self._offset.data.copy_(decoded['offsets'].view(-1, self.n_offsets, 3))
            self._scaling.data.copy_(decoded['scaling'])
            self._rotation.data.copy_(decoded['rotation'])
            self._opacity.data.copy_(decoded['opacity'])

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        
        return data

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::self.ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        
        
        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 锚点位置始终是可优化参数
        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))

        if self.use_hash_grid:
            # ---- HAC++ 模式 ----
            # 更新哈希网格包围盒
            self.hash_grid.update_bbox(fused_point_cloud.detach())
            print(f'[HAC++] HashGrid bbox updated: '
                  f'min={self.hash_grid.bbox_min.cpu().numpy()}, '
                  f'max={self.hash_grid.bbox_max.cpu().numpy()}')

            # HAC++ 模式下, offset/scaling/rotation/opacity 由 ContextMLP 解码,
            # 不再作为独立的可优化参数存储。
            # 但仍需保留张量引用以兼容 anchor_growing/prune 等操作。
            # 使用不参与梯度的 placeholder (初始值会被 decode 覆盖)。
            offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
            anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

            self._offset = nn.Parameter(offsets.requires_grad_(False))
            self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(False))
            self._scaling = nn.Parameter(scales.requires_grad_(False))
            self._rotation = nn.Parameter(rots.requires_grad_(False))
            self._opacity = nn.Parameter(opacities.requires_grad_(False))
        else:
            # ---- 原始 Scaffold-GS 模式 ----
            offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
            anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

            self._offset = nn.Parameter(offsets.requires_grad_(True))
            self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(False))
            self._opacity = nn.Parameter(opacities.requires_grad_(False))

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        if self.use_hash_grid:
            # ---- HAC++ 模式: 优化 HashGrid + ContextMLP + AQM + 熵模型 ----
            l = [
                # 锚点位置仍然可优化
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                # HAC++ 模式下 offset/feat/scaling/rotation/opacity 不作为直接优化参数,
                # 但需要在 optimizer 中保持占位以兼容 anchor_growing/prune
                {'params': [self._offset], 'lr': 0.0, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': 0.0, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': 0.0, "name": "opacity"},
                {'params': [self._scaling], 'lr': 0.0, "name": "scaling"},
                {'params': [self._rotation], 'lr': 0.0, "name": "rotation"},

                # HashGrid 哈希表参数
                {'params': self.hash_grid.parameters(), 'lr': training_args.feature_lr, "name": "hash_grid"},
                # ContextMLP 所有参数
                {'params': self.context_mlp.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "context_mlp"},

                # AQM 步长参数
                {'params': self.aqm_anchor_feat.parameters(), 'lr': 1e-3, "name": "aqm_anchor_feat"},
                {'params': self.aqm_offset.parameters(), 'lr': 1e-3, "name": "aqm_offset"},
                {'params': self.aqm_scaling.parameters(), 'lr': 1e-3, "name": "aqm_scaling"},
                {'params': self.aqm_rotation.parameters(), 'lr': 1e-3, "name": "aqm_rotation"},
                {'params': self.aqm_opacity.parameters(), 'lr': 1e-3, "name": "aqm_opacity"},

                # 熵模型参数
                {'params': self.entropy_anchor_feat.parameters(), 'lr': 1e-3, "name": "entropy_anchor_feat"},
                {'params': self.entropy_offset.parameters(), 'lr': 1e-3, "name": "entropy_offset"},
                {'params': self.entropy_scaling.parameters(), 'lr': 1e-3, "name": "entropy_scaling"},
                {'params': self.entropy_rotation.parameters(), 'lr': 1e-3, "name": "entropy_rotation"},
                {'params': self.entropy_opacity.parameters(), 'lr': 1e-3, "name": "entropy_opacity"},
            ]

            if self.appearance_dim > 0:
                l.append({'params': self.embedding_appearance.parameters(),
                          'lr': training_args.appearance_lr_init, "name": "embedding_appearance"})

        elif self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        elif self.appearance_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

        # HAC++ 学习率调度 (使用 context_mlp 的统一调度)
        if self.use_hash_grid:
            self.hash_grid_scheduler_args = get_expon_lr_func(
                lr_init=training_args.feature_lr,
                lr_final=training_args.feature_lr * 0.01,
                lr_delay_mult=0.01,
                max_steps=training_args.position_lr_max_steps)
            self.context_mlp_scheduler_args = get_expon_lr_func(
                lr_init=training_args.mlp_opacity_lr_init,
                lr_final=training_args.mlp_opacity_lr_final,
                lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                max_steps=training_args.mlp_opacity_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                if not self.use_hash_grid:
                    lr = self.offset_scheduler_args(iteration)
                    param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            # ---- HAC++ 学习率调度 ----
            if self.use_hash_grid:
                if param_group["name"] == "hash_grid":
                    param_group['lr'] = self.hash_grid_scheduler_args(iteration)
                if param_group["name"] == "context_mlp":
                    param_group['lr'] = self.context_mlp_scheduler_args(iteration)
            if self.hac_mode >= 2:
                if param_group["name"] == "context_model":
                    lr = self.context_model_scheduler_args(iteration)
                    param_group['lr'] = lr
            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
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
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
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


    def _is_neural_param_group(self, name: str) -> bool:
        """
        判断某个 optimizer param_group 是否为神经网络模块参数
        (即不应在 anchor growing/pruning 时被 cat/prune 的参数组)。
        """
        skip_keywords = ['mlp', 'conv', 'feat_base', 'embedding']
        if self.use_hash_grid:
            skip_keywords += [
                'hash_grid', 'context_mlp',
                'aqm_anchor_feat', 'aqm_offset', 'aqm_scaling', 'aqm_rotation', 'aqm_opacity',
                'entropy_anchor_feat', 'entropy_offset', 'entropy_scaling', 'entropy_rotation', 'entropy_opacity',
            ]
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


    # statis grad information to guide liftting. 
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
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
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
            
        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    
    def anchor_growing(self, grads, threshold, offset_mask):
        ## 
        # [HAC++] 同步解码后的属性到占位符，使 growing 逻辑能读取真实的 offset/scaling
        self._sync_decoded_placeholders()

        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)


            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            
            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
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
                self._opacity = optimizable_tensors["opacity"]
                


    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)
        
        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))

        if self.use_hash_grid:
            # ---- HAC++ 模式: 统一保存所有神经网络模块 ----
            state = {
                'hash_grid': self.hash_grid.state_dict(),
                'context_mlp': self.context_mlp.state_dict(),
                'aqm_anchor_feat': self.aqm_anchor_feat.state_dict(),
                'aqm_offset': self.aqm_offset.state_dict(),
                'aqm_scaling': self.aqm_scaling.state_dict(),
                'aqm_rotation': self.aqm_rotation.state_dict(),
                'aqm_opacity': self.aqm_opacity.state_dict(),
                'entropy_anchor_feat': self.entropy_anchor_feat.state_dict(),
                'entropy_offset': self.entropy_offset.state_dict(),
                'entropy_scaling': self.entropy_scaling.state_dict(),
                'entropy_rotation': self.entropy_rotation.state_dict(),
                'entropy_opacity': self.entropy_opacity.state_dict(),
            }
            if self.appearance_dim > 0:
                state['appearance'] = self.embedding_appearance.state_dict()
            torch.save(state, os.path.join(path, 'hac_checkpoints.pth'))
            return

        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            if self.appearance_dim:
                self.embedding_appearance.eval()
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()

        elif mode == 'unite':
            if self.use_feat_bank:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            elif self.appearance_dim > 0:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            else:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    }, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if self.use_hash_grid:
            # ---- HAC++ 模式: 加载所有神经网络模块 ----
            checkpoint = torch.load(os.path.join(path, 'hac_checkpoints.pth'))
            self.hash_grid.load_state_dict(checkpoint['hash_grid'])
            self.context_mlp.load_state_dict(checkpoint['context_mlp'])
            self.aqm_anchor_feat.load_state_dict(checkpoint['aqm_anchor_feat'])
            self.aqm_offset.load_state_dict(checkpoint['aqm_offset'])
            self.aqm_scaling.load_state_dict(checkpoint['aqm_scaling'])
            self.aqm_rotation.load_state_dict(checkpoint['aqm_rotation'])
            self.aqm_opacity.load_state_dict(checkpoint['aqm_opacity'])
            self.entropy_anchor_feat.load_state_dict(checkpoint['entropy_anchor_feat'])
            self.entropy_offset.load_state_dict(checkpoint['entropy_offset'])
            self.entropy_scaling.load_state_dict(checkpoint['entropy_scaling'])
            self.entropy_rotation.load_state_dict(checkpoint['entropy_rotation'])
            self.entropy_opacity.load_state_dict(checkpoint['entropy_opacity'])
            if self.appearance_dim > 0 and 'appearance' in checkpoint:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
            return

        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError

    # ==================================================================
    #  HAC++ Methods
    # ==================================================================

    def init_hac(self):
        """
        Initialise the Binary Hash Grid and Context Model.
        Called at the HAC++ entropy-start milestone (e.g. iteration 10000).
        """
        self.hash_grid = BinaryHashGrid().cuda()
        self.hash_grid.update_bbox(self._anchor.detach())

        hash_dim = self.hash_grid.output_dim   # (12+4)*4 = 64

        # Ensure feat_dim is divisible by num_chunks
        num_chunks = 4
        if self.feat_dim % num_chunks != 0:
            # Fall back to a divisor
            for nc in [4, 2, 1]:
                if self.feat_dim % nc == 0:
                    num_chunks = nc
                    break

        self.context_model = HACContextModel(
            hash_feat_dim=hash_dim,
            anchor_feat_dim=self.feat_dim,
            scaling_dim=self._scaling.shape[1],  # typically 6
            n_offsets=self.n_offsets,
            num_mixtures=3,
            num_chunks=num_chunks,
        ).cuda()

    def add_hac_to_optimizer(self, hash_lr=1e-3, context_lr=5e-4):
        """Add HAC++ parameters to the existing Adam optimizer."""
        self.optimizer.add_param_group(
            {'params': list(self.hash_grid.parameters()),
             'lr': hash_lr, "name": "hash_grid"}
        )
        self.optimizer.add_param_group(
            {'params': list(self.context_model.parameters()),
             'lr': context_lr, "name": "context_model"}
        )
        # Learning rate schedulers
        self.hash_grid_scheduler_args = get_expon_lr_func(
            lr_init=hash_lr, lr_final=hash_lr * 0.01,
            lr_delay_mult=0.01, max_steps=30_000)
        self.context_model_scheduler_args = get_expon_lr_func(
            lr_init=context_lr, lr_final=context_lr * 0.01,
            lr_delay_mult=0.01, max_steps=30_000)

    def get_quantized_attributes(self, visible_mask):
        """
        Return (feat, grid_scaling, grid_offsets) for rendering.

        hac_mode 0 : raw values (standard Scaffold-GS)
        hac_mode 1 : base Q0 noise / round  (warmup)
        hac_mode 2 : adaptive quantisation from hash context
        """
        feat = self._anchor_feat[visible_mask]
        scaling_raw = self._scaling[visible_mask]
        offset = self._offset[visible_mask]

        if self.hac_mode == 0:
            return feat, self.scaling_activation(scaling_raw), offset

        N = feat.shape[0]
        is_training = self.mlp_color.training

        if self.hac_mode == 1:
            # Base-Q0 noise / round (no hash grid needed)
            q_feat = torch.full_like(feat, HACContextModel.Q0_FEAT)
            q_scaling = torch.full_like(scaling_raw, HACContextModel.Q0_SCALING)
            q_offset = torch.full(
                (N, self.n_offsets * 3), HACContextModel.Q0_OFFSET,
                device=feat.device)
        else:
            # Adaptive quantisation via hash context
            anchor_pos = self._anchor[visible_mask]
            hash_feat = self.hash_grid(anchor_pos.detach())
            q_feat, q_scaling, q_offset = \
                self.context_model.compute_quantization_step(hash_feat)

        feat = HACContextModel.quantize(feat, q_feat, is_training)
        scaling_raw = HACContextModel.quantize(
            scaling_raw, q_scaling, is_training)

        offset_flat = offset.reshape(N, -1)
        offset_flat = HACContextModel.quantize(offset_flat, q_offset, is_training)
        offset = offset_flat.view(N, self.n_offsets, 3)

        return feat, self.scaling_activation(scaling_raw), offset

    def compute_entropy_loss(self, anchor_indices):
        """
        Compute per-anchor bits for a sampled subset of anchors.

        Returns: bits_feat, bits_scaling, bits_offset  (each [S])
                 and the three quantised tensors (unused by caller typically).
        """
        feat = self._anchor_feat[anchor_indices]
        scaling = self._scaling[anchor_indices]
        offset = self._offset[anchor_indices]
        anchor_pos = self._anchor[anchor_indices]

        hash_feat = self.hash_grid(anchor_pos.detach())
        is_training = self.mlp_color.training
        use_adaptive = (self.hac_mode >= 2)

        return self.context_model(
            hash_feat, feat, scaling, offset,
            training=is_training, use_adaptive_q=use_adaptive,
        )

    # ---- Save / Load HAC++ checkpoints ----
    def save_hac_checkpoints(self, path):
        if self.hac_mode >= 2 and self.hash_grid is not None:
            mkdir_p(os.path.dirname(path) if not os.path.isdir(path) else path)
            torch.save({
                'hash_grid': self.hash_grid.state_dict(),
                'context_model': self.context_model.state_dict(),
                'hac_mode': self.hac_mode,
            }, os.path.join(path, 'hac_checkpoints.pth'))

    def load_hac_checkpoints(self, path):
        ckpt = os.path.join(path, 'hac_checkpoints.pth')
        if os.path.exists(ckpt):
            data = torch.load(ckpt)
            self.init_hac()
            self.hash_grid.load_state_dict(data['hash_grid'])
            self.context_model.load_state_dict(data['context_model'])
            self.hac_mode = data['hac_mode']
