#
# HAC++ (Hash-grid Assisted Context ++) Modules
#
# 包含以下核心组件：
#   1. MultiResolutionHashGrid  - 多分辨率哈希网格编码 (参考 Instant-NGP)
#   2. ContextMLP               - 带 Intra-anchor 上下文依赖的属性解码器
#   3. AdaptiveQuantizationModule (AQM) - 可微分自适应量化模块 (STE)
#   4. EntropyModel             - 参数化熵估计模型
#   5. RateDistortionLoss       - 率失真联合损失函数
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import numpy as np
import math


# ================================================================
#  1. Multi-Resolution Hash Grid  (Instant-NGP style)
# ================================================================

class MultiResolutionHashGrid(nn.Module):
    """
    多分辨率哈希网格编码，参考 Instant-NGP (Müller et al., 2022)。

    给定 3D 位置，通过多层不同分辨率的哈希表 + 三线性插值，
    输出拼接后的特征向量。

    *** 性能优化版本 ***
    - 所有哈希表合并为单个 Parameter [L, T, F]，避免 ParameterList 逐层索引
    - forward 中全层级批量并行计算 (无 Python for 循环)
    - 消除 .item() CPU-GPU 同步

    参数:
        n_levels            : 分辨率层数 L
        n_features_per_level: 每层特征维度 F
        log2_hashmap_size   : 哈希表大小 T = 2^log2_hashmap_size
        base_resolution     : 最粗分辨率 N_min
        max_resolution      : 最细分辨率 N_max

    输出维度: L * F
    """

    # 用于空间哈希的大质数
    PRIMES = [1, 2654435761, 805459861]

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        max_resolution: int = 2048,
    ):
        super().__init__()

        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 2 ** log2_hashmap_size
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.output_dim = n_levels * n_features_per_level

        # 几何级数增长因子  b = exp( ln(N_max/N_min) / (L-1) )
        if n_levels > 1:
            self.growth_factor = math.exp(
                (math.log(max_resolution) - math.log(base_resolution)) / (n_levels - 1)
            )
        else:
            self.growth_factor = 1.0

        # --- 所有层级的哈希表合并为单个 Parameter: [L, T, F] ---
        self.hash_tables = nn.Parameter(
            torch.zeros(n_levels, self.hashmap_size, n_features_per_level)
        )
        nn.init.uniform_(self.hash_tables, -1e-4, 1e-4)

        # AABB 包围盒 (会在 create_from_pcd 时更新)
        self.register_buffer('bbox_min', torch.tensor([-1.0, -1.0, -1.0]))
        self.register_buffer('bbox_max', torch.tensor([1.0, 1.0, 1.0]))

        # 质数常量 (用于空间哈希)
        self.register_buffer(
            'primes',
            torch.tensor(self.PRIMES, dtype=torch.long)
        )

        # 三线性插值的 8 个角偏移量: [8, 3]
        offsets = torch.tensor(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
             [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
            dtype=torch.long
        )
        self.register_buffer('corner_offsets', offsets)

        # 预计算每层分辨率 (浮点, 用于批量乘法; 同时保留 long 版本)
        resolutions = [
            int(math.ceil(base_resolution * (self.growth_factor ** lvl)))
            for lvl in range(n_levels)
        ]
        self.register_buffer(
            'resolutions',
            torch.tensor(resolutions, dtype=torch.long)
        )
        self.register_buffer(
            'resolutions_float',
            torch.tensor(resolutions, dtype=torch.float32)
        )

    # ------------------------------------------------------------------
    def _spatial_hash(self, int_coords: torch.Tensor) -> torch.Tensor:
        """
        空间哈希: 将 int32 坐标映射到 [0, T) 的 int32 索引。

        Args:
            int_coords: [..., 3]  (int32)
        Returns:
            [...] int32 类型的哈希索引
        """
        # 提升到 int64 做乘法 (避免 int32 溢出), 结果截断回 int32
        x = int_coords[..., 0].long()
        y = int_coords[..., 1].long()
        z = int_coords[..., 2].long()
        result = (x * self.primes[0]) ^ (y * self.primes[1]) ^ (z * self.primes[2])
        return (result % self.hashmap_size).int()  # 结果 < 512K, int32 足够

    # ------------------------------------------------------------------
    def _forward_chunk(self, positions: torch.Tensor) -> torch.Tensor:
        """
        对一小块位置计算哈希特征 (显存友好的核心实现)。

        使用 int32 索引, 减少 ~50% 的索引张量内存。

        Args:
            positions: [M, 3]  (M << N, 一个 chunk)
        Returns:
            [M, L*F]
        """
        M = positions.shape[0]
        L = self.n_levels

        # 1) 归一化
        normalized = (positions - self.bbox_min) / (self.bbox_max - self.bbox_min + 1e-8)
        normalized = torch.clamp(normalized, 0.0, 1.0 - 1e-6)

        # 2) 缩放: [M, L, 3]
        scaled = normalized.unsqueeze(1) * self.resolutions_float.view(1, L, 1)

        # 3) 整数部分 (int32!) & 小数部分
        floor_coords = torch.floor(scaled).int()  # int32, 非 int64
        frac = scaled - floor_coords.float()

        # 4) 8 个角: [M, L, 8, 3] (int32)
        corners = floor_coords.unsqueeze(2) + self.corner_offsets.int().view(1, 1, 8, 3)

        # 5) 空间哈希: [M, L, 8] (int32)
        hash_idx = self._spatial_hash(corners)
        del corners  # 释放最大的中间张量

        # 6) 层级索引: [1, L, 1] (int32)
        level_idx = torch.arange(L, device=positions.device, dtype=torch.int32).view(1, L, 1).expand(M, L, 8)

        # 7) 查表: [M, L, 8, F]
        corner_feats = self.hash_tables[level_idx.long(), hash_idx.long()]
        del level_idx, hash_idx

        # 8) 三线性插值
        fx = frac[..., 0:1]
        fy = frac[..., 1:2]
        fz = frac[..., 2:3]
        del frac

        c00 = corner_feats[:, :, 0] * (1 - fz) + corner_feats[:, :, 1] * fz
        c01 = corner_feats[:, :, 2] * (1 - fz) + corner_feats[:, :, 3] * fz
        c10 = corner_feats[:, :, 4] * (1 - fz) + corner_feats[:, :, 5] * fz
        c11 = corner_feats[:, :, 6] * (1 - fz) + corner_feats[:, :, 7] * fz
        del corner_feats

        c0 = c00 * (1 - fy) + c01 * fy
        c1 = c10 * (1 - fy) + c11 * fy
        del c00, c01, c10, c11

        result = c0 * (1 - fx) + c1 * fx
        del c0, c1

        return result.reshape(M, -1)

    # ------------------------------------------------------------------
    def forward(self, positions: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
        """
        显存优化的前向传播: 分块处理 + 梯度检查点。

        策略:
          - 将 N 个锚点分成 chunk_size 大小的块
          - 每个块用 torch.utils.checkpoint 包裹 → 前向时不存储中间张量
          - 反向时逐块重算 → 峰值显存仅为 1 个块的大小

        显存对比 (N=37K, L=16, chunk=4096):
          旧方案: ~500 MB 中间张量 (全部同时在显存)
          新方案: ~ 30 MB 峰值 (仅 1 个 chunk)

        Args:
            positions:  [N, 3]  世界坐标
            chunk_size: 每块大小, 越小越省显存但越慢
        Returns:
            features: [N, L*F]
        """
        N = positions.shape[0]

        # 小数据: 直接计算, 不分块
        if N <= chunk_size:
            return self._forward_chunk(positions)

        if not torch.is_grad_enabled():
            # 推理模式: 仍然分块以控制峰值显存, 但不需要梯度检查点
            outputs = []
            for i in range(0, N, chunk_size):
                outputs.append(self._forward_chunk(positions[i:i + chunk_size]))
            return torch.cat(outputs, dim=0)

        # 训练模式: 分块 + 梯度检查点
        outputs = []
        for i in range(0, N, chunk_size):
            chunk_pos = positions[i:i + chunk_size]
            # grad_checkpoint: 前向不存中间张量, 反向时重算
            chunk_out = grad_checkpoint(
                self._forward_chunk, chunk_pos, use_reentrant=False
            )
            outputs.append(chunk_out)

        return torch.cat(outputs, dim=0)

    # ------------------------------------------------------------------
    def update_bbox(self, points: torch.Tensor, margin: float = 0.1):
        """
        根据点云更新 AABB 包围盒 (含 margin)。

        Args:
            points: [N, 3]
            margin: 包围盒外扩比例 (相对于范围)
        """
        pmin = points.min(dim=0)[0]
        pmax = points.max(dim=0)[0]
        extent = (pmax - pmin).clamp(min=1e-6)
        self.bbox_min.copy_(pmin - margin * extent)
        self.bbox_max.copy_(pmax + margin * extent)


# ================================================================
#  2. Context MLP  (Intra-anchor Context Decoding)
# ================================================================

class ContextMLP(nn.Module):
    """
    带有 Intra-anchor 上下文依赖的属性解码器。

    分两个阶段解码锚点属性:
      Stage 1 (几何阶段) : hash_features → 几何属性 + geometry_context
      Stage 2 (外观阶段) : hash_features + geometry_context + view_dir → 外观属性

    这种设计使得外观属性可以利用几何属性作为上下文,
    建立属性间的依赖关系 (Intra-anchor context)。
    """

    def __init__(
        self,
        hash_feat_dim: int = 32,
        anchor_feat_dim: int = 32,
        hidden_dim: int = 64,
        geometry_context_dim: int = 32,
        n_offsets: int = 5,
        appearance_dim: int = 32,
        add_opacity_dist: bool = False,
        add_cov_dist: bool = False,
        add_color_dist: bool = False,
    ):
        super().__init__()

        self.hash_feat_dim = hash_feat_dim
        self.anchor_feat_dim = anchor_feat_dim
        self.hidden_dim = hidden_dim
        self.geometry_context_dim = geometry_context_dim
        self.n_offsets = n_offsets
        self.appearance_dim = appearance_dim

        self.opacity_dist_dim = 1 if add_opacity_dist else 0
        self.cov_dist_dim = 1 if add_cov_dist else 0
        self.color_dist_dim = 1 if add_color_dist else 0

        # ============================================================
        #  Stage 1: 几何解码器 (视角无关)
        #    输入: hash_features
        #    输出: anchor_feat, offset, scaling, rotation, opacity
        #          + geometry_context (传递给 Stage 2)
        # ============================================================
        self.geometry_backbone = nn.Sequential(
            nn.Linear(hash_feat_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )

        # --- 各属性解码头 ---
        # anchor_feat: 重建后的锚点特征 (供 Stage 2 MLPs 使用)
        self.anchor_feat_head = nn.Linear(hidden_dim, anchor_feat_dim)

        # offset: [n_offsets * 3]
        self.offset_head = nn.Linear(hidden_dim, n_offsets * 3)

        # scaling: [6] (Scaffold-GS 使用 6 维 scaling)
        self.scaling_head = nn.Linear(hidden_dim, 6)

        # rotation: [4] (四元数)
        self.rotation_head = nn.Linear(hidden_dim, 4)

        # opacity: [1] (锚点级基础不透明度)
        self.opacity_head = nn.Linear(hidden_dim, 1)

        # geometry_context: 传递给外观阶段的上下文特征
        self.geometry_context_proj = nn.Sequential(
            nn.Linear(hidden_dim, geometry_context_dim),
            nn.ReLU(True),
        )

        # ============================================================
        #  Stage 2: 外观解码器 (视角相关)
        #    输入: anchor_feat + geometry_context + view_dir [+ dist] [+ appearance]
        #    输出: neural_opacity, color, cov (per neural gaussian)
        #
        #  这些解码器替代原始 Scaffold-GS 的 mlp_opacity/cov/color,
        #  但额外接收 geometry_context 作为 intra-anchor 上下文。
        # ============================================================

        # 基础特征维度 = anchor_feat_dim + geometry_context_dim + 3 (view_dir)
        base_dim = anchor_feat_dim + geometry_context_dim + 3

        # --- 不透明度解码器 ---
        self.opacity_decoder = nn.Sequential(
            nn.Linear(base_dim + self.opacity_dist_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, n_offsets),
            nn.Tanh(),
        )

        # --- 协方差解码器 (scale 3 + rotation 4 per offset) ---
        self.cov_decoder = nn.Sequential(
            nn.Linear(base_dim + self.cov_dist_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 7 * n_offsets),
        )

        # --- 颜色解码器 ---
        self.color_decoder = nn.Sequential(
            nn.Linear(base_dim + self.color_dist_dim + appearance_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 3 * n_offsets),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    #  Stage 1: 几何属性解码
    # ------------------------------------------------------------------
    def decode_geometry(
        self, hash_features: torch.Tensor
    ) -> dict:
        """
        从哈希特征解码视角无关的几何属性。

        Args:
            hash_features: [N, hash_feat_dim]

        Returns:
            dict 包含:
                anchor_feat:      [N, anchor_feat_dim]
                offsets:          [N, n_offsets * 3]
                scaling:          [N, 6]
                rotation:         [N, 4]
                opacity:          [N, 1]
                geometry_context: [N, geometry_context_dim]
        """
        backbone_feat = self.geometry_backbone(hash_features)

        return {
            'anchor_feat':      self.anchor_feat_head(backbone_feat),
            'offsets':          self.offset_head(backbone_feat),
            'scaling':          self.scaling_head(backbone_feat),
            'rotation':         self.rotation_head(backbone_feat),
            'opacity':          self.opacity_head(backbone_feat),
            'geometry_context': self.geometry_context_proj(backbone_feat),
        }

    # ------------------------------------------------------------------
    #  Stage 2: 外观属性解码 (视角相关)
    # ------------------------------------------------------------------
    def decode_appearance(
        self,
        anchor_feat: torch.Tensor,
        geometry_context: torch.Tensor,
        view_dir: torch.Tensor,
        ob_dist: torch.Tensor = None,
        appearance_embed: torch.Tensor = None,
    ) -> dict:
        """
        基于几何上下文 + 视角信息解码外观属性 (per neural gaussian)。

        Args:
            anchor_feat:      [N, anchor_feat_dim]  (来自 Stage 1)
            geometry_context: [N, geometry_context_dim]  (来自 Stage 1)
            view_dir:         [N, 3]  归一化视角方向
            ob_dist:          [N, 1]  距离 (可选)
            appearance_embed: [N, appearance_dim]  外观嵌入 (可选)

        Returns:
            dict 包含:
                neural_opacity: [N, n_offsets]
                color:          [N, n_offsets * 3]
                scale_rot:      [N, 7 * n_offsets]
        """
        base_input = torch.cat([anchor_feat, geometry_context, view_dir], dim=-1)

        # --- Opacity ---
        if ob_dist is not None and self.opacity_dist_dim > 0:
            opa_input = torch.cat([base_input, ob_dist], dim=-1)
        else:
            opa_input = base_input
        neural_opacity = self.opacity_decoder(opa_input)

        # --- Covariance ---
        if ob_dist is not None and self.cov_dist_dim > 0:
            cov_input = torch.cat([base_input, ob_dist], dim=-1)
        else:
            cov_input = base_input
        scale_rot = self.cov_decoder(cov_input)

        # --- Color ---
        color_input = base_input
        if ob_dist is not None and self.color_dist_dim > 0:
            color_input = torch.cat([color_input, ob_dist], dim=-1)
        if appearance_embed is not None:
            color_input = torch.cat([color_input, appearance_embed], dim=-1)
        color = self.color_decoder(color_input)

        return {
            'neural_opacity': neural_opacity,
            'color': color,
            'scale_rot': scale_rot,
        }

    # ------------------------------------------------------------------
    #  完整前向传播
    # ------------------------------------------------------------------
    def forward(
        self,
        hash_features: torch.Tensor,
        view_dir: torch.Tensor,
        ob_dist: torch.Tensor = None,
        appearance_embed: torch.Tensor = None,
    ) -> dict:
        """
        两阶段完整前向传播。

        Args:
            hash_features:    [N, hash_feat_dim]
            view_dir:         [N, 3]
            ob_dist:          [N, 1]  (可选)
            appearance_embed: [N, appearance_dim]  (可选)

        Returns:
            dict 包含 Stage 1 + Stage 2 所有输出
        """
        geom = self.decode_geometry(hash_features)
        appear = self.decode_appearance(
            anchor_feat=geom['anchor_feat'],
            geometry_context=geom['geometry_context'],
            view_dir=view_dir,
            ob_dist=ob_dist,
            appearance_embed=appearance_embed,
        )
        return {**geom, **appear}


# ================================================================
#  3. Adaptive Quantization Module  (AQM)
# ================================================================

class StraightThroughQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) 量化。

    前向: 舍入到最近的 step_size 倍数
    反向: 梯度直通 (∂quantize/∂x ≈ 1)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, step_size: torch.Tensor) -> torch.Tensor:
        return step_size * torch.round(x / step_size)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # 梯度直通: 对 x 返回原梯度, 对 step_size 不传梯度
        return grad_output, None


class AdaptiveQuantizationModule(nn.Module):
    """
    可微分自适应量化模块 (AQM)。

    对每个属性通道学习独立的量化步长 (step size):
      - 训练时: 使用 STE + 均匀噪声 模拟量化效果
      - 推理时: 硬量化 (round)

    Args:
        n_channels:     需要量化的通道数
        init_step_size: 初始步长
    """

    def __init__(self, n_channels: int, init_step_size: float = 0.01):
        super().__init__()
        # 在对数空间学习步长, 保证正数 + 数值稳定
        self.log_step_sizes = nn.Parameter(
            torch.full((n_channels,), math.log(init_step_size))
        )

    @property
    def step_sizes(self) -> torch.Tensor:
        """当前量化步长 [C]。"""
        return torch.exp(self.log_step_sizes)

    def forward(
        self, x: torch.Tensor, is_training: bool = True
    ) -> tuple:
        """
        对输入张量施加可微量化。

        Args:
            x:           [..., C]  待量化张量
            is_training: bool      是否为训练模式

        Returns:
            x_quantized: [..., C]  量化后张量
            step_sizes:  [C]       当前步长 (供熵估计使用)
        """
        step_sizes = self.step_sizes  # [C]

        if is_training:
            # 训练模式: 加均匀噪声 + STE 量化
            noise = (torch.rand_like(x) - 0.5) * step_sizes
            x_noisy = x + noise
            x_quantized = StraightThroughQuantize.apply(x_noisy, step_sizes)
        else:
            # 推理模式: 硬量化
            x_quantized = step_sizes * torch.round(x / step_sizes)

        return x_quantized, step_sizes


# ================================================================
#  4. Entropy Model  (参数化概率模型)
# ================================================================

class FactorizedEntropyModel(nn.Module):
    """
    因子化熵模型: 对每个通道建模独立的高斯分布,
    用于估计量化符号的比特率。

    概率模型:
        P(x_q) = CDF(x_q + step/2) - CDF(x_q - step/2)
    其中 CDF 是高斯累积分布函数, 参数 (mean, scale) 可学习。

    Args:
        n_channels: 通道数
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.means = nn.Parameter(torch.zeros(n_channels))
        self.log_scales = nn.Parameter(torch.zeros(n_channels))

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales).clamp(min=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        step_sizes: torch.Tensor = None,
    ) -> tuple:
        """
        估计比特率。

        Args:
            x:          [N, C]  (量化前或量化后的值)
            step_sizes: [C]     AQM 的量化步长

        Returns:
            total_rate:        标量, 总估计比特数
            rate_per_element:  [N, C]  逐元素比特估计
        """
        means = self.means.unsqueeze(0)    # [1, C]
        scales = self.scales.unsqueeze(0)  # [1, C]
        centered = x - means

        if step_sizes is not None:
            # 量化 bin 概率: P = Φ((x+Δ/2-μ)/σ) - Φ((x-Δ/2-μ)/σ)
            half_step = step_sizes.unsqueeze(0) * 0.5
            upper = (centered + half_step) / scales
            lower = (centered - half_step) / scales
            # 使用 erf 计算标准正态 CDF
            prob = 0.5 * (torch.erf(upper * 0.7071067811865476)
                          - torch.erf(lower * 0.7071067811865476))
        else:
            # 无步长时, 使用 PDF 近似
            prob = torch.exp(-0.5 * (centered / scales) ** 2) / (
                scales * math.sqrt(2 * math.pi)
            )

        # 数值安全
        prob = prob.clamp(min=1e-10)

        # 比特率: -log2(P)
        rate_per_element = -torch.log2(prob)
        total_rate = rate_per_element.sum()

        return total_rate, rate_per_element


# ================================================================
#  5. Rate-Distortion Loss
# ================================================================

class RateDistortionLoss(nn.Module):
    """
    率失真联合损失函数。

    L_total = L_distortion + λ_rate * L_rate + λ_scaling * L_scaling_reg

    其中:
      L_distortion = (1 - λ_ssim) * L1 + λ_ssim * (1 - SSIM)
      L_rate       = Σ 所有被量化属性的归一化熵估计
      L_scaling_reg = scaling 正则化

    Args:
        lambda_rate:    比特率损失权重
        lambda_ssim:    SSIM 损失权重
        lambda_scaling: scaling 正则化权重
    """

    def __init__(
        self,
        lambda_rate: float = 1e-4,
        lambda_ssim: float = 0.2,
        lambda_scaling: float = 0.01,
    ):
        super().__init__()
        self.lambda_rate = lambda_rate
        self.lambda_ssim = lambda_ssim
        self.lambda_scaling = lambda_scaling

    def forward(
        self,
        rendered_image: torch.Tensor,
        gt_image: torch.Tensor,
        rate_dict: dict,
        l1_loss_fn=None,
        ssim_fn=None,
        scaling_reg: torch.Tensor = None,
    ) -> tuple:
        """
        Args:
            rendered_image: [3, H, W]
            gt_image:       [3, H, W]
            rate_dict:      {属性名: (total_rate, n_elements)} 各属性的比特率
            l1_loss_fn:     L1 损失函数
            ssim_fn:        SSIM 函数 (返回相似度, 非损失)
            scaling_reg:    scaling 正则化值

        Returns:
            total_loss:  标量
            loss_dict:   各分量字典
        """
        # --- 失真损失 ---
        if l1_loss_fn is not None:
            l1 = l1_loss_fn(rendered_image, gt_image)
        else:
            l1 = F.l1_loss(rendered_image, gt_image)

        if ssim_fn is not None:
            ssim_val = ssim_fn(rendered_image, gt_image)
            ssim_loss = 1.0 - ssim_val
        else:
            ssim_loss = torch.tensor(0.0, device=rendered_image.device)

        distortion = (1.0 - self.lambda_ssim) * l1 + self.lambda_ssim * ssim_loss

        # --- 比特率损失 ---
        total_rate = torch.tensor(0.0, device=rendered_image.device)
        rate_breakdown = {}
        for name, rate_info in rate_dict.items():
            if isinstance(rate_info, tuple):
                rate_val, n_elem = rate_info
                # 归一化: 每个元素的平均比特率
                norm_rate = rate_val / max(n_elem, 1)
            else:
                norm_rate = rate_info
            total_rate = total_rate + norm_rate
            rate_breakdown[name] = norm_rate

        rate_loss = self.lambda_rate * total_rate

        # --- Scaling 正则化 ---
        reg = self.lambda_scaling * scaling_reg if scaling_reg is not None else 0.0

        total_loss = distortion + rate_loss + reg

        return total_loss, {
            'total': total_loss,
            'distortion': distortion,
            'l1': l1,
            'ssim_loss': ssim_loss,
            'rate_loss': rate_loss,
            'rate_breakdown': rate_breakdown,
            'scaling_reg': reg,
        }
