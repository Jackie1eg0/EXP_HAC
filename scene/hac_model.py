"""
HAC++ (Hash-grid Assisted Context) for 3D Gaussian Splatting Compression.

Key Idea:
    - The Hash Grid provides CONTEXT for entropy coding of anchor attributes.
    - Hash features f^h do NOT replace anchor features f^a for rendering.
    - f^h is input to MLPs to predict probability distributions of f^a for entropy coding.

Reference: "HAC++: Towards 100X Compression of 3D Gaussian Splatting"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
#  Straight-Through Estimator (STE) for Binary Quantization
# =====================================================================

class BinarizeSTE(torch.autograd.Function):
    """Binarize to {-1, +1} with straight-through gradient estimator."""
    @staticmethod
    def forward(ctx, input):
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Pass gradient through unchanged


def binarize_ste(x):
    """Binarize tensor to {-1, +1} using STE. Zeros become +1."""
    out = BinarizeSTE.apply(x)
    # sign(0) = 0, map to +1
    out = out + (out == 0).float()
    return out


# =====================================================================
#  3D Hash Encoding Level (Trilinear Interpolation)
# =====================================================================

class HashLevel3D(nn.Module):
    """Single resolution level of 3D multi-resolution hash encoding."""

    PRIMES = [1, 2654435761, 805459861]

    def __init__(self, table_size, feat_dim, resolution):
        super().__init__()
        self.table_size = table_size
        self.feat_dim = feat_dim
        self.resolution = resolution
        self.hash_table = nn.Parameter(torch.randn(table_size, feat_dim) * 0.01)

    def hash_fn(self, coords):
        """Instant-NGP style spatial hash function."""
        x = coords[..., 0].long()
        y = coords[..., 1].long()
        z = coords[..., 2].long()
        h = (x * self.PRIMES[0]) ^ (y * self.PRIMES[1]) ^ (z * self.PRIMES[2])
        return ((h % self.table_size) + self.table_size) % self.table_size

    def forward(self, positions, do_binarize=True):
        """
        Args:
            positions: [N, 3] normalised to [0, 1)
        Returns:
            [N, feat_dim] trilinearly interpolated features
        """
        scaled = positions * self.resolution
        floor_c = torch.floor(scaled).long()
        w = scaled - floor_c.float()  # fractional part [N, 3]

        # 8 corner offsets (000, 001, 010, 011, 100, 101, 110, 111)
        offsets = torch.tensor(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
             [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
            device=positions.device, dtype=torch.long,
        )
        corners = floor_c.unsqueeze(1) + offsets.unsqueeze(0)  # [N, 8, 3]
        indices = self.hash_fn(corners)  # [N, 8]

        table = binarize_ste(self.hash_table) if do_binarize else self.hash_table
        feats = table[indices]  # [N, 8, D]

        wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
        weights = torch.stack([
            (1 - wx) * (1 - wy) * (1 - wz),
            (1 - wx) * (1 - wy) * wz,
            (1 - wx) * wy * (1 - wz),
            (1 - wx) * wy * wz,
            wx * (1 - wy) * (1 - wz),
            wx * (1 - wy) * wz,
            wx * wy * (1 - wz),
            wx * wy * wz,
        ], dim=1)  # [N, 8]

        return (feats * weights.unsqueeze(-1)).sum(dim=1)  # [N, D]


# =====================================================================
#  2D Hash Encoding Level (Tri-Plane with Bilinear Interpolation)
# =====================================================================

class HashLevel2D(nn.Module):
    """Single resolution level of 2D tri-plane hash encoding (XY, XZ, YZ)."""

    PRIMES = [1, 2654435761]

    def __init__(self, table_size, feat_dim, resolution):
        super().__init__()
        self.table_size = table_size
        self.feat_dim = feat_dim
        self.resolution = resolution
        self.table_xy = nn.Parameter(torch.randn(table_size, feat_dim) * 0.01)
        self.table_xz = nn.Parameter(torch.randn(table_size, feat_dim) * 0.01)
        self.table_yz = nn.Parameter(torch.randn(table_size, feat_dim) * 0.01)

    def hash_fn_2d(self, coords):
        x = coords[..., 0].long()
        y = coords[..., 1].long()
        h = (x * self.PRIMES[0]) ^ (y * self.PRIMES[1])
        return ((h % self.table_size) + self.table_size) % self.table_size

    def _bilinear(self, pos2d, table, do_binarize):
        """Bilinear interpolation on a single 2D plane."""
        scaled = pos2d * self.resolution
        floor_c = torch.floor(scaled).long()
        w = scaled - floor_c.float()

        offsets = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]],
                               device=pos2d.device, dtype=torch.long)
        corners = floor_c.unsqueeze(1) + offsets.unsqueeze(0)  # [N, 4, 2]
        indices = self.hash_fn_2d(corners)  # [N, 4]

        tab = binarize_ste(table) if do_binarize else table
        feats = tab[indices]  # [N, 4, D]

        wx, wy = w[:, 0], w[:, 1]
        weights = torch.stack([
            (1 - wx) * (1 - wy), (1 - wx) * wy,
            wx * (1 - wy), wx * wy,
        ], dim=1)  # [N, 4]

        return (feats * weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, positions, do_binarize=True):
        """
        Args:
            positions: [N, 3] normalised to [0, 1)
        Returns:
            [N, feat_dim]  (sum of three planes)
        """
        f_xy = self._bilinear(positions[:, :2], self.table_xy, do_binarize)
        f_xz = self._bilinear(positions[:, [0, 2]], self.table_xz, do_binarize)
        f_yz = self._bilinear(positions[:, 1:3], self.table_yz, do_binarize)
        return f_xy + f_xz + f_yz


# =====================================================================
#  Binary Hash Grid (mixed 3D + 2D multi-resolution)
# =====================================================================

class BinaryHashGrid(nn.Module):
    """
    Mixed 3D-2D structured binary hash grid.
    3D: 12 levels, resolution 16-512, table 2^13, D^h=4
    2D: 4 levels, resolution 128-1024, table 2^15, D^h=4
    """

    def __init__(
        self,
        num_levels_3d: int = 12,
        num_levels_2d: int = 4,
        feat_dim: int = 4,
        min_res_3d: int = 16,
        max_res_3d: int = 512,
        min_res_2d: int = 128,
        max_res_2d: int = 1024,
        table_size_3d: int = 2 ** 13,
        table_size_2d: int = 2 ** 15,
    ):
        super().__init__()
        self.feat_dim = feat_dim

        def _resolutions(n, lo, hi):
            if n <= 1:
                return [lo]
            return [int(lo * (hi / lo) ** (i / (n - 1))) for i in range(n)]

        self.levels_3d = nn.ModuleList([
            HashLevel3D(table_size_3d, feat_dim, r)
            for r in _resolutions(num_levels_3d, min_res_3d, max_res_3d)
        ])
        self.levels_2d = nn.ModuleList([
            HashLevel2D(table_size_2d, feat_dim, r)
            for r in _resolutions(num_levels_2d, min_res_2d, max_res_2d)
        ])

        self.output_dim = (num_levels_3d + num_levels_2d) * feat_dim

        # Scene AABB (updated from anchor positions)
        self.register_buffer('bbox_min', torch.zeros(3))
        self.register_buffer('bbox_max', torch.ones(3))

    # ---- bbox helpers ----
    def update_bbox(self, points):
        """Set bounding box from current anchor positions (with margin)."""
        with torch.no_grad():
            pmin = points.min(dim=0)[0]
            pmax = points.max(dim=0)[0]
            extent = (pmax - pmin).max().clamp(min=1e-4) * 0.1
            self.bbox_min.copy_(pmin - extent)
            self.bbox_max.copy_(pmax + extent)

    def normalize(self, positions):
        return (positions - self.bbox_min) / (self.bbox_max - self.bbox_min + 1e-8)

    # ---- forward ----
    def forward(self, positions, do_binarize=True):
        """
        Args:
            positions: [N, 3] world coordinates
        Returns:
            [N, output_dim] concatenated hash features
        """
        norm = torch.clamp(self.normalize(positions), 0.0, 0.9999)
        parts = []
        for lv in self.levels_3d:
            parts.append(lv(norm, do_binarize))
        for lv in self.levels_2d:
            parts.append(lv(norm, do_binarize))
        return torch.cat(parts, dim=-1)

    # ---- hash grid entropy loss (Eq. 12) ----
    def get_binary_loss(self):
        """
        Total binary entropy of the hash grid (measures compressibility).
        Lower = more compressible.  Each raw value's sigmoid gives P(+1).
        """
        total = torch.tensor(0.0, device=self.bbox_min.device)
        tables = [lv.hash_table for lv in self.levels_3d]
        for lv in self.levels_2d:
            tables.extend([lv.table_xy, lv.table_xz, lv.table_yz])
        for t in tables:
            p = torch.sigmoid(t)
            ent = -p * torch.log2(p + 1e-10) - (1 - p) * torch.log2(1 - p + 1e-10)
            total = total + ent.sum()
        return total


# =====================================================================
#  HAC++ Context Model (MLP_q, MLP_c, MLP_a) + GMM Probability
# =====================================================================

class HACContextModel(nn.Module):
    """
    Context model for entropy coding of Scaffold-GS anchor attributes.

    Components
    ----------
    MLP_q  : hash feat -> adaptive quantisation refinement r
    MLP_c  : hash feat -> inter-anchor GMM params (mu, sigma, pi) per chunk
    MLP_a  : hash feat + prev chunks -> intra-anchor GMM params per chunk
    MLP_c_scaling / MLP_c_offset : inter-anchor context for l and o
    """

    # Base quantisation steps (Table I of HAC++)
    Q0_FEAT = 1.0
    Q0_SCALING = 0.001
    Q0_OFFSET = 0.2

    def __init__(
        self,
        hash_feat_dim: int,
        anchor_feat_dim: int = 32,
        scaling_dim: int = 6,
        n_offsets: int = 10,
        num_mixtures: int = 3,
        num_chunks: int = 4,
    ):
        super().__init__()
        self.anchor_feat_dim = anchor_feat_dim
        self.scaling_dim = scaling_dim
        self.offset_dim = n_offsets * 3
        self.n_offsets = n_offsets
        self.num_mixtures = num_mixtures
        self.num_chunks = num_chunks
        self.chunk_size = anchor_feat_dim // num_chunks

        assert anchor_feat_dim % num_chunks == 0, (
            f"feat_dim ({anchor_feat_dim}) must be divisible by "
            f"num_chunks ({num_chunks})"
        )

        K = num_mixtures
        cs = self.chunk_size

        # ---------- MLP_q: adaptive quantisation step ----------
        total_attr_dim = anchor_feat_dim + scaling_dim + self.offset_dim
        self.mlp_q = nn.Sequential(
            nn.Linear(hash_feat_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, total_attr_dim),
        )

        # ---------- MLP_c: inter-anchor context (per feature chunk) ----------
        self.mlp_c_feat = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hash_feat_dim, 128),
                nn.ReLU(True),
                nn.Linear(128, cs * K * 3),
            )
            for _ in range(num_chunks)
        ])

        # Inter-anchor for scaling and offset
        self.mlp_c_scaling = nn.Sequential(
            nn.Linear(hash_feat_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, scaling_dim * K * 3),
        )
        self.mlp_c_offset = nn.Sequential(
            nn.Linear(hash_feat_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, self.offset_dim * K * 3),
        )

        # ---------- MLP_a: intra-anchor context (autoregressive chunks) ----------
        self.mlp_a = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hash_feat_dim + i * cs, 128),
                nn.ReLU(True),
                nn.Linear(128, cs * K * 3),
            )
            for i in range(1, num_chunks)
        ])

        # Careful initialisation for smooth transition at milestone
        self._init_output_layers()

    # ----------------------------------------------------------------
    def _init_output_layers(self):
        """
        Initialise output layers so that:
        - MLP_q outputs ~0  =>  q = Q0*(1+tanh(0)) = Q0  (smooth start)
        - Context MLPs start with wide sigma  =>  low initial entropy loss
        """
        # MLP_q: zero init -> tanh(0) = 0 -> q = Q0
        nn.init.zeros_(self.mlp_q[-1].weight)
        nn.init.zeros_(self.mlp_q[-1].bias)

        # Context MLPs: small weights, sigma bias = 2 (softplus(2) ~ 2.13)
        context_mlps = (
            list(self.mlp_c_feat)
            + [self.mlp_c_scaling, self.mlp_c_offset]
            + list(self.mlp_a)
        )
        for mlp in context_mlps:
            last = mlp[-1]
            nn.init.xavier_uniform_(last.weight, gain=0.01)
            bias = last.bias.data
            bias.zero_()
            # Every 3rd element starting from index 1 is sigma_raw
            for j in range(1, bias.shape[0], 3):
                bias[j] = 2.0

    # ----------------------------------------------------------------
    #  Adaptive Quantisation Module (AQM)
    # ----------------------------------------------------------------
    def compute_quantization_step(self, hash_feats):
        """
        q_i = Q0 * (1 + tanh(r_i))   =>  q in (0, 2*Q0)
        """
        r = self.mlp_q(hash_feats)
        r_f = r[:, :self.anchor_feat_dim]
        r_s = r[:, self.anchor_feat_dim:self.anchor_feat_dim + self.scaling_dim]
        r_o = r[:, self.anchor_feat_dim + self.scaling_dim:]

        q_feat = self.Q0_FEAT * (1.0 + torch.tanh(r_f))
        q_scaling = self.Q0_SCALING * (1.0 + torch.tanh(r_s))
        q_offset = self.Q0_OFFSET * (1.0 + torch.tanh(r_o))
        return q_feat, q_scaling, q_offset

    @staticmethod
    def quantize(x, q, training=True):
        """
        Training : x + Uniform(-0.5, 0.5) * q   (differentiable proxy)
        Eval     : round(x / q) * q              (actual quantisation)
        """
        if training:
            noise = (torch.rand_like(x) - 0.5) * q
            return x + noise
        else:
            return torch.round(x / (q + 1e-10)) * q

    # ----------------------------------------------------------------
    #  GMM probability  (Eq. 8)
    # ----------------------------------------------------------------
    @staticmethod
    def _gmm_probability(x_hat, q, mu, sigma_raw, pi_logits):
        """
        p(x_hat) = sum_k w_k [Phi_upper - Phi_lower]

        Args:
            x_hat      : [N, D]
            q          : [N, D]
            mu         : [N, D, K]
            sigma_raw  : [N, D, K]  (before softplus)
            pi_logits  : [N, D, K]
        Returns:
            prob : [N, D]   probability per dimension
        """
        weights = F.softmax(pi_logits, dim=-1)       # [N, D, K]
        sigma = F.softplus(sigma_raw) + 1e-6          # positive

        x_e = x_hat.unsqueeze(-1)  # [N, D, 1]
        q_e = q.unsqueeze(-1)

        inv_sqrt2 = 0.7071067811865476
        upper = (x_e + 0.5 * q_e - mu) / sigma * inv_sqrt2
        lower = (x_e - 0.5 * q_e - mu) / sigma * inv_sqrt2

        cdf_up = 0.5 * (1.0 + torch.erf(upper))
        cdf_lo = 0.5 * (1.0 + torch.erf(lower))

        prob_k = torch.clamp(cdf_up - cdf_lo, min=1e-10)
        prob = (weights * prob_k).sum(dim=-1)          # [N, D]
        return torch.clamp(prob, min=1e-10)

    # helper: reshape raw MLP output -> (mu, sigma_raw, pi_logits)
    def _parse(self, raw, dim, K):
        N = raw.shape[0]
        t = raw.view(N, dim, K, 3)
        return t[..., 0], t[..., 1], t[..., 2]

    # ----------------------------------------------------------------
    #  Full forward: quantise + estimate bits
    # ----------------------------------------------------------------
    def forward(
        self,
        hash_feats,      # [N, H]
        anchor_feat,     # [N, F]
        scaling,         # [N, 6]
        offset,          # [N, n_offsets, 3]
        training=True,
        use_adaptive_q=True,
    ):
        """
        Returns
        -------
        bits_feat     : [N]  bits for anchor features
        bits_scaling  : [N]  bits for scaling
        bits_offset   : [N]  bits for offsets
        feat_hat      : [N, F]
        scaling_hat   : [N, 6]
        offset_hat    : [N, n_offsets, 3]
        """
        N = hash_feats.shape[0]
        K = self.num_mixtures
        offset_flat = offset.reshape(N, -1)            # [N, n_off*3]

        # --- quantisation steps ---
        if use_adaptive_q:
            q_f, q_s, q_o = self.compute_quantization_step(hash_feats)
        else:
            q_f = torch.full_like(anchor_feat, self.Q0_FEAT)
            q_s = torch.full_like(scaling, self.Q0_SCALING)
            q_o = torch.full_like(offset_flat, self.Q0_OFFSET)

        # --- quantise ---
        feat_hat = self.quantize(anchor_feat, q_f, training)
        scaling_hat = self.quantize(scaling, q_s, training)
        offset_hat_flat = self.quantize(offset_flat, q_o, training)

        # ===== Feature bits (intra-anchor autoregressive) =====
        bits_feat = torch.zeros(N, device=hash_feats.device)
        chunks = feat_hat.split(self.chunk_size, dim=1)
        q_chunks = q_f.split(self.chunk_size, dim=1)

        for i in range(self.num_chunks):
            chunk = chunks[i]
            q_c = q_chunks[i]

            # inter-anchor context
            inter_raw = self.mlp_c_feat[i](hash_feats)
            mu_i, sig_i, pi_i = self._parse(inter_raw, self.chunk_size, K)

            if i > 0:
                # intra-anchor: condition on previous quantised chunks
                prev = torch.cat(chunks[:i], dim=1)
                intra_in = torch.cat([hash_feats, prev], dim=1)
                intra_raw = self.mlp_a[i - 1](intra_in)
                mu_a, sig_a, pi_a = self._parse(intra_raw, self.chunk_size, K)

                # combine to 2K mixture components
                mu = torch.cat([mu_i, mu_a], dim=-1)
                sigma_raw = torch.cat([sig_i, sig_a], dim=-1)
                pi_logits = torch.cat([pi_i, pi_a], dim=-1)
            else:
                mu, sigma_raw, pi_logits = mu_i, sig_i, pi_i

            prob = self._gmm_probability(chunk, q_c, mu, sigma_raw, pi_logits)
            bits_feat = bits_feat + (-torch.log2(prob)).sum(dim=1)

        # ===== Scaling bits (inter-anchor only) =====
        sc_raw = self.mlp_c_scaling(hash_feats)
        mu_s, sig_s, pi_s = self._parse(sc_raw, self.scaling_dim, K)
        prob_s = self._gmm_probability(scaling_hat, q_s, mu_s, sig_s, pi_s)
        bits_scaling = (-torch.log2(prob_s)).sum(dim=1)

        # ===== Offset bits (inter-anchor only) =====
        off_raw = self.mlp_c_offset(hash_feats)
        mu_o, sig_o, pi_o = self._parse(off_raw, self.offset_dim, K)
        prob_o = self._gmm_probability(offset_hat_flat, q_o, mu_o, sig_o, pi_o)
        bits_offset = (-torch.log2(prob_o)).sum(dim=1)

        offset_hat = offset_hat_flat.view(N, self.n_offsets, 3)
        return bits_feat, bits_scaling, bits_offset, feat_hat, scaling_hat, offset_hat
