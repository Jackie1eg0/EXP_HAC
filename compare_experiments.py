"""
Scaffold-GS vs HAC++ 消融实验与对比评测脚本

功能:
  1. 按 llffhold=8 (每隔8张取一张测试) 划分训练集和测试集
  2. 自动运行 Scaffold-GS (基线) 和 HAC++ (多组 lambda_rate) 实验
  3. 收集所有实验的 PSNR / SSIM / LPIPS / Model Size
  4. 生成消融对比表格 + 柱状图 + 率失真曲线
  5. 自动输出消融结论: HAC++ 在保证压缩率的同时保持与 Scaffold-GS 相近的性能

用法:
  # 完整训练+评测+绘图
  python compare_experiments.py --scene flowers --data_root data/mipnerf360 --gpu 0

  # 只收集已有结果并生成图表
  python compare_experiments.py --scene flowers --data_root data/mipnerf360 --collect_only

  # 自定义 lambda 做消融
  python compare_experiments.py --scene flowers --data_root data/mipnerf360 \\
      --lambda_rates 1e-4 5e-4 1e-3 --gpu 0
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from pathlib import Path
from collections import OrderedDict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ====================================================================
#  配置: 需要运行的实验列表
# ====================================================================

def build_split_configs(split_mode, llffhold_values, lod_values):
    """构建可枚举的数据划分配置。"""
    split_cfgs = []
    if split_mode == "llffhold":
        vals = llffhold_values if llffhold_values else [8]
        for v in vals:
            tag = f"llffhold{v}"
            # 保持兼容: 默认单个 llffhold=8 时沿用旧目录结构
            if len(vals) == 1 and v == 8:
                tag = ""
            split_cfgs.append({
                "tag": tag,
                "name": f"llffhold={v}",
                "extra_args": ["--llffhold", str(v), "--lod", "0"],
            })
    elif split_mode == "lod":
        vals = lod_values if lod_values else [0]
        for v in vals:
            tag = f"lod{v}"
            # 保持兼容: 默认单个 lod=0 时沿用旧目录结构
            if len(vals) == 1 and v == 0:
                tag = ""
            split_cfgs.append({
                "tag": tag,
                "name": f"lod={v}",
                "extra_args": ["--lod", str(v)],
            })
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")
    return split_cfgs


def get_experiments(scene_name, data_root, output_root="outputs", split_cfgs=None, lambda_rates=None,
                    voxel_size=0.001, resolution=-1):
    """
    定义所有需要运行和对比的实验配置。

    返回 list[dict], 每个 dict 包含:
      - name:  实验名称 (用于显示和目录名)
      - cmd:   完整的训练命令参数列表
      - model_path: 输出目录
    """
    data_path = f"{data_root}/{scene_name}"
    base_args = [
        "--eval",
        "-s", data_path,
        "--voxel_size", str(voxel_size),
        "--update_init_factor", "16",
        "--appearance_dim", "0",
        "--ratio", "1",
        "--iterations", "30000",
    ]
    if resolution > 0:
        base_args += ["-r", str(resolution)]

    experiments = []
    split_cfgs = split_cfgs if split_cfgs is not None else [dict(tag="", name="llffhold=8", extra_args=["--llffhold", "8", "--lod", "0"])]
    lambda_rates = lambda_rates if lambda_rates is not None else [1e-4, 5e-4, 1e-3]

    for split in split_cfgs:
        split_prefix = f"{output_root}/{scene_name}"
        if split["tag"]:
            split_prefix = f"{split_prefix}/{split['tag']}"
        split_name = f"[{split['name']}] "

        # ---- 1. 原版 Scaffold-GS (基线) ----
        scaffold_path = f"{split_prefix}/scaffold_gs"
        experiments.append({
            "name": f"{split_name}Scaffold-GS (Baseline)",
            "model_path": scaffold_path,
            "cmd": ["python", "train.py"] + base_args + split["extra_args"] + [
                "-m", scaffold_path,
                "--port", "12340",
            ],
        })

        # ---- 2. HAC++ 不同 lambda_rate (率失真曲线) ----
        for lr in lambda_rates:
            lr_str = f"{lr:.0e}" if lr > 0 else "0"
            hac_path = f"{split_prefix}/hac_lr{lr_str}"
            experiments.append({
                "name": f"{split_name}HAC++ (λ_rate={lr_str})",
                "model_path": hac_path,
                "cmd": ["python", "train.py"] + base_args + split["extra_args"] + [
                    "-m", hac_path,
                    "--port", "12341",
                    "--use_hash_grid",
                    "--lambda_rate", str(lr),
                ],
            })

    return experiments


# ====================================================================
#  模型大小统计
# ====================================================================

def get_model_size_mb(model_path: str) -> dict:
    """
    统计模型的各部分存储大小 (MB)。

    Returns:
        dict: {
            'ply_size_mb':     ply 文件大小,
            'mlp_size_mb':     网络权重文件大小,
            'total_size_mb':   总大小,
            'num_anchors':     锚点数量 (从 ply 估算),
        }
    """
    result = {
        'ply_size_mb': 0.0,
        'mlp_size_mb': 0.0,
        'total_size_mb': 0.0,
        'num_anchors': 0,
    }

    # 查找最新 iteration 的 ply
    pc_dir = Path(model_path) / "point_cloud"
    if pc_dir.exists():
        iter_dirs = sorted(pc_dir.iterdir(), key=lambda x: x.name)
        if iter_dirs:
            latest = iter_dirs[-1]
            ply_file = latest / "point_cloud.ply"
            if ply_file.exists():
                size_bytes = ply_file.stat().st_size
                result['ply_size_mb'] = size_bytes / (1024 * 1024)
                # 粗略估算锚点数: 排除 header 后每个锚点约 (3+3+15+32+1+6+4)*4 bytes
                # 实际以 ply header 中的 element vertex count 为准
                try:
                    with open(ply_file, 'rb') as f:
                        for line in f:
                            line_str = line.decode('ascii', errors='ignore').strip()
                            if line_str.startswith('element vertex'):
                                result['num_anchors'] = int(line_str.split()[-1])
                                break
                            if line_str == 'end_header':
                                break
                except Exception:
                    pass

    # 查找网络权重文件
    mlp_total = 0.0
    model_dir = Path(model_path)
    weight_patterns = ['*.pt', '*.pth', 'checkpoints.pth', 'hac_checkpoints.pth']
    for pattern in weight_patterns:
        for f in model_dir.rglob(pattern):
            # 排除 point_cloud 目录和 chkpnt 文件
            if 'point_cloud' not in str(f) and 'chkpnt' not in str(f):
                mlp_total += f.stat().st_size / (1024 * 1024)

    result['mlp_size_mb'] = mlp_total
    result['total_size_mb'] = result['ply_size_mb'] + result['mlp_size_mb']
    return result


# ====================================================================
#  结果收集
# ====================================================================

def collect_results(model_path: str) -> dict:
    """
    从训练输出目录收集质量指标和模型大小。

    Returns:
        dict: {
            'psnr': float, 'ssim': float, 'lpips': float,
            'ply_size_mb': float, 'mlp_size_mb': float, 'total_size_mb': float,
            'num_anchors': int, 'fps': float,
        }
    """
    result = {'psnr': None, 'ssim': None, 'lpips': None, 'fps': None}

    # 读取 results.json
    # train.py evaluate() 写出的格式: { "ours_30000": { "PSNR": xx, "SSIM": xx, "LPIPS": xx } }
    results_file = Path(model_path) / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        for method_key in data:
            metrics = data[method_key]
            if isinstance(metrics, dict):
                result['psnr'] = metrics.get('PSNR')
                result['ssim'] = metrics.get('SSIM')
                result['lpips'] = metrics.get('LPIPS')
                break

    # 读取输出日志获取 FPS
    log_file = Path(model_path) / "outputs.log"
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if 'Test FPS' in line:
                        # 提取 FPS 数值
                        parts = line.split('Test FPS:')
                        if len(parts) > 1:
                            fps_str = parts[1].strip().replace('\033[1;35m', '').replace('\033[0m', '')
                            result['fps'] = float(fps_str)
        except Exception:
            pass

    # 模型大小
    size_info = get_model_size_mb(model_path)
    result.update(size_info)

    return result


# ====================================================================
#  结果展示
# ====================================================================

def print_comparison_table(all_results: list):
    """打印对比表格。"""

    print("\n" + "=" * 120)
    print("  Scaffold-GS vs HAC++  对比结果")
    print("=" * 120)

    header = f"{'实验名称':<30} {'PSNR↑':>8} {'SSIM↑':>8} {'LPIPS↓':>8} {'Size(MB)':>10} {'PLY(MB)':>10} {'MLP(MB)':>10} {'#Anchors':>10} {'FPS':>8}"
    print(header)
    print("-" * 120)

    baseline_psnr = None
    baseline_size = None

    for exp_name, result in all_results:
        psnr = f"{result['psnr']:.2f}" if result['psnr'] is not None else "N/A"
        ssim = f"{result['ssim']:.4f}" if result['ssim'] is not None else "N/A"
        lpips = f"{result['lpips']:.4f}" if result['lpips'] is not None else "N/A"
        total = f"{result['total_size_mb']:.2f}" if result['total_size_mb'] else "N/A"
        ply = f"{result['ply_size_mb']:.2f}" if result['ply_size_mb'] else "N/A"
        mlp = f"{result['mlp_size_mb']:.2f}" if result['mlp_size_mb'] else "N/A"
        anchors = f"{result['num_anchors']:,}" if result['num_anchors'] else "N/A"
        fps = f"{result['fps']:.1f}" if result['fps'] is not None else "N/A"

        row = f"{exp_name:<30} {psnr:>8} {ssim:>8} {lpips:>8} {total:>10} {ply:>10} {mlp:>10} {anchors:>10} {fps:>8}"
        print(row)

        # 记录基线用于计算压缩比
        if 'Baseline' in exp_name:
            baseline_psnr = result['psnr']
            baseline_size = result['total_size_mb']

    print("-" * 120)

    # 打印压缩比
    if baseline_size and baseline_size > 0:
        print(f"\n{'实验名称':<30} {'PSNR差值':>10} {'压缩比':>10} {'大小节省':>10}")
        print("-" * 65)
        for exp_name, result in all_results:
            if result['total_size_mb'] and result['total_size_mb'] > 0:
                ratio = baseline_size / result['total_size_mb']
                saving = (1 - result['total_size_mb'] / baseline_size) * 100
                psnr_diff = (result['psnr'] - baseline_psnr) if (result['psnr'] and baseline_psnr) else 0
                sign = "+" if psnr_diff >= 0 else ""
                print(f"{exp_name:<30} {sign}{psnr_diff:>8.2f}dB {ratio:>9.2f}x {saving:>9.1f}%")
        print()


def export_rd_curve_data(all_results: list, output_file: str):
    """导出率失真曲线数据为 JSON，方便后续绘图。"""
    rd_data = []
    for exp_name, result in all_results:
        rd_data.append({
            'name': exp_name,
            'psnr': result.get('psnr'),
            'ssim': result.get('ssim'),
            'lpips': result.get('lpips'),
            'size_mb': result.get('total_size_mb'),
            'num_anchors': result.get('num_anchors'),
            'fps': result.get('fps'),
        })
    with open(output_file, 'w') as f:
        json.dump(rd_data, f, indent=2, ensure_ascii=False)
    print(f"率失真曲线数据已保存到: {output_file}")


# ====================================================================
#  消融实验: 绘图模块
# ====================================================================

def plot_ablation_comparison(all_results: list, output_dir: str, scene_name: str):
    """
    生成消融实验的核心对比图:
      Figure 1: PSNR / SSIM 分组柱状图 (左轴 PSNR, 右轴 SSIM)
      Figure 2: 模型大小柱状图 + 压缩比标注
      Figure 3: 率失真散点图 (Size vs PSNR)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- 过滤有效结果 ----
    valid = [(n, r) for n, r in all_results if r['psnr'] is not None]
    if not valid:
        print("  [WARNING] 没有有效的实验结果, 跳过绘图")
        return

    names   = [n.split("] ")[-1] if "] " in n else n for n, _ in valid]
    psnrs   = [r['psnr']  for _, r in valid]
    ssims   = [r['ssim']  for _, r in valid]
    lpipss  = [r['lpips'] for _, r in valid]
    sizes   = [r['total_size_mb'] for _, r in valid]

    # 颜色: Scaffold-GS 红色, HAC++ 蓝色系
    colors = []
    for n, _ in valid:
        if 'Baseline' in n or 'Scaffold' in n:
            colors.append('#E53935')
        else:
            colors.append('#1E88E5')

    x = np.arange(len(names))
    bar_w = 0.55

    # ======== Figure 1: PSNR + SSIM 对比 ========
    fig, ax1 = plt.subplots(figsize=(max(8, 2.2 * len(names)), 6))
    bars = ax1.bar(x, psnrs, bar_w, color=colors, edgecolor='white',
                   linewidth=0.8, alpha=0.9, zorder=3)
    ax1.set_ylabel('PSNR (dB) ↑', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=18, ha='right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3, zorder=0)

    # 在柱子上标数值
    for bar, p in zip(bars, psnrs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                 f'{p:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 右轴: SSIM
    ax2 = ax1.twinx()
    ax2.plot(x, ssims, 's--', color='#43A047', markersize=8,
             linewidth=2, label='SSIM', zorder=5)
    for i, s in enumerate(ssims):
        ax2.annotate(f'{s:.4f}', (x[i], s),
                     textcoords='offset points', xytext=(0, 10),
                     fontsize=8, color='#2E7D32', ha='center')
    ax2.set_ylabel('SSIM ↑', fontsize=12, color='#43A047', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#43A047')

    fig.suptitle(f'Ablation: HAC++ vs Scaffold-GS  –  {scene_name}\n'
                 f'(Train/Test split: every 8th image as test)',
                 fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(output_dir, 'ablation_psnr_ssim.png'), dpi=200)
    fig.savefig(os.path.join(output_dir, 'ablation_psnr_ssim.pdf'))
    plt.close(fig)
    print(f"  Saved: ablation_psnr_ssim.png/.pdf")

    # ======== Figure 2: 模型大小 + 压缩比 ========
    # 找到 baseline 大小
    baseline_size = None
    for n, r in valid:
        if 'Baseline' in n or 'Scaffold' in n:
            baseline_size = r['total_size_mb']
            break

    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(names)), 6))
    bars = ax.bar(x, sizes, bar_w, color=colors, edgecolor='white',
                  linewidth=0.8, alpha=0.9, zorder=3)

    for bar, sz in zip(bars, sizes):
        label = f'{sz:.1f} MB'
        if baseline_size and baseline_size > 0 and sz < baseline_size * 0.99:
            ratio = baseline_size / sz
            label += f'\n({ratio:.1f}x)'
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(sizes) * 0.01,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Model Size (MB) ↓', fontsize=12, fontweight='bold')
    ax.set_xlabel('Method', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_title(f'Storage Comparison  –  {scene_name}', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ablation_size.png'), dpi=200)
    fig.savefig(os.path.join(output_dir, 'ablation_size.pdf'))
    plt.close(fig)
    print(f"  Saved: ablation_size.png/.pdf")

    # ======== Figure 3: 率失真散点图 (Size vs PSNR) ========
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (n, r) in enumerate(valid):
        is_baseline = ('Baseline' in n or 'Scaffold' in n)
        marker = '*' if is_baseline else 'o'
        ms = 250 if is_baseline else 120
        c = '#E53935' if is_baseline else '#1E88E5'
        label = names[i]
        ax.scatter(r['total_size_mb'], r['psnr'], s=ms, c=c,
                   marker=marker, edgecolors='black', linewidths=0.8,
                   zorder=5, label=label)
        ax.annotate(names[i],
                    (r['total_size_mb'], r['psnr']),
                    textcoords='offset points',
                    xytext=(10, 5) if is_baseline else (10, -12),
                    fontsize=8, alpha=0.85)

    # HAC++ 连线 (仅连 HAC 的点)
    hac_pts = [(r['total_size_mb'], r['psnr'])
               for n, r in valid if 'HAC' in n]
    if len(hac_pts) > 1:
        hac_pts.sort()
        hx, hy = zip(*hac_pts)
        ax.plot(hx, hy, '--', color='#1E88E5', linewidth=1.5, alpha=0.6, zorder=3)

    ax.set_xlabel('Model Size (MB) →', fontsize=12)
    ax.set_ylabel('PSNR (dB) ↑', fontsize=12)
    ax.set_title(f'Rate-Distortion Curve  –  {scene_name}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ablation_rd_curve.png'), dpi=200)
    fig.savefig(os.path.join(output_dir, 'ablation_rd_curve.pdf'))
    plt.close(fig)
    print(f"  Saved: ablation_rd_curve.png/.pdf")

    # ======== Figure 4: 综合对比雷达图 (PSNR/SSIM/LPIPS/Size 归一化) ========
    if len(valid) >= 2 and all(r['lpips'] is not None for _, r in valid):
        fig, ax = plt.subplots(figsize=(10, 5))
        # 归一化到 [0, 1]: PSNR/SSIM 越高越好, LPIPS/Size 越低越好
        p_min, p_max = min(psnrs), max(psnrs)
        s_min, s_max = min(ssims), max(ssims)
        l_min, l_max = min(lpipss), max(lpipss)
        z_min, z_max = min(sizes), max(sizes)

        metrics_labels = ['PSNR ↑', 'SSIM ↑', '1-LPIPS ↑', 'Compression ↑']
        table_data = []
        for i, (n, r) in enumerate(valid):
            p_norm = (r['psnr'] - p_min) / (p_max - p_min + 1e-8)
            s_norm = (r['ssim'] - s_min) / (s_max - s_min + 1e-8)
            l_norm = 1.0 - (r['lpips'] - l_min) / (l_max - l_min + 1e-8)  # 反转
            z_norm = 1.0 - (r['total_size_mb'] - z_min) / (z_max - z_min + 1e-8)  # 反转
            table_data.append([f'{r["psnr"]:.2f}', f'{r["ssim"]:.4f}',
                               f'{r["lpips"]:.4f}', f'{r["total_size_mb"]:.1f} MB'])

        # 用表格 + 文字形式展示
        ax.axis('off')
        col_labels = ['Method', 'PSNR (dB)↑', 'SSIM↑', 'LPIPS↓', 'Size (MB)↓']
        cell_text = []
        cell_colors = []
        for i, (n, r) in enumerate(valid):
            row = [names[i]] + table_data[i]
            cell_text.append(row)
            if 'Baseline' in n or 'Scaffold' in n:
                cell_colors.append(['#FFCDD2'] * 5)
            else:
                cell_colors.append(['#BBDEFB'] * 5)

        table = ax.table(cellText=cell_text, colLabels=col_labels,
                         cellColours=cell_colors, cellLoc='center',
                         loc='center', colColours=['#E0E0E0'] * 5)
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 1.8)

        ax.set_title(f'Ablation Summary  –  {scene_name}\n'
                     f'Data split: llffhold=8 (every 8th image as test)',
                     fontsize=13, fontweight='bold', pad=20)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'ablation_summary_table.png'), dpi=200)
        fig.savefig(os.path.join(output_dir, 'ablation_summary_table.pdf'))
        plt.close(fig)
        print(f"  Saved: ablation_summary_table.png/.pdf")


def print_ablation_conclusion(all_results: list, scene_name: str, output_dir: str):
    """
    自动生成消融实验结论: 证明 HAC++ 在大幅压缩的同时保持与 Scaffold-GS 相近的质量。
    """
    valid = [(n, r) for n, r in all_results if r['psnr'] is not None]
    if not valid:
        return

    # 找到 Scaffold-GS baseline
    baseline = None
    for n, r in valid:
        if 'Baseline' in n or 'Scaffold' in n:
            baseline = (n, r)
            break

    if baseline is None:
        print("  [WARNING] 未找到 Scaffold-GS 基线, 无法输出消融结论")
        return

    bl_name, bl = baseline
    hac_results = [(n, r) for n, r in valid if 'HAC' in n]
    if not hac_results:
        print("  [WARNING] 未找到 HAC++ 结果, 无法输出消融结论")
        return

    lines = []
    sep = "=" * 80
    lines.append("")
    lines.append(sep)
    lines.append("  消融实验结论 (Ablation Conclusion)")
    lines.append(sep)
    lines.append(f"  场景: {scene_name}")
    lines.append(f"  数据划分: llffhold=8 (每隔 8 张图片取 1 张作为测试集)")
    lines.append(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"  基线 Scaffold-GS:")
    lines.append(f"    PSNR  = {bl['psnr']:.2f} dB")
    lines.append(f"    SSIM  = {bl['ssim']:.4f}")
    lines.append(f"    LPIPS = {bl['lpips']:.4f}" if bl['lpips'] else "    LPIPS = N/A")
    lines.append(f"    Size  = {bl['total_size_mb']:.2f} MB")
    lines.append("")
    lines.append("-" * 80)
    lines.append(f"  {'HAC++ Config':<28s} {'ΔPSNR(dB)':>10s} {'ΔSSIM':>10s}"
                 f" {'Size(MB)':>10s} {'压缩比':>8s} {'结论':>10s}")
    lines.append("-" * 80)

    best_hac = None
    best_score = -999

    for n, r in hac_results:
        short_name = n.split("] ")[-1] if "] " in n else n
        dpsnr = r['psnr'] - bl['psnr']
        dssim = r['ssim'] - bl['ssim'] if (r['ssim'] and bl['ssim']) else 0
        ratio = bl['total_size_mb'] / r['total_size_mb'] if r['total_size_mb'] > 0 else 0

        # 判定: PSNR 差在 1dB 内视为"相近"
        if abs(dpsnr) <= 1.0:
            verdict = "✓ 相近"
        elif dpsnr > 0:
            verdict = "✓ 更优"
        else:
            verdict = "△ 有差距"

        sign_p = "+" if dpsnr >= 0 else ""
        sign_s = "+" if dssim >= 0 else ""
        lines.append(f"  {short_name:<28s} {sign_p}{dpsnr:>8.2f} {sign_s}{dssim:>9.4f}"
                     f" {r['total_size_mb']:>10.2f} {ratio:>7.1f}x {verdict:>10s}")

        # 挑选"最优 HAC++": 压缩比最高且 PSNR 差 <1dB
        score = ratio if abs(dpsnr) <= 1.0 else ratio * 0.1
        if score > best_score:
            best_score = score
            best_hac = (short_name, r, dpsnr, ratio)

    lines.append("-" * 80)

    if best_hac:
        bname, br, bdp, bratio = best_hac
        lines.append("")
        lines.append(f"  >>> 最佳消融点: {bname}")
        lines.append(f"      压缩比: {bratio:.1f}x (模型大小从 {bl['total_size_mb']:.1f}MB "
                     f"→ {br['total_size_mb']:.1f}MB)")
        sign = "+" if bdp >= 0 else ""
        lines.append(f"      PSNR 变化: {sign}{bdp:.2f} dB (基线 {bl['psnr']:.2f} → {br['psnr']:.2f})")
        lines.append(f"      SSIM: {br['ssim']:.4f} vs 基线 {bl['ssim']:.4f}")
        lines.append("")
        if abs(bdp) <= 0.5:
            lines.append(f"  结论: HAC++ 实现了 {bratio:.1f}x 压缩, PSNR 几乎无损 ({sign}{bdp:.2f}dB),")
            lines.append(f"        证明 HAC++ 能在保证高压缩率的同时维持与 Scaffold-GS 相当的渲染质量。")
        elif abs(bdp) <= 1.0:
            lines.append(f"  结论: HAC++ 实现了 {bratio:.1f}x 压缩, PSNR 仅下降 {abs(bdp):.2f}dB,")
            lines.append(f"        在可接受范围内, 验证了 HAC++ 的压缩有效性。")
        else:
            lines.append(f"  结论: HAC++ 实现了 {bratio:.1f}x 压缩, 但 PSNR 下降 {abs(bdp):.2f}dB,")
            lines.append(f"        建议尝试更小的 lambda_rate 以获得更好的质量-压缩平衡。")

    lines.append("")
    lines.append(sep)

    conclusion_text = "\n".join(lines)
    print(conclusion_text)

    # 保存到文件
    conclusion_file = os.path.join(output_dir, "ablation_conclusion.txt")
    with open(conclusion_file, 'w', encoding='utf-8') as f:
        f.write(conclusion_text)
    print(f"  结论已保存到: {conclusion_file}")


# ====================================================================
#  主流程
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Scaffold-GS vs HAC++ 对比实验")
    parser.add_argument("--scene", type=str, required=True, help="场景名称 (如 flowers)")
    parser.add_argument("--data_root", type=str, default="data/mipnerf360", help="数据集根目录")
    parser.add_argument("--output_root", type=str, default="outputs", help="实验输出根目录")
    parser.add_argument("--gpu", type=str, default="0", help="GPU 编号")
    parser.add_argument("--split_mode", type=str, default="llffhold", choices=["llffhold", "lod"],
                        help="数据划分模式: llffhold 或 lod")
    parser.add_argument("--llffhold_values", nargs="+", type=int, default=[8],
                        help="当 split_mode=llffhold 时使用, 可传多个值做消融")
    parser.add_argument("--lod_values", nargs="+", type=int, default=[0],
                        help="当 split_mode=lod 时使用, 可传多个值做消融")
    parser.add_argument("--lambda_rates", nargs="+", type=float,
                        default=[1e-4, 5e-4, 1e-3],
                        help="HAC++ lambda_rate 列表 (消融实验: 不同压缩强度)")
    parser.add_argument("--voxel_size", type=float, default=0.001,
                        help="体素大小 (越大锚点越少, 越省显存; 默认 0.001)")
    parser.add_argument("--resolution", type=int, default=-1,
                        help="图像分辨率缩放 (1=原始, 2=半分辨率, 4=1/4; 默认 -1 自动)")
    parser.add_argument("--collect_only", action="store_true", help="仅收集已有结果,不重新训练")
    parser.add_argument("--skip_baseline", action="store_true", help="跳过 Scaffold-GS 基线训练")
    args = parser.parse_args()

    split_cfgs = build_split_configs(args.split_mode, args.llffhold_values, args.lod_values)
    experiments = get_experiments(
        args.scene,
        args.data_root,
        args.output_root,
        split_cfgs=split_cfgs,
        lambda_rates=args.lambda_rates,
        voxel_size=args.voxel_size,
        resolution=args.resolution,
    )

    # ---- 打印实验配置与数据划分信息 ----
    print("\n" + "=" * 80)
    print("  消融实验配置")
    print("=" * 80)
    print(f"  场景:         {args.scene}")
    print(f"  数据集路径:   {args.data_root}/{args.scene}")
    print(f"  划分模式:     {args.split_mode}")
    if args.split_mode == "llffhold":
        print(f"  llffhold值:   {args.llffhold_values}")
        print(f"  划分说明:     每隔 {args.llffhold_values[0]} 张图片取 1 张作为测试集")
        print(f"                训练集 = idx % {args.llffhold_values[0]} != 0")
        print(f"                测试集 = idx % {args.llffhold_values[0]} == 0")
    else:
        print(f"  lod值:        {args.lod_values}")
    print(f"  HAC++ λ值:    {args.lambda_rates}")
    print(f"  实验总数:     {len(experiments)}")
    for i, exp in enumerate(experiments):
        tag = "Baseline" if 'Baseline' in exp['name'] else "HAC++"
        print(f"    [{i+1}] {tag:<10s} {exp['name']}")
    print("=" * 80)

    # ---- 运行实验 ----
    if not args.collect_only:
        for i, exp in enumerate(experiments):
            if args.skip_baseline and 'Baseline' in exp['name']:
                print(f"\n[跳过] {exp['name']}")
                continue

            print(f"\n{'='*80}")
            print(f"  [{i+1}/{len(experiments)}] 运行实验: {exp['name']}")
            print(f"  输出目录: {exp['model_path']}")
            print(f"{'='*80}\n")

            # 检查是否已有结果
            if Path(exp['model_path'], 'results.json').exists():
                print(f"  [已存在结果, 跳过训练] {exp['model_path']}/results.json")
                continue

            cmd = exp['cmd'] + ["--gpu", args.gpu]
            print(f"  命令: {' '.join(cmd)}\n")

            # 设置 CUDA 显存分配优化, 减少碎片化导致的 OOM
            env = os.environ.copy()
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

            try:
                subprocess.run(cmd, check=True, env=env)
            except subprocess.CalledProcessError as e:
                print(f"  [错误] 实验失败: {e}")
                continue
            except KeyboardInterrupt:
                print("\n[中断] 用户终止实验")
                break

    # ---- 收集结果 ----
    print(f"\n{'='*80}")
    print("  收集所有实验结果...")
    print(f"{'='*80}")

    all_results = []
    for exp in experiments:
        result = collect_results(exp['model_path'])
        all_results.append((exp['name'], result))
        status = "OK" if result['psnr'] is not None else "未找到"
        print(f"  {exp['name']:<30} -> {status}")

    # ---- 展示对比 ----
    print_comparison_table(all_results)

    # ---- 导出率失真曲线数据 ----
    plot_dir = f"{args.output_root}/{args.scene}/ablation_plots"
    os.makedirs(plot_dir, exist_ok=True)
    rd_file = os.path.join(plot_dir, f"rd_curve_data_{args.split_mode}.json")
    export_rd_curve_data(all_results, rd_file)

    # ---- 生成消融对比图 ----
    print("\n" + "=" * 80)
    print("  生成消融对比图表...")
    print("=" * 80)
    plot_ablation_comparison(all_results, plot_dir, args.scene)

    # ---- 输出消融实验结论 ----
    print_ablation_conclusion(all_results, args.scene, plot_dir)

    # ---- 汇总提示 ----
    print("\n" + "=" * 80)
    print(f"  所有结果已保存到: {plot_dir}/")
    print("  包含:")
    print(f"    - ablation_psnr_ssim.png/pdf    PSNR+SSIM 对比柱状图")
    print(f"    - ablation_size.png/pdf          模型大小对比 + 压缩比")
    print(f"    - ablation_rd_curve.png/pdf      率失真散点图")
    print(f"    - ablation_summary_table.png/pdf 汇总表格")
    print(f"    - ablation_conclusion.txt        消融结论")
    print(f"    - {os.path.basename(rd_file)}  原始数据 (JSON)")
    print("=" * 80)


if __name__ == "__main__":
    main()
