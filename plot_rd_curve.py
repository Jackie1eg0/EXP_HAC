"""
绘制率失真 (Rate-Distortion) 曲线。

用法:
  python plot_rd_curve.py --input outputs/flowers/rd_curve_data.json --output rd_curve.png
"""

import json
import argparse
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端
import matplotlib.pyplot as plt
import numpy as np


def plot_rd_curve(data, output_path, scene_name=""):
    """绘制 Model Size vs PSNR 率失真曲线。"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Scaffold-GS vs HAC++  Rate-Distortion Comparison  ({scene_name})',
                 fontsize=14, fontweight='bold')

    # 分离基线和 HAC++ 数据点
    baseline = None
    hac_points = []

    for item in data:
        if item['size_mb'] is None or item['size_mb'] == 0:
            continue
        if 'Baseline' in item['name']:
            baseline = item
        else:
            hac_points.append(item)

    # 按 size 排序
    hac_points.sort(key=lambda x: x['size_mb'] if x['size_mb'] else 0)

    # ---- 图1: Size vs PSNR ----
    ax = axes[0]
    if baseline and baseline['psnr']:
        ax.axhline(y=baseline['psnr'], color='red', linestyle='--', linewidth=1.5,
                    label=f"Scaffold-GS ({baseline['psnr']:.2f}dB, {baseline['size_mb']:.1f}MB)")
        ax.axvline(x=baseline['size_mb'], color='red', linestyle=':', alpha=0.3)
        ax.scatter([baseline['size_mb']], [baseline['psnr']], color='red', s=100,
                    zorder=5, marker='*')

    sizes = [p['size_mb'] for p in hac_points if p['psnr']]
    psnrs = [p['psnr'] for p in hac_points if p['psnr']]
    names = [p['name'] for p in hac_points if p['psnr']]

    if sizes:
        ax.plot(sizes, psnrs, 'b-o', linewidth=2, markersize=8, label='HAC++')
        for s, p, n in zip(sizes, psnrs, names):
            # 提取 lambda 值作为标签
            label = n.split('=')[-1].rstrip(')') if '=' in n else n
            ax.annotate(label, (s, p), textcoords="offset points",
                       xytext=(5, 8), fontsize=7, color='blue')

    ax.set_xlabel('Model Size (MB)', fontsize=11)
    ax.set_ylabel('PSNR (dB) ↑', fontsize=11)
    ax.set_title('Rate-Distortion: Size vs PSNR')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- 图2: Size vs SSIM ----
    ax = axes[1]
    if baseline and baseline['ssim']:
        ax.axhline(y=baseline['ssim'], color='red', linestyle='--', linewidth=1.5,
                    label=f"Scaffold-GS ({baseline['ssim']:.4f})")
        ax.scatter([baseline['size_mb']], [baseline['ssim']], color='red', s=100,
                    zorder=5, marker='*')

    ssims = [p['ssim'] for p in hac_points if p['ssim']]
    sizes_s = [p['size_mb'] for p in hac_points if p['ssim']]
    if sizes_s:
        ax.plot(sizes_s, ssims, 'b-o', linewidth=2, markersize=8, label='HAC++')

    ax.set_xlabel('Model Size (MB)', fontsize=11)
    ax.set_ylabel('SSIM ↑', fontsize=11)
    ax.set_title('Rate-Distortion: Size vs SSIM')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- 图3: Size vs LPIPS ----
    ax = axes[2]
    if baseline and baseline['lpips']:
        ax.axhline(y=baseline['lpips'], color='red', linestyle='--', linewidth=1.5,
                    label=f"Scaffold-GS ({baseline['lpips']:.4f})")
        ax.scatter([baseline['size_mb']], [baseline['lpips']], color='red', s=100,
                    zorder=5, marker='*')

    lpipss = [p['lpips'] for p in hac_points if p['lpips']]
    sizes_l = [p['size_mb'] for p in hac_points if p['lpips']]
    if sizes_l:
        ax.plot(sizes_l, lpipss, 'b-o', linewidth=2, markersize=8, label='HAC++')

    ax.set_xlabel('Model Size (MB)', fontsize=11)
    ax.set_ylabel('LPIPS ↓', fontsize=11)
    ax.set_title('Rate-Distortion: Size vs LPIPS')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"率失真曲线已保存到: {output_path}")
    plt.close()


def plot_compression_bar(data, output_path, scene_name=""):
    """绘制压缩比和质量对比柱状图。"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Compression & Quality Comparison ({scene_name})',
                 fontsize=14, fontweight='bold')

    baseline = None
    others = []
    for item in data:
        if 'Baseline' in item['name']:
            baseline = item
        elif item['size_mb'] and item['size_mb'] > 0:
            others.append(item)

    if not baseline or not baseline['size_mb']:
        print("警告: 未找到基线结果, 跳过压缩比图")
        plt.close()
        return

    names = [it['name'].replace('HAC++ ', '').replace('(', '').replace(')', '') for it in others]
    compression_ratios = [baseline['size_mb'] / it['size_mb'] for it in others]
    psnr_diffs = [(it['psnr'] - baseline['psnr']) if it['psnr'] else 0 for it in others]

    # ---- 图1: 压缩比 ----
    ax = axes[0]
    colors = ['green' if r > 1 else 'orange' for r in compression_ratios]
    bars = ax.bar(range(len(names)), compression_ratios, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=1.0, color='red', linestyle='--', label='Scaffold-GS (1x)')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Compression Ratio ↑')
    ax.set_title('Storage Compression Ratio (vs Scaffold-GS)')
    ax.legend()
    for bar, ratio in zip(bars, compression_ratios):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{ratio:.2f}x', ha='center', va='bottom', fontsize=9)

    # ---- 图2: PSNR 差值 ----
    ax = axes[1]
    colors = ['green' if d >= -0.5 else 'red' for d in psnr_diffs]
    bars = ax.bar(range(len(names)), psnr_diffs, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='red', linestyle='--', label='Scaffold-GS baseline')
    ax.axhline(y=-0.5, color='orange', linestyle=':', alpha=0.5, label='-0.5dB threshold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('ΔPSNR (dB)')
    ax.set_title('PSNR Difference (vs Scaffold-GS)')
    ax.legend()
    for bar, diff in zip(bars, psnr_diffs):
        sign = "+" if diff >= 0 else ""
        ax.text(bar.get_x() + bar.get_width()/2.,
                bar.get_height() + (0.05 if diff >= 0 else -0.15),
                f'{sign}{diff:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"压缩对比图已保存到: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="rd_curve_data.json 文件路径")
    parser.add_argument("--output", type=str, default="rd_curve.png", help="输出图片路径")
    parser.add_argument("--scene", type=str, default="", help="场景名称 (用于标题)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)

    # 绘制率失真曲线
    plot_rd_curve(data, args.output, args.scene)

    # 绘制压缩比柱状图
    bar_path = args.output.replace('.png', '_compression.png')
    plot_compression_bar(data, bar_path, args.scene)
