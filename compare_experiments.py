"""
Scaffold-GS vs HAC++ 对比实验脚本

功能:
  1. 自动运行 Scaffold-GS (原版) 和 HAC++ (多组 lambda_rate) 实验
  2. 收集所有实验的质量指标 (PSNR / SSIM / LPIPS)
  3. 统计模型大小 (ply + 网络权重)
  4. 输出对比表格 + 率失真曲线数据

用法:
  python compare_experiments.py --scene flowers --data_root data/mipnerf360 --gpu 0

  或者只收集已有结果 (不重新训练):
  python compare_experiments.py --scene flowers --data_root data/mipnerf360 --collect_only
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from collections import OrderedDict

# ====================================================================
#  配置: 需要运行的实验列表
# ====================================================================

def get_experiments(scene_name, data_root, output_root="outputs"):
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
        "--lod", "0",
        "--voxel_size", "0.001",
        "--update_init_factor", "16",
        "--appearance_dim", "0",
        "--ratio", "1",
        "--iterations", "30000",
    ]

    experiments = []

    # ---- 1. 原版 Scaffold-GS (基线) ----
    scaffold_path = f"{output_root}/{scene_name}/scaffold_gs"
    experiments.append({
        "name": "Scaffold-GS (Baseline)",
        "model_path": scaffold_path,
        "cmd": ["python", "train.py"] + base_args + [
            "-m", scaffold_path,
            "--port", "12340",
        ],
    })

    # ---- 2. HAC++ 不同 lambda_rate (率失真曲线) ----
    lambda_rates = [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    for lr in lambda_rates:
        lr_str = f"{lr:.0e}" if lr > 0 else "0"
        hac_path = f"{output_root}/{scene_name}/hac_lr{lr_str}"
        experiments.append({
            "name": f"HAC++ (λ_rate={lr_str})",
            "model_path": hac_path,
            "cmd": ["python", "train.py"] + base_args + [
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
    results_file = Path(model_path) / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        # results.json 的结构: { model_path: { method: { PSNR, SSIM, LPIPS } } }
        for scene_key in data:
            for method_key in data[scene_key]:
                metrics = data[scene_key][method_key]
                result['psnr'] = metrics.get('PSNR')
                result['ssim'] = metrics.get('SSIM')
                result['lpips'] = metrics.get('LPIPS')
                break
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
#  主流程
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Scaffold-GS vs HAC++ 对比实验")
    parser.add_argument("--scene", type=str, required=True, help="场景名称 (如 flowers)")
    parser.add_argument("--data_root", type=str, default="data/mipnerf360", help="数据集根目录")
    parser.add_argument("--output_root", type=str, default="outputs", help="实验输出根目录")
    parser.add_argument("--gpu", type=str, default="0", help="GPU 编号")
    parser.add_argument("--collect_only", action="store_true", help="仅收集已有结果,不重新训练")
    parser.add_argument("--skip_baseline", action="store_true", help="跳过 Scaffold-GS 基线训练")
    args = parser.parse_args()

    experiments = get_experiments(args.scene, args.data_root, args.output_root)

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

            try:
                subprocess.run(cmd, check=True)
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
    rd_file = f"{args.output_root}/{args.scene}/rd_curve_data.json"
    os.makedirs(os.path.dirname(rd_file), exist_ok=True)
    export_rd_curve_data(all_results, rd_file)

    # ---- 生成绘图建议 ----
    print("\n" + "=" * 80)
    print("  绘图建议 (用 matplotlib 或其他工具)")
    print("=" * 80)
    print(f"""
  1. 率失真曲线 (最核心):
     X轴: Model Size (MB)  |  Y轴: PSNR (dB)
     每个实验是一个点, Scaffold-GS 是水平参考线
     数据文件: {rd_file}

  2. 压缩比柱状图:
     X轴: 方法名  |  Y轴: 压缩比 (相对 Scaffold-GS)

  3. 质量对比渲染图:
     同一视角的 GT / Scaffold-GS / HAC++ 渲染结果 + error map
     图片目录: {{model_path}}/test/ours_30000/renders/

  4. 收敛曲线 (如果使用了 tensorboard):
     X轴: iteration  |  Y轴: training loss
""")


if __name__ == "__main__":
    main()
