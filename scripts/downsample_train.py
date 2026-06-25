"""
downsample_train.py — 将训练集每种语言降采样至 max_samples_per_lang 条

用法:
  python3 scripts/downsample_train.py [--max-samples 20000]
  python3 scripts/downsample_train.py --dry-run   # 只看统计，不写入
"""

import csv
import os
import random
import argparse
from collections import defaultdict

DATA_DIR = "/Users/feiyu/Desktop/CodeTextClassifier/data"
TRAIN_CSV = os.path.join(DATA_DIR, "train_create_ml.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "train_create_ml.csv")  # 原地覆盖


def main():
    parser = argparse.ArgumentParser(description="降采样训练集")
    parser.add_argument("--max-samples", "-n", type=int, default=20000,
                        help="每种语言最多保留的样本数 (默认: 20000)")
    parser.add_argument("--dry-run", "-d", action="store_true",
                        help="只打印统计，不写入文件")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--output", "-o",
                        help="输出文件路径，默认覆盖原文件")
    args = parser.parse_args()

    random.seed(args.seed)

    # ── 读取 ──
    print(f"读取训练集: {TRAIN_CSV}")
    samples_by_lang = defaultdict(list)
    header = None
    total = 0

    with open(TRAIN_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                header = row
                continue
            if len(row) >= 2:
                # 文本中可能含逗号，csv.reader 已正确解析
                text = ",".join(row[:-1])  # 前 N-1 个字段是文本
                label = row[-1]            # 最后一个字段是标签
                samples_by_lang[label].append(text)
                total += 1
            else:
                print(f"  跳过异常行 {i+1}: {row}")

    print(f"  共读取 {total} 条样本，{len(samples_by_lang)} 种语言\n")

    # ── 统计 ──
    print(f"{'语言':<20} {'原始':>8} {'降采样后':>10}")
    print("-" * 40)
    total_original = 0
    total_downsampled = 0
    selected = {}  # label -> list of texts

    for lang in sorted(samples_by_lang.keys()):
        samples = samples_by_lang[lang]
        original_count = len(samples)
        total_original += original_count

        if original_count > args.max_samples:
            downsampled = random.sample(samples, args.max_samples)
        else:
            downsampled = samples

        selected[lang] = downsampled
        total_downsampled += len(downsampled)
        print(f"{lang:<20} {original_count:>8} {len(downsampled):>10}")

    print("-" * 40)
    print(f"{'TOTAL':<20} {total_original:>8} {total_downsampled:>10}")

    # ── 写入 ──
    if not args.dry_run:
        output_path = args.output or OUTPUT_CSV
        backup_path = output_path + ".bak"

        # 创建备份
        if os.path.exists(output_path):
            os.rename(output_path, backup_path)
            print(f"\n原始文件已备份: {backup_path}")

        print(f"\n写入降采样后的训练集: {output_path}")
        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for lang in sorted(selected.keys()):
                for text in selected[lang]:
                    writer.writerow([text, lang])

        print(f"  完成！共写入 {total_downsampled} 条样本")
    else:
        print("\n(dry-run 模式，未写入文件)")


if __name__ == "__main__":
    main()
