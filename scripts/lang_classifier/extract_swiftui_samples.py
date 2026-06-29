"""
extract_swiftui_samples.py — 从指定的 SwiftUI 开源项目中提取 Swift 代码样本，
追加到训练集 CSV 中。

用法:
  python3 scripts/extract_swiftui_samples.py [--max-samples 20000] [--dry-run]
"""

import csv
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# ── 配置 ────────────────────────────────────────────────────
DATA_DIR = str(Path(__file__).resolve().parent.parent.parent / "data" / "lang_classifier")
TRAIN_CSV = os.path.join(DATA_DIR, "train_create_ml.csv")
REPOS_BASE = str(Path(__file__).resolve().parent.parent.parent / "repos" / "Swift")

# 只处理新增的 SwiftUI 项目（不包含已有 repos）
SWIFTUI_REPOS = [
    "SwiftUIX",
    "swift-composable-architecture",
    "clean-architecture-swiftui",
    "PostApp",
    "swiftui-introspect",
    "KeyboardKit",
    "SwiftUICalendar",
    "Grid",
    "Pulse",
]

# 已有项目中可能也包含 SwiftUI 代码，但我们只从新增项目中提取
# 避免与已有 2 万条样本重叠

# ── 注释剥离 ────────────────────────────────────────────────
COMMENT_PATTERNS = {
    "Swift": (r'//.*', r'/\*[\s\S]*?\*/'),
}

def strip_comments(text: str) -> str:
    single_pat, multi_pat = COMMENT_PATTERNS["Swift"]
    if multi_pat:
        text = re.sub(multi_pat, '', text)
    if single_pat:
        text = re.sub(single_pat, '', text, flags=re.MULTILINE)
    lines = [l for l in text.split('\n') if l.strip()]
    return '\n'.join(lines) if lines else ''


# ── 代码块提取 ──────────────────────────────────────────────
def find_matching_brace(lines: list[str], start_idx: int, max_lines: int = 500) -> int:
    brace_depth = 0
    started = False
    end = min(start_idx + max_lines, len(lines))
    for i in range(start_idx, end):
        line = lines[i]
        for ch in line:
            if ch == '{':
                brace_depth += 1
                started = True
            elif ch == '}':
                brace_depth -= 1
        if started and brace_depth == 0:
            return i
    return end - 1


def extract_braces_block_at(lines: list[str], start_idx: int) -> int | None:
    combined = ''
    for i in range(start_idx, min(start_idx + 20, len(lines))):
        combined += lines[i]
        if '{' in lines[i]:
            return find_matching_brace(lines, i)
    return None


def extract_python_block(lines: list[str], start_idx: int) -> int:
    """对于 indent_based 的 SwiftUI 预览之类也用缩进检测"""
    if start_idx >= len(lines):
        return start_idx
    base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
    last_idx = start_idx
    for i in range(start_idx + 1, len(lines)):
        if lines[i].strip() == '':
            continue
        indent = len(lines[i]) - len(lines[i].lstrip())
        if indent <= base_indent and lines[i].strip() != '':
            break
        last_idx = i
    return last_idx


# ── Swift 函数/结构/枚举/协议 正则 ────────────────────────
SWIFT_FUNCTION_PATTERN = re.compile(
    r'^\s*(public|private|internal|open|fileprivate|static|final|override|class|struct|enum|extension|protocol)?'
    r'(?:\s+(public|private|internal|open|fileprivate|static|final|override|class|struct|enum|extension|protocol))*'
    r'\s*(func|init|subscript|deinit)\s+\w+'
)
SWIFT_CLASS_PATTERN = re.compile(
    r'^\s*(public|private|internal|open|fileprivate)?\s*(class|struct|enum|extension|protocol)\s+\w+'
)
SWIFT_VAR_PATTERN = re.compile(
    r'^\s*(public|private|internal|open|fileprivate|static)?\s*(var|let)\s+\w+'
)

MAX_SAMPLE_CHARS = 5000
MAX_BLOCK_LINES = 500


def extract_samples_from_swift_file(filepath: str) -> list[str]:
    """从单个 .swift 文件提取样本，返回代码片段列表。"""
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
    except (IOError, OSError):
        return []

    if b'\0' in raw[:8192] or len(raw) == 0:
        return []

    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        try:
            text = raw.decode('latin-1')
        except UnicodeDecodeError:
            return []

    # 只处理包含 import SwiftUI 的文件
    if 'import SwiftUI' not in text:
        return []

    text = strip_comments(text)
    if not text.strip():
        return []

    lines = text.split('\n')
    if len(lines) > 2000:
        lines = lines[:2000]

    samples = set()
    line_count = len(lines)

    # 1. 文件前 N 行截取
    for n in [3, 5, 7, 10, 15]:
        if line_count >= n:
            snippet = '\n'.join(lines[:n]).strip()
            if len(snippet) >= 10:
                samples.add(snippet)

    # 2. 完整文件（小文件）
    if line_count <= 100:
        text_stripped = text.strip()
        if len(text_stripped) >= 10:
            samples.add(text_stripped)

    # 3. 随机连续行组
    if line_count >= 8:
        for _ in range(min(5, max(1, line_count // 5))):
            start = random.randint(0, max(0, line_count - 8))
            length = random.randint(3, min(8, line_count - start))
            snippet = '\n'.join(lines[start:start + length]).strip()
            if len(snippet) >= 10:
                samples.add(snippet)

    # 4. 函数/类/枚举/协议体提取
    for i, line in enumerate(lines):
        matched = False
        for pat in [SWIFT_FUNCTION_PATTERN, SWIFT_CLASS_PATTERN, SWIFT_VAR_PATTERN]:
            if pat.match(line):
                matched = True
                break
        if not matched:
            continue

        # 找大括号块
        block_end = extract_braces_block_at(lines, i)
        if block_end is not None and block_end > i:
            block = '\n'.join(lines[i:block_end + 1]).strip()
            if len(block) >= 10:
                samples.add(block)
                # 前 N 行截取
                body_lines = lines[i:block_end + 1]
                for n in [3, 5]:
                    if len(body_lines) >= n + 1:
                        snippet = '\n'.join(body_lines[:n + 1]).strip()
                        if len(snippet) >= 10:
                            samples.add(snippet)

    # 5. 大块内容（没有匹配到函数的文件）
    if not any(SWIFT_FUNCTION_PATTERN.match(l) or SWIFT_CLASS_PATTERN.match(l) for l in lines[:50]):
        for n in [20, 30, 50, 80]:
            if line_count >= n:
                snippet = '\n'.join(lines[:n]).strip()
                if len(snippet) >= 10:
                    samples.add(snippet)
        if line_count >= 20:
            for _ in range(min(10, max(1, line_count // 10))):
                start = random.randint(0, max(0, line_count - 20))
                length = random.randint(5, min(20, line_count - start))
                snippet = '\n'.join(lines[start:start + length]).strip()
                if len(snippet) >= 10:
                    samples.add(snippet)

    return [s for s in samples if len(s) <= MAX_SAMPLE_CHARS]


def collect_swiftui_files() -> list[str]:
    """从 SWIFTUI_REPOS 收集所有包含 import SwiftUI 的 .swift 文件。"""
    files = []
    for repo_name in SWIFTUI_REPOS:
        repo_path = os.path.join(REPOS_BASE, repo_name)
        if not os.path.isdir(repo_path):
            continue
        for root, dirs, fnames in os.walk(repo_path):
            # 跳过隐藏目录和常见忽略目录
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in (
                'node_modules', 'vendor', 'Pods', 'DerivedData', 'build', '.build',
                'dist', 'third_party', '.venv', 'venv', 'env', '__pycache__', 'target',
                'tests', 'Tests', 'test', 'Test',
            )]
            for fname in fnames:
                if not fname.endswith('.swift'):
                    continue
                fpath = os.path.join(root, fname)
                # 快速检查文件是否包含 import SwiftUI
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        first_kb = f.read(4096)
                        if 'import SwiftUI' in first_kb:
                            files.append(fpath)
                except Exception:
                    continue
    return files


def main():
    import argparse
    parser = argparse.ArgumentParser(description="从 SwiftUI 开源项目提取样本并追加到训练集")
    parser.add_argument("--max-samples", "-n", type=int, default=20000,
                        help="最多提取的 SwiftUI 样本数 (默认: 20000)")
    parser.add_argument("--dry-run", "-d", action="store_true",
                        help="只打印统计，不写入")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", "-o",
                        help="输出文件路径，默认覆盖训练集")
    args = parser.parse_args()

    random.seed(args.seed)

    # ── 收集 SwiftUI 文件 ──
    print("扫描 SwiftUI 代码文件...")
    swiftui_files = collect_swiftui_files()
    print(f"  找到 {len(swiftui_files)} 个包含 import SwiftUI 的 .swift 文件")

    if len(swiftui_files) == 0:
        print("  错误：未找到任何 SwiftUI 文件！请检查仓库是否正确克隆。")
        sys.exit(1)

    # ── 提取样本 ──
    print("\n提取代码样本...")
    all_samples = []
    file_counts = defaultdict(int)
    for fpath in swiftui_files:
        samples = extract_samples_from_swift_file(fpath)
        if samples:
            repo_name = os.path.basename(os.path.dirname(fpath))
            all_samples.extend(samples)
            file_counts[repo_name] += 1

    print(f"  共提取 {len(all_samples)} 个原始样本")

    # 去重
    unique_samples = list(set(all_samples))
    print(f"  去重后 {len(unique_samples)} 个样本")

    # ── 按文件来源分组后降采样 ──
    # 先尽量保持每个 repo 的分布
    random.shuffle(unique_samples)

    if len(unique_samples) > args.max_samples:
        final_samples = random.sample(unique_samples, args.max_samples)
    else:
        final_samples = unique_samples

    print(f"\n  最终选取 {len(final_samples)} 个样本（目标: {args.max_samples}）")

    # ── 写入 ──
    if args.dry_run:
        print("\n(dry-run) 准备追加的样本:")
        print(f"  标签: Swift")
        print(f"  数量: {len(final_samples)}")
        print(f"  目标文件: {args.output or TRAIN_CSV}")
        return

    output_path = args.output or TRAIN_CSV

    print(f"\n读取已有训练集: {output_path}")
    existing_rows = []
    with open(output_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            existing_rows.append(row)

    header = existing_rows[0]
    data_rows = existing_rows[1:]
    print(f"  已有 {len(data_rows)} 条数据")

    print(f"\n追加 {len(final_samples)} 条 SwiftUI 样本...")
    for sample_text in final_samples:
        # 清理文本：与 export_create_ml_csv.py 保持一致
        cleaned = sample_text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
        cleaned = cleaned.replace(',', ' ').replace('"', ' ')
        cleaned = cleaned.replace('\t', ' ')
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        data_rows.append([cleaned, "Swift"])

    random.shuffle(data_rows)

    print(f"写入 {output_path} ...")
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data_rows:
            writer.writerow(row)

    print(f"  完成！总行数: {len(data_rows)}（含 Swift: {sum(1 for r in data_rows if r[-1]=='Swift')}）")

    # ── 统计 ──
    from collections import Counter
    lang_counts = Counter(r[-1] for r in data_rows)
    print(f"\n训练集语言分布:")
    print(f"{'语言':<20} {'数量':>8}")
    print("-" * 28)
    for lang in sorted(lang_counts.keys()):
        print(f"{lang:<20} {lang_counts[lang]:>8}")
    print("-" * 28)
    print(f"{'TOTAL':<20} {len(data_rows):>8}")


if __name__ == "__main__":
    main()
