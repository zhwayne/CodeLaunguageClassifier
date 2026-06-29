"""验证输出数据集质量"""
import json
from collections import Counter
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "lang_classifier"

for fname in ["train.jsonl", "val.jsonl", "test.jsonl"]:
    path = data_dir / fname
    print(f"\n{'='*60}")
    print(f"检查: {fname}")
    print(f"{'='*60}")

    lines = path.read_text(encoding='utf-8').strip().split('\n')
    print(f"  总行数: {len(lines)}")

    errors = 0
    labels = Counter()
    lengths = []
    has_path_leak = 0

    for i, line in enumerate(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            errors += 1
            print(f"  ❌ 第 {i+1} 行 JSON 解析失败")
            continue

        # 检查字段
        if "text" not in entry or "label" not in entry:
            errors += 1
            print(f"  ❌ 第 {i+1} 行缺少字段: {list(entry.keys())}")
            continue

        text = entry["text"]
        label = entry["label"]

        # 检查标签是否在预期列表中
        expected_labels = {
            "Swift", "Objective-C", "Kotlin", "Java", "JavaScript",
            "TypeScript", "Python", "Go", "Rust", "C", "C++", "C#",
            "Ruby", "PHP", "Shell"
        }
        if label not in expected_labels:
            errors += 1
            print(f"  ⚠  未知标签: {label}")

        labels[label] += 1
        lengths.append(len(text))

        # 检查路径泄露
        leaked = False
        for marker in ['/', '\\', '.swift', '.py', '.java', '.kt', '.js', '.ts', '.go', '.rs',
                        '.c', '.cpp', '.h', '.cs', '.rb', '.php', '.sh', '.m', '.mm']:
            if '\n' not in text and marker in text:
                # 有路径特征的文本需要仔细检查
                if any(p in text for p in ['/Users/', '/home/', 'C:\\', 'src/', 'lib/']):
                    has_path_leak += 1
                    leaked = True
                    break
        if leaked and has_path_leak <= 3:
            print(f"  ⚠  可能的路径泄露: {text[:80]}...")

    print(f"\n  格式错误: {errors}")
    print(f"  可能的路径泄露: {has_path_leak}")
    print(f"\n  语言分布 ({fname}):")
    print(f"  {'语言':<20} {'数量':>8}")
    print(f"  {'-'*28}")
    for lang, count in sorted(labels.items()):
        print(f"  {lang:<20} {count:>8}")
    print(f"  {'-'*28}")
    print(f"  {'总计':<20} {sum(labels.values()):>8}")

    # 样本长度统计
    if lengths:
        avg_len = sum(lengths) / len(lengths)
        max_len = max(lengths)
        short_samples = sum(1 for l in lengths if l < 50)
        print(f"\n  样本长度:")
        print(f"    平均: {avg_len:.0f} 字符")
        print(f"    最短: {min(lengths)} 字符")
        print(f"    最长: {max_len} 字符")
        print(f"    短样本(<50字符): {short_samples} ({short_samples/len(lengths)*100:.1f}%)")

    total_lines = sum(len(p.read_text(encoding='utf-8').strip().split('\n')) for p in data_dir.glob("*.jsonl"))
    print(f"\n  总样本数: {total_lines}")
