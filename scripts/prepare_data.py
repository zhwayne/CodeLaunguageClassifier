"""
prepare_data.py — 编程语言识别训练数据准备管线

从指定源码目录提取代码样本，生成 JSONL 格式训练数据。

用法:
  python3 scripts/prepare_data.py
  python3 scripts/prepare_data.py --help
"""
import argparse
import json
import os
import re
import random
import sys
from collections import defaultdict
from pathlib import Path

# ── 语言配置 ─────────────────────────────────────────────────
LANGUAGE_CONFIG = {
    "Swift": {
        "extensions": [".swift"],
        "linguist_dir": "Swift",
        "function_pattern": re.compile(
            r'^\s*(public|private|internal|open|fileprivate|static|final|override|class|struct|enum|extension|protocol)?'
            r'(?:\s+(public|private|internal|open|fileprivate|static|final|override|class|struct|enum|extension|protocol))*'
            r'\s*(func|init|subscript|deinit)\s+\w+'
        ),
        "class_pattern": re.compile(r'^\s*(public|private|internal|open|fileprivate)?\s*(class|struct|enum|extension|protocol)\s+\w+'),
    },
    "Objective-C": {
        "extensions": [".m", ".h", ".mm"],
        "linguist_dir": "Objective-C",
        "function_pattern": re.compile(r'^[\s+\-*]*\([\w\s\*<>,]+\)\s*\w+\s*[:;{]'),
        "class_pattern": re.compile(r'^@(interface|implementation|protocol)\s+\w+'),
    },
    "Kotlin": {
        "extensions": [".kt", ".kts"],
        "linguist_dir": "Kotlin",
        "function_pattern": re.compile(
            r'^\s*(public|private|internal|protected|open|override|abstract|final|suspend|inline|operator|infix|tailrec|external)?'
            r'\s*(fun)\s+\w+'
        ),
        "class_pattern": re.compile(r'^\s*(public|private|internal|protected)?\s*(class|interface|object|enum class|data class|sealed class)\s+\w+'),
    },
    "Java": {
        "extensions": [".java"],
        "linguist_dir": "Java",
        "function_pattern": re.compile(
            r'^\s*(public|private|protected|static|final|abstract|synchronized|native|transient|volatile|strictfp|default)?'
            r'(?:\s+(public|private|protected|static|final|abstract|synchronized|native|transient|volatile|strictfp|default))*'
            r'\s*(<[^>]+>\s*)?\w+(?:\.\w+)*\s+\w+\s*\('
        ),
        "class_pattern": re.compile(r'^\s*(public|private|protected|static|abstract|final)?\s*(class|interface|enum|@interface|record)\s+\w+'),
    },
    "JavaScript": {
        "extensions": [".js", ".jsx", ".cjs", ".mjs"],
        "linguist_dir": "JavaScript",
        "function_pattern": re.compile(
            r'^\s*(export\s+)?(default\s+)?(async\s+)?(function\s*\*?\s+\w+|const\s+\w+\s*=\s*(async\s*)?\(|let\s+\w+\s*=\s*(async\s*)?\(|var\s+\w+\s*=\s*(async\s*)?\()'
        ),
        "class_pattern": re.compile(r'^\s*(export\s+)?(default\s+)?(abstract\s+)?class\s+\w+'),
    },
    "TypeScript": {
        "extensions": [".ts", ".tsx"],
        "linguist_dir": "TypeScript",
        "function_pattern": re.compile(
            r'^\s*(export\s+)?(default\s+)?(public|private|protected|static|readonly|abstract|async)?'
            r'\s*(function\s+\w+|const\s+\w+\s*=\s*(async\s*)?\(|let\s+\w+\s*=\s*\(|var\s+\w+\s*=\s*\()'
        ),
        "class_pattern": re.compile(r'^\s*(export\s+)?(default\s+)?(abstract\s+)?class\s+\w+'),
    },
    "Python": {
        "extensions": [".py"],
        "linguist_dir": "Python",
        "function_pattern": re.compile(r'^\s*(async\s+)?def\s+\w+'),
        "class_pattern": re.compile(r'^\s*class\s+\w+'),
        "indent_based": True,
    },
    "Go": {
        "extensions": [".go"],
        "linguist_dir": "Go",
        "function_pattern": re.compile(r'^\s*func\s+\w+'),
        "class_pattern": re.compile(r'^\s*type\s+\w+\s+struct'),
    },
    "Rust": {
        "extensions": [".rs"],
        "linguist_dir": "Rust",
        "function_pattern": re.compile(r'^\s*(pub\s+)?(unsafe\s+)?(async\s+)?(extern\s+\w+\s+)?fn\s+\w+'),
        "class_pattern": re.compile(r'^\s*(pub\s+)?(unsafe\s+)?(struct|enum|trait|impl|union)\s+\w+'),
    },
    "C": {
        "extensions": [".c", ".h"],
        "linguist_dir": "C",
        "function_pattern": re.compile(
            r'^\s*(static\s+|extern\s+|inline\s+|const\s+|unsigned\s+|signed\s+|volatile\s+|restrict\s+)?'
            r'(\w+\s+)*\w+\s+\w+\s*\('
        ),
    },
    "C++": {
        "extensions": [".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx", ".h", ".c++", ".cp", ".cxx", ".tcc"],
        "linguist_dir": "C++",
        "function_pattern": re.compile(
            r'^\s*(template\s*<[^>]*>\s*)?(virtual\s+|static\s+|inline\s+|const\s+|explicit\s+|friend\s+|override\s+|final\s+|constexpr\s+|noexcept\s+)?'
            r'(\w+(?:::|<\w+>)?\s+)*\w+\s+~?\w+\s*\('
        ),
        "class_pattern": re.compile(r'^\s*(class|struct|enum|union)\s+\w+'),
    },
    "C#": {
        "extensions": [".cs"],
        "linguist_dir": "C#",
        "function_pattern": re.compile(
            r'^\s*(public|private|protected|internal|static|virtual|override|abstract|sealed|readonly|async|unsafe|partial|new|fixed|extern)?'
            r'(?:\s+(public|private|protected|internal|static|virtual|override|abstract|sealed|readonly|async|unsafe|partial|new|fixed|extern))*'
            r'\s*(<[^>]+>\s*)?\w+\s+\w+\s*\('
        ),
        "class_pattern": re.compile(r'^\s*(public|private|protected|internal|static|abstract|sealed|partial|readonly)?\s*(class|struct|interface|enum|record)\s+\w+'),
    },
    "Ruby": {
        "extensions": [".rb"],
        "linguist_dir": "Ruby",
        "function_pattern": re.compile(r'^\s*(def)\s+\w+'),
        "class_pattern": re.compile(r'^\s*(class|module)\s+\w+'),
        "end_based": True,
    },
    "PHP": {
        "extensions": [".php"],
        "linguist_dir": "PHP",
        "function_pattern": re.compile(r'^\s*(public|private|protected|static|abstract|final)?\s*function\s+\w+'),
        "class_pattern": re.compile(r'^\s*(abstract\s+|final\s+)?class\s+\w+'),
    },
    "Shell": {
        "extensions": [".sh", ".bash", ".zsh"],
        "linguist_dir": "Shell",
        "function_pattern": re.compile(r'^\s*(function\s+\w+|\w+\s*\(\s*\)\s*\{?)'),
        "line_based": True,
    },
    "HTML": {
        "extensions": [".html", ".htm", ".vue", ".svelte"],
        "linguist_dir": "HTML",
        "tag_based": True,
        "no_function": True,
    },
    "CSS": {
        "extensions": [".css", ".scss", ".less"],
        "linguist_dir": "CSS",
        "no_function": True,
    },
    "XML": {
        "extensions": [".xml", ".xsd", ".xsl", ".svg"],
        "linguist_dir": "XML",
        "tag_based": True,
        "no_function": True,
    },
    "SQL": {
        "extensions": [".sql", ".psql", ".plsql"],
        "linguist_dir": "SQL",
        "function_pattern": re.compile(r'^\s*(CREATE\s+(OR\s+REPLACE\s+)?(FUNCTION|PROCEDURE|TRIGGER|VIEW|INDEX)\s+\w+|SELECT|INSERT|UPDATE|DELETE|WITH)\s', re.IGNORECASE),
    },
    "JSON": {
        "extensions": [".json"],
        "linguist_dir": "JSON",
        "no_function": True,
    },
    "YAML": {
        "extensions": [".yaml", ".yml"],
        "linguist_dir": "YAML",
        "no_function": True,
    },
    "Lua": {
        "extensions": [".lua"],
        "linguist_dir": "Lua",
        "function_pattern": re.compile(r'^\s*(local\s+)?function\s+\w+'),
        "class_pattern": re.compile(r'^\s*local\s+\w+\s*=\s*\w+[:.]'),
    },
    "Dart": {
        "extensions": [".dart"],
        "linguist_dir": "Dart",
        "function_pattern": re.compile(
            r'^\s*(static\s+|abstract\s+|external\s+|factory\s+)?'
            r'(?:\s*(void|int|String|bool|double|Future|List|Map|dynamic|var|final|const))?\s*\w+\s*\('
        ),
        "class_pattern": re.compile(r'^\s*(abstract\s+)?(class|enum|mixin|extension)\s+\w+'),
    },
}

# ── 注释剥离 ─────────────────────────────────────────────────

COMMENT_PATTERNS = {
    "Swift":      (r'//.*', r'/\*[\s\S]*?\*/'),
    "Objective-C":(r'//.*', r'/\*[\s\S]*?\*/'),
    "Kotlin":     (r'//.*', r'/\*[\s\S]*?\*/'),
    "Java":       (r'//.*', r'/\*[\s\S]*?\*/'),
    "JavaScript": (r'//.*', r'/\*[\s\S]*?\*/'),
    "TypeScript": (r'//.*', r'/\*[\s\S]*?\*/'),
    "Python":     (r'#.*',  r'"""[\s\S]*?"""|' "'''[\\s\\S]*?'''"),
    "Go":         (r'//.*', r'/\*[\s\S]*?\*/'),
    "Rust":       (r'//.*', r'/\*[\s\S]*?\*/'),
    "C":          (r'//.*', r'/\*[\s\S]*?\*/'),
    "C++":        (r'//.*', r'/\*[\s\S]*?\*/'),
    "C#":         (r'//.*', r'/\*[\s\S]*?\*/'),
    "Ruby":       (r'#.*',  r'=begin[\s\S]*?=end'),
    "PHP":        (r'//.*|#.*', r'/\*[\s\S]*?\*/'),
    "Shell":      (r'#.*',  None),
    "HTML":       (None,    r'<!--[\s\S]*?-->'),
    "CSS":        (r'//.*', r'/\*[\s\S]*?\*/'),
    "XML":        (None,    r'<!--[\s\S]*?-->'),
    "SQL":        (r'--.*', r'/\*[\s\S]*?\*/'),
    "JSON":       (None,    None),
    "YAML":       (r'#.*',  None),
    "Lua":        (r'--.*', r'--\[\[[\s\S]*?\]\]'),
    "Dart":       (r'//.*', r'/\*[\s\S]*?\*/'),
}

def strip_comments(text: str, lang: str) -> str:
    single_pat, multi_pat = COMMENT_PATTERNS.get(lang, (r'//.*', r'/\*[\s\S]*?\*/'))
    if multi_pat:
        text = re.sub(multi_pat, '', text)
    if single_pat:
        text = re.sub(single_pat, '', text, flags=re.MULTILINE)
    # 清理空白行
    lines = [l for l in text.split('\n') if l.strip()]
    return '\n'.join(lines) if lines else ''


# ── 工具函数 ─────────────────────────────────────────────────

def is_binary(content: bytes) -> bool:
    """检查是否包含 null 字节（二进制文件特征）。"""
    return b'\0' in content[:8192]


def find_matching_brace(lines: list[str], start_idx: int, max_lines: int = 500) -> int:
    """从 start_idx 开始，找到匹配的闭合大括号行号（brace-based 语言）。"""
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


def find_matching_end(lines: list[str], start_idx: int) -> int:
    """用 `end` 关键字匹配块结束（Ruby）。"""
    depth = 1
    for i in range(start_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith('#'):
            continue
        if re.match(r'^\s*(class|module|def|if|unless|case|begin|do)\s', stripped) or stripped in ('if', 'do'):
            depth += 1
        elif stripped == 'end':
            depth -= 1
        if depth == 0:
            return i
    return len(lines) - 1


def extract_python_block(lines: list[str], start_idx: int) -> int:
    """提取 Python 缩进块。"""
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


def extract_braces_block_at(lines: list[str], start_idx: int) -> int | None:
    """找到函数声明行中或之后的 '{', 然后提取到匹配的 '}'。"""
    # 先在 start_idx 行找 '{'
    combined = ''
    search_start = start_idx
    for i in range(start_idx, min(start_idx + 20, len(lines))):
        combined += lines[i]
        if '{' in lines[i]:
            search_start = i
            break
    else:
        return None  # 没找到 {
    return find_matching_brace(lines, search_start)


# ── 样本提取 ─────────────────────────────────────────────────

def extract_samples_from_file(filepath: str, lang: str, config: dict) -> list[str]:
    """从单个文件提取多个代码片段样本。"""
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
    except (IOError, OSError):
        return []

    if is_binary(raw) or len(raw) == 0:
        return []

    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        try:
            text = raw.decode('latin-1')
        except UnicodeDecodeError:
            return []

    text = strip_comments(text, lang)
    if not text.strip():
        return []

    lines = text.split('\n')
    if len(lines) > 2000:
        lines = lines[:2000]

    MAX_SAMPLE_CHARS = 5000
    MAX_BLOCK_LINES = 500

    samples = set()
    line_count = len(lines)

    # 1. 文件前 N 行截取（短片段）
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

    # 3. 随机连续行组（增加短片段多样性）
    if line_count >= 8:
        for _ in range(min(5, line_count // 5)):
            start = random.randint(0, max(0, line_count - 8))
            length = random.randint(3, min(8, line_count - start))
            snippet = '\n'.join(lines[start:start + length]).strip()
            if len(snippet) >= 10:
                samples.add(snippet)

    # 4. 函数/方法/类体提取
    if config.get("indent_based"):
        # Python
        for i, line in enumerate(lines):
            if config["function_pattern"].match(line) or config.get("class_pattern", re.compile(r'^$')).match(line):
                end = extract_python_block(lines, i)
                block = '\n'.join(lines[i:end + 1]).strip()
                if len(block) >= 10:
                    samples.add(block)
                    # 函数体前 N 行截取
                    body_lines = lines[i:end + 1]
                    for n in [3, 5]:
                        if len(body_lines) >= n + 1:
                            snippet = '\n'.join(body_lines[:n + 1]).strip()
                            if len(snippet) >= 10:
                                samples.add(snippet)

    elif config.get("end_based"):
        # Ruby
        for i, line in enumerate(lines):
            if config["function_pattern"].match(line) or config.get("class_pattern", re.compile(r'^$')).match(line):
                end = find_matching_end(lines, i)
                if end > i:
                    block = '\n'.join(lines[i:end + 1]).strip()
                    if len(block) >= 10:
                        samples.add(block)
                        body_lines = lines[i:end + 1]
                        for n in [3, 5]:
                            if len(body_lines) >= n + 1:
                                snippet = '\n'.join(body_lines[:n + 1]).strip()
                                if len(snippet) >= 10:
                                    samples.add(snippet)

    elif not config.get("no_function") and "function_pattern" in config:
        # Brace-based 语言 —— 用函数/类/结构体正则
        combined_patterns = [config["function_pattern"]]
        if "class_pattern" in config:
            combined_patterns.append(config["class_pattern"])

        for i, line in enumerate(lines):
            for pat in combined_patterns:
                if pat.match(line):
                    block_end = extract_braces_block_at(lines, i)
                    if block_end is not None and block_end > i:
                        block = '\n'.join(lines[i:block_end + 1]).strip()
                        if len(block) >= 10:
                            samples.add(block)
                            # 函数体前 N 行截取
                            body_lines = lines[i:block_end + 1]
                            for n in [3, 5]:
                                if len(body_lines) >= n + 1:
                                    snippet = '\n'.join(body_lines[:n + 1]).strip()
                                    if len(snippet) >= 10:
                                        samples.add(snippet)
                    break  # 一个函数只匹配一次
    elif config.get("no_function"):
        # HTML/CSS/XML/JSON/YAML —— 大块内容提取
        for n in [20, 30, 50, 80]:
            if line_count >= n:
                snippet = '\n'.join(lines[:n]).strip()
                if len(snippet) >= 10:
                    samples.add(snippet)
        # 多段随机截取
        if line_count >= 20:
            for _ in range(min(10, line_count // 10)):
                start = random.randint(0, max(0, line_count - 20))
                length = random.randint(5, min(20, line_count - start))
                snippet = '\n'.join(lines[start:start + length]).strip()
                if len(snippet) >= 10:
                    samples.add(snippet)

    return [s for s in samples if len(s) <= MAX_SAMPLE_CHARS]


def scan_source_files(src_dirs: list[str], extensions: set) -> list[str]:
    """扫描目录，收集指定扩展名的所有文件。"""
    files = []
    for src_dir in src_dirs:
        src_path = Path(src_dir)
        if not src_path.exists():
            continue
        for f in src_path.rglob('*'):
            if f.suffix.lower() in extensions and f.is_file():
                # 跳过隐藏路径和常见非代码目录
                rel = str(f.relative_to(src_path))
                if any(part.startswith('.') for part in f.parts):
                    continue
                if any(ignore in rel for ignore in [
                    '/node_modules/', 'node_modules/',
                    '/vendor/', 'vendor/',
                    '/.git/', '.git/',
                    '/Pods/', 'Pods/',
                    '/DerivedData/', 'DerivedData/',
                    '/build/', '/Build/',
                    '/dist/', '/.build/',
                    '/third_party/', 'third_party/',
                    '/.venv/', '/venv/', '/env/',
                    '/__pycache__/', '__pycache__',
                    '/target/', 'target/',
                    'package-lock.json', 'yarn.lock',
                    'min.', '.min.',
                    '/test/', '/tests/', '/spec/', '/fixtures/',
                ]):
                    continue
                files.append(str(f))
    return files


def scan_all_repos(extensions: set) -> list[str]:
    """扫描所有已克隆的 repo，收集指定扩展名的文件。"""
    repos_base = Path(__file__).resolve().parent.parent / "repos"
    files = []
    for repo_dir in repos_base.iterdir():
        if not repo_dir.is_dir() or repo_dir.name == "linguist":
            continue
        for f in repo_dir.rglob('*'):
            if f.suffix.lower() in extensions and f.is_file():
                if any(part.startswith('.') for part in f.parts):
                    continue
                rel = str(f.relative_to(repo_dir))
                if any(ignore in rel for ignore in [
                    '/node_modules/', 'node_modules/',
                    '/vendor/', 'vendor/',
                    '/.git/', '.git/',
                    '/Pods/', 'Pods/',
                    '/DerivedData/', 'DerivedData/',
                    '/build/', '/Build/',
                    '/dist/', '/.build/',
                    '/third_party/', 'third_party/',
                    '/.venv/', '/venv/', '/env/',
                    '/__pycache__/', '__pycache__',
                    '/target/', 'target/',
                    'package-lock.json', 'yarn.lock',
                    'min.', '.min.',
                ]):
                    continue
                files.append(str(f))
    return files


# ── 主流程 ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="编程语言识别训练数据准备")
    parser.add_argument("--lang", "-l", help="只处理指定语言")
    parser.add_argument("--output", "-o", default="data", help="输出目录")
    parser.add_argument("--samples-per-lang", "-n", type=int, default=50000,
                        help="每种语言目标样本数 (默认: 50000)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    base_dir = Path(__file__).resolve().parent.parent
    output_dir = base_dir / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    repos_dir = base_dir / "repos"
    linguist_dir = repos_dir / "linguist" / "samples"

    languages_to_process = [args.lang] if args.lang else list(LANGUAGE_CONFIG.keys())

    all_samples = defaultdict(list)         # lang -> list of (sample_text, source_file_id)
    file_id_counter = 0

    for lang in languages_to_process:
        config = LANGUAGE_CONFIG[lang]
        print(f"\n{'='*60}")
        print(f"处理语言: {lang}")
        print(f"{'='*60}")

        src_dirs = []

        # Linguist 样本目录
        linguist_lang_dir = linguist_dir / config["linguist_dir"]
        if linguist_lang_dir.exists():
            src_dirs.append(str(linguist_lang_dir))
            print(f"  Linguist 目录: {linguist_lang_dir}")

        # 项目 repos 目录
        lang_repo_dir = repos_dir / lang
        if lang_repo_dir.exists():
            src_dirs.append(str(lang_repo_dir))
            n_dirs = len(list(lang_repo_dir.iterdir())) if lang_repo_dir.is_dir() else 0
            print(f"  项目仓库目录: {lang_repo_dir} ({n_dirs} 个子目录)")

        # 如果源文件不足，搜索所有已克隆的 repo
        # 搜索所有已克隆的 repo（补充源文件不足的语言）
        files = scan_source_files(src_dirs, set(ext.lower() for ext in config["extensions"]))
        if len(files) < 3000:
            print(f"  源文件不足 ({len(files)} 个)，搜索所有 repo...")
            all_repos_files = scan_all_repos(set(ext.lower() for ext in config["extensions"]))
            existing = set(files)
            for f in all_repos_files:
                if f not in existing:
                    files.append(f)
            print(f"  搜索后共 {len(files)} 个源文件")

        print(f"  扫描到 {len(files)} 个源文件")

        random.shuffle(files)

        lang_samples = []
        file_source = {}  # file_id -> original_filepath

        for filepath in files:
            samples = extract_samples_from_file(filepath, lang, config)
            if samples:
                file_id = file_id_counter
                file_id_counter += 1
                file_source[file_id] = filepath
                for sample in samples:
                    lang_samples.append((sample, file_id))

        print(f"  提取到 {len(lang_samples)} 个样本，来自 {len(file_source)} 个文件")

        # 按文件级降采样
        if len(lang_samples) > args.samples_per_lang:
            file_counts = defaultdict(int)
            for _, fid in lang_samples:
                file_counts[fid] += 1
            file_ids = list(file_counts.keys())
            random.shuffle(file_ids)
            selected = set()
            total = 0
            for fid in file_ids:
                selected.add(fid)
                total += file_counts[fid]
                if total >= args.samples_per_lang:
                    break
            lang_samples = [(s, fid) for s, fid in lang_samples if fid in selected]
            print(f"  降采样至 {len(lang_samples)} 个样本（{len(selected)} 个文件）")
        elif len(lang_samples) < args.samples_per_lang and len(lang_samples) > 0:
            print(f"  ⚠  样本数 ({len(lang_samples)}) 未达到目标 ({args.samples_per_lang})")

        all_samples[lang] = lang_samples
        print(f"  → 最终保留 {len(lang_samples)} 个样本")

    # ── 按文件级划分 train/val/test ──
    print(f"\n{'='*60}")
    print("划分 train/val/test...")
    print(f"{'='*60}")

    train_file_ids = set()
    val_file_ids = set()
    test_file_ids = set()

    train_samples = []
    val_samples = []
    test_samples = []

    for lang in languages_to_process:
        lang_data = all_samples.get(lang, [])
        if not lang_data:
            continue

        # 收集该语言所有文件 ID
        lang_file_ids = list(set(fid for _, fid in lang_data))
        random.shuffle(lang_file_ids)

        n = len(lang_file_ids)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        lang_train_ids = set(lang_file_ids[:n_train])
        lang_val_ids = set(lang_file_ids[n_train:n_train + n_val])
        lang_test_ids = set(lang_file_ids[n_train + n_val:])

        for sample, fid in lang_data:
            entry = json.dumps({"text": sample, "label": lang}, ensure_ascii=False)
            if fid in lang_train_ids:
                train_samples.append(entry)
            elif fid in lang_val_ids:
                val_samples.append(entry)
            else:
                test_samples.append(entry)

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    # ── 写入输出 ──
    out_files = {
        "train.jsonl": train_samples,
        "val.jsonl": val_samples,
        "test.jsonl": test_samples,
    }
    for fname, data in out_files.items():
        path = output_dir / fname
        with open(path, 'w', encoding='utf-8') as f:
            for line in data:
                f.write(line + '\n')
        print(f"  写入 {path} ({len(data)} 行)")

    # ── 统计 ──
    print(f"\n{'='*60}")
    print("数据集统计")
    print(f"{'='*60}")
    stats = defaultdict(lambda: defaultdict(int))
    for split_name, split_data in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        for line in split_data:
            entry = json.loads(line)
            stats[split_name][entry["label"]] += 1

    print(f"{'语言':<20} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 52)
    all_langs = sorted(set(list(stats["train"].keys()) + list(stats["val"].keys()) + list(stats["test"].keys())))
    for lang in all_langs:
        t = stats["train"].get(lang, 0)
        v = stats["val"].get(lang, 0)
        te = stats["test"].get(lang, 0)
        print(f"{lang:<20} {t:>8} {v:>8} {te:>8} {t+v+te:>8}")
    print("-" * 52)
    total = sum(sum(s.values()) for s in stats.values())
    print(f"{'TOTAL':<20} {sum(stats['train'].values()):>8} {sum(stats['val'].values()):>8} {sum(stats['test'].values()):>8} {total:>8}")


if __name__ == "__main__":
    main()
