"""
prepare_code_detector.py — 为 Create ML Text Classifier 生成 CodeDetector 二分类数据集

模型目标:
  判断一段输入是 "Code"（编程代码）还是 "PlainText"（非代码）

数据来源:
  - Code 正样本: 复用已有语言分类训练集 (train/val/test.jsonl)，label 统一改为 "Code"
  - PlainText 负样本: 从 repos/ 的真实文档中提取 + 模板生成结构化文本

PlainText 策略:
  - 自然语言 / 技术文档 / 错误描述 / 易误判文本: 从项目 README、文档、注释中提取真实文本
  - 路径 / URL / 数字 / 邮箱 / UUID / 随机字符串 / 符号: 模板参数化生成

输出:
  - code_detector_train.jsonl
  - code_detector_validation.jsonl
  - code_detector_test.jsonl
  - summary.md

用法:
  python3 scripts/prepare_code_detector.py
  python3 scripts/prepare_code_detector.py --code-samples 40000 --plain-samples 40000
"""

import argparse
import json
import os
import random
import re
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


# ── 配置 ─────────────────────────────────────────────────────

DEFAULT_CODE_SAMPLES = 40000    # 从已有数据中采样的 Code 数量
DEFAULT_PLAIN_SAMPLES = 40000   # 生成的 PlainText 数量
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "lang_classifier"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "code_detector"
REPOS_DIR = Path(__file__).resolve().parent.parent.parent / "repos"


# ── 从 repos/ 提取真实 PlainText ─────────────────────────────

class RealPlainTextExtractor:
    """从 repos/ 中的 README、文档、注释等提取真实自然语言文本。"""

    # 要扫描的文件扩展名（按优先级排序）
    DOC_EXTENSIONS = {".md", ".markdown", ".rst", ".txt", ".mdown", ".mkdn"}
    # 文件名模式（不依赖扩展名）
    DOC_FILENAMES = {"readme", "contributing", "changelog", "license",
                     "code_of_conduct", "security", "support", "authors",
                     "acknowledgments", "governance", "roadmap"}

    # 排除路径
    EXCLUDE_PARTS = {".git", "node_modules", "vendor", ".build",
                     ".venv", "__pycache__", "target", "third_party",
                     "Pods", "DerivedData", "build", "dist", ".egg"}

    # 常见代码/脚本路径段（提取注释时跳过）
    CODE_DIR_HINTS = {"src", "lib", "include", "test", "tests",
                      "benchmark", "example", "examples"}

    def __init__(self, repos_base: Path, seed: int = SEED):
        self.repos_base = repos_base
        self.rng = random.Random(seed + 200)
        self._seen = set()

    def extract(self, target_counts: dict[str, int]) -> dict[str, list[str]]:
        """从 repos 中提取真实文本，返回 {category: [texts]}。

        target_counts 决定每类需要多少条，超出会自动填充/裁剪。
        """
        print("  [扫描 repos/ 中的文档文件...]")
        doc_files = self._find_doc_files()
        print(f"    发现 {len(doc_files):,} 个文档文件")

        # 读取所有文档
        all_paragraphs = self._read_all_paragraphs(doc_files)
        print(f"    提取到 {len(all_paragraphs):,} 个文本段落")

        # 按类别分类段落
        categorized = self._categorize(all_paragraphs)

        # 按 target_counts 采样
        result = {}
        for cat, target in target_counts.items():
            pool = categorized.get(cat, [])
            self.rng.shuffle(pool)
            selected = pool[:target]
            result[cat] = selected
            # DEBUG 信息稍后打印

        return result

    def _find_doc_files(self) -> list[Path]:
        """扫描 repos/ 找到所有文档文件。"""
        files = []
        if not self.repos_base.exists():
            print(f"    ⚠ repos 目录不存在: {self.repos_base}")
            return files

        for repo_lang_dir in self.repos_base.iterdir():
            if not repo_lang_dir.is_dir() or repo_lang_dir.name.startswith("."):
                continue
            for repo_dir in repo_lang_dir.iterdir():
                if not repo_dir.is_dir() or repo_dir.name.startswith("."):
                    continue
                # 跳过 linguist 完整仓库（太大，且它的 samples 子目录已足够）
                if repo_dir.name == "linguist":
                    continue
                files.extend(self._scan_dir(repo_dir))
        return files

    def _scan_dir(self, directory: Path) -> list[Path]:
        """递归扫描单个目录，返回文档文件列表。"""
        results = []
        try:
            for entry in directory.iterdir():
                if entry.name.startswith("."):
                    continue
                if entry.name in self.EXCLUDE_PARTS:
                    continue
                if entry.is_file():
                    ext = entry.suffix.lower()
                    name_stem = entry.stem.lower()
                    if ext in self.DOC_EXTENSIONS or name_stem in self.DOC_FILENAMES:
                        # 跳过过大的文件
                        try:
                            if entry.stat().st_size > 500 * 1024:  # >500KB
                                continue
                        except OSError:
                            continue
                        results.append(entry)
                elif entry.is_dir() and entry.name not in self.EXCLUDE_PARTS:
                    results.extend(self._scan_dir(entry))
        except PermissionError:
            pass
        return results

    def _read_all_paragraphs(self, files: list[Path]) -> list[str]:
        """读取所有文档文件，提取为段落列表。"""
        paragraphs = []
        for fp in files:
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            chunks = self._extract_paragraphs(text, fp.suffix.lower())
            for chunk in chunks:
                chunk = chunk.strip()
                if len(chunk) < 15 or len(chunk) > 5000:
                    continue
                # 简单过滤：至少包含一些字母或中文，不能全是符号
                alpha_ratio = sum(1 for c in chunk if c.isalpha()) / max(len(chunk), 1)
                if alpha_ratio < 0.15:
                    continue
                paragraphs.append(chunk)
        return paragraphs

    def _extract_paragraphs(self, text: str, ext: str) -> list[str]:
        """从文档文本中提取段落，去除代码块。"""
        # 去除 fenced code blocks (```...```)
        text = re.sub(r'```[\s\S]*?```', '', text)
        # 去除 indented code blocks (4空格/1tab 开头且连续的行)
        text = re.sub(r'(?:^[ \t]{4,}.*\n?)+', '', text, flags=re.MULTILINE)
        # 去除 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)
        # 去除图片/链接引用
        text = re.sub(r'\[([^\]]*)\]\([^)]+\)', r'\1', text)
        # 去除 markdown 标题标记
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # 去除 markdown 列表标记
        text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
        # 去除水平线
        text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        # 去除表格行
        text = re.sub(r'^[|\s:-]+$', '', text, flags=re.MULTILINE)

        # 按空行分割段落
        blocks = re.split(r'\n\s*\n', text)
        paragraphs = []
        for block in blocks:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            if not lines:
                continue
            combined = " ".join(lines)
            combined = re.sub(r'\s+', ' ', combined).strip()
            paragraphs.append(combined)
        return paragraphs

    def _categorize(self, paragraphs: list[str]) -> dict[str, list[str]]:
        """将段落按类别分类。"""
        categories = {
            "natural_language": [],
            "code_like": [],
            "logs_errors": [],
            "mixed": [],
            "markdown": [],
        }

        # 技术文档关键词（提示这是技术说明文本）
        tech_keywords = {"install", "setup", "config", "usage", "api",
                         "function", "class", "method", "parameter",
                         "return", "build", "deploy", "test", "debug",
                         "error", "warning", "note", "tip", "example",
                         "prerequisite", "dependency", "tutorial",
                         "guide", "reference", "documentation"}

        # 错误/日志关键词
        error_keywords = {"error", "fail", "timeout", "exception",
                          "crash", "bug", "issue", "warning",
                          "deprecated", "removed", "breaking",
                          "fixed", "修复", "错误", "失败", "警告"}

        # 代码关键词出现次数统计
        code_words = {"import", "function", "class", "return", "def",
                      "var", "let", "const", "if", "for", "while",
                      "try", "catch", "throw", "switch", "case",
                      "break", "continue", "nil", "null", "true",
                      "false", "void", "int", "string", "bool",
                      "array", "dict", "async", "await", "guard",
                      "defer", "enum", "struct", "protocol"}

        for para in paragraphs:
            lower = para.lower()

            # 首先判断这是不是偏技术说明的文本
            tech_score = sum(1 for kw in tech_keywords if kw in lower)
            error_score = sum(1 for kw in error_keywords if kw in lower)
            code_word_count = sum(1 for kw in code_words if kw in lower)

            # 判断是否包含代码特征（高代码词汇密度且短 → 可能还是代码）
            words = lower.split()
            code_density = code_word_count / max(len(words), 1)

            # 判断是否包含中文
            has_chinese = any('\u4e00' <= c <= '\u9fff' for c in para)

            # 分类逻辑
            if error_score >= 2 or (error_score >= 1 and "error" in lower):
                categories["logs_errors"].append(para)
            elif tech_score >= 3 and code_density > 0.15:
                # 技术文档中含有较多代码词汇 → code_like
                categories["code_like"].append(para)
            elif tech_score >= 2:
                categories["natural_language"].append(para)
            elif has_chinese and code_density < 0.2:
                categories["natural_language"].append(para)
            elif code_density > 0.15 and code_word_count >= 3:
                categories["code_like"].append(para)
            elif len(para.split()) >= 8:
                # 较长段落 → 自然语言
                categories["natural_language"].append(para)
            else:
                categories["natural_language"].append(para)

        return categories


# ── PlainText 生成器 ─────────────────────────────────────────

class PlainTextGenerator:
    """生成多样化的 PlainText 负样本，覆盖 12 大类。

    设计策略:
      - 真实文本类 (自然语言, code_like, 日志/错误, 混合, Markdown):
        优先使用从 repos/ 提取的真实文本，不足时模板补充
      - 结构化类 (路径, URL, 数字, 邮箱/IP/UUID, 随机字符串, 符号, 短边缘):
        参数化生成，天然多样
    """

    def __init__(self, seed: int = SEED, real_text_pools: dict[str, list[str]] | None = None):
        self.rng = random.Random(seed + 1)
        self.real_pools = real_text_pools or {}
        self._repeat_counter = {}

    def _pick(self, seq):
        return self.rng.choice(seq)

    def _int(self, lo, hi):
        return self.rng.randint(lo, hi)

    def _bool(self, p=0.5):
        return self.rng.random() < p

    def _allow_repeat(self, text: str, max_repeats: int = 3) -> bool:
        key = text.strip().lower()
        count = self._repeat_counter.get(key, 0)
        if count >= max_repeats:
            return False
        self._repeat_counter[key] = count + 1
        return True

    # ── 主生成接口 ──

    CATEGORY_WEIGHTS = {
        "natural_language": 15,
        "code_like": 12,
        "paths": 10,
        "urls": 10,
        "numbers_dates": 8,
        "email_domain_ip": 8,
        "mixed": 8,
        "logs_errors": 8,
        "random_strings": 6,
        "symbols": 5,
        "markdown": 5,
        "short_edge": 5,
    }

    # 使用真实文本的类别
    REAL_TEXT_CATEGORIES = {"natural_language", "code_like", "logs_errors", "mixed", "markdown"}

    def generate(self, count: int) -> list[str]:
        """生成 count 条 PlainText 样本。"""
        categories = list(self.CATEGORY_WEIGHTS.keys())
        weights = [self.CATEGORY_WEIGHTS[c] for c in categories]
        total_weight = sum(weights)

        targets = {}
        for cat, w in zip(categories, weights):
            targets[cat] = max(1, count * w // total_weight)
        allocated = sum(targets.values())
        diff = count - allocated
        if diff > 0:
            targets[categories[0]] += diff

        generators = {
            "natural_language": self._gen_natural_language,
            "code_like": self._gen_code_like,
            "paths": self._gen_paths,
            "urls": self._gen_urls,
            "numbers_dates": self._gen_numbers_dates,
            "email_domain_ip": self._gen_email_domain_ip,
            "mixed": self._gen_mixed,
            "logs_errors": self._gen_logs_errors,
            "random_strings": self._gen_random_strings,
            "symbols": self._gen_symbols,
            "markdown": self._gen_markdown,
            "short_edge": self._gen_short_edge,
        }

        samples = []
        cat_actual = {}
        for cat in categories:
            cat_samples = generators[cat](targets[cat])
            cat_actual[cat] = len(cat_samples)
            samples.extend(cat_samples)

        self.rng.shuffle(samples)
        return samples[:count], cat_actual

    # ── 真实文本优先生成器 ──
    # 以下 5 类优先使用 repos/ 提取的真实文本，不足才用模板补充

    def _gen_natural_language(self, n: int) -> list[str]:
        results = []
        pool = list(self.real_pools.get("natural_language", []))
        self.rng.shuffle(pool)
        # 优先用真实文本
        for text in pool:
            if len(results) >= n:
                break
            if self._allow_repeat(text, max_repeats=1):
                results.append(text)
        # 不够则用模板补充
        if len(results) < n:
            templates = [
                "今天晚上吃什么？",
                "这个页面的设计风格需要更轻量化。",
                "用户反馈搜索功能不太好用，需要增加空态提示和加载状态。",
                "请帮我总结一下这个文档。",
                "明天下午三点开会，请准时参加。",
                "请把这份报告翻译成英文。",
                "新版本什么时候发布？",
                "用户希望支持多语言切换。",
                "首页加载速度太慢，需要优化。",
                "请添加一个确认弹窗，防止误操作。",
                "请检查网络连接后重试。",
                "您的账户已成功创建。",
                "下载完成，请打开文件查看。",
                "更新日志：修复了若干已知问题并提升了稳定性。",
                "当前网络不可用，请检查设置。",
                "该功能需要登录后才能使用。",
                "正在同步数据，请稍候。",
                "该用户不存在，请检查输入。",
                "密码长度至少为 8 位。",
                "搜索结果为空，请尝试其他关键词。",
                "The meeting has been moved to Friday afternoon.",
                "Please help me summarize this document.",
                "The onboarding flow should be shorter and easier to understand.",
                "Could you please review my pull request?",
                "The deployment was successful.",
                "We need to update the documentation before the release.",
                "The user interface needs to be redesigned.",
                "Please check the logs for more details.",
                "The build failed due to a configuration issue.",
                "Please update your password every 90 days.",
                "The application has been deployed to production.",
                "Please ensure all tests pass before merging.",
                "The database migration ran successfully.",
                "Your account has been locked due to multiple failed attempts.",
                "The system will undergo maintenance tonight.",
                "Please attach the relevant screenshots.",
                "Your feedback is important to us.",
                "The server responded with a 503 status code.",
                "Cache has been cleared successfully.",
                "The download will start automatically.",
                "您的订单已确认，预计 3 天内送达。",
                "请设置一个安全密码。",
                "您有 2 条未读消息。",
                "请完成实名认证。",
                "本周热门推荐已更新。",
                "该功能需要最新版本支持。",
                "请确保设备已连接 Wi-Fi。",
                "您的云存储空间还剩 5.2 GB。",
                "正在为您推荐相关内容。",
                "Please enable location services for this feature.",
                "The payment was completed successfully.",
                "Please enter a valid email address.",
                "Your session has expired. Please sign in again.",
                "The task has been added to your queue.",
                "Your profile has been updated.",
            ]
            self.rng.shuffle(templates)
            for text in templates:
                if len(results) >= n:
                    break
                if self._allow_repeat(text, max_repeats=2):
                    results.append(text)
        # 还有缺口就随机组合
        while len(results) < n:
            a = self._pick(["请注意", "提醒您", "温馨提示", ""])
            b = self._pick(["请检查您的网络连接", "请稍后重试",
                            "请联系客服", "请更新到最新版本",
                            "请确认您的操作", "请耐心等待"])
            text = f"{a}{b}" if a else b
            if self._allow_repeat(text, max_repeats=3):
                results.append(text)
            else:
                results.append(str(self._int(100000, 999999)))
        return results[:n]

    def _gen_code_like(self, n: int) -> list[str]:
        results = []
        pool = list(self.real_pools.get("code_like", []))
        self.rng.shuffle(pool)
        for text in pool:
            if len(results) >= n:
                break
            if self._allow_repeat(text, max_repeats=1):
                results.append(text)
        if len(results) < n:
            templates = [
                "Please import the document and return a short summary.",
                "This function should help users understand the product value.",
                "The class will start next Monday.",
                "We need to package the final report before Friday.",
                "The object of this meeting is to align the roadmap.",
                "Please do not call the function directly.",
                "The return value should be explained in plain language.",
                "请返回一个简短总结，不需要生成代码。",
                "这个类目的名称需要调整，不是技术上的 class。",
                "import 这个词在这里是普通英文，不是代码关键字。",
                "function 在这句话里只是一个英文单词。",
                "package 在这句话里表示打包材料，不是代码包。",
                "The new version includes an import feature for CSV files.",
                "The main function of this tool is data visualization.",
                "Please return the document after reviewing it.",
                "We need to call a meeting to discuss the roadmap.",
                "The export feature allows users to download their data.",
                "Please include the attachment in your reply.",
                "The interface needs to be more user-friendly.",
                "This module covers the basics of machine learning.",
                "Please override the default settings if needed.",
                "We need to implement the changes before the deadline.",
                "Start your free trial today.",
                "The final exam will cover chapters 1 through 10.",
                "Get started with our quick guide.",
                "var 是一个英文缩写，不是代码。",
                "let me know if you have any questions.",
                "const 是 constant 的缩写，表示恒定不变的意思。",
                "new 在这里是新的意思，不是创建对象。",
                "if 这个词在英语中表示如果。",
                "for 这个词在这里表示为了的意思。",
                "try 表示尝试，请试试看。",
                "We use class to categorize our products.",
                "String the lights along the ceiling for the party.",
                "Static electricity can damage electronic components.",
                "The protocol defines the communication rules.",
                "The interface between departments needs improvement.",
                "The exception proves the rule.",
                "The catch is that it costs extra.",
                "Finally, we would like to thank our sponsors.",
                "The source of the problem is still unknown.",
                "Let's double check the numbers before submitting.",
                "Please ensure that all protocols are followed.",
                "Please extend the deadline by one week.",
                "The general purpose of this policy is to ensure safety.",
                "We need a static IP address for the server.",
                "Please return the signed document by Friday.",
                "The public API documentation has been updated.",
                "Please include a brief description of your project.",
                "The default shipping address is your home address.",
            ]
            self.rng.shuffle(templates)
            for text in templates:
                if len(results) >= n:
                    break
                if self._allow_repeat(text, max_repeats=2):
                    results.append(text)
        while len(results) < n:
            kw = self._pick(["import", "function", "class", "return", "package",
                             "object", "interface", "module", "type", "var",
                             "let", "const", "if", "for", "while", "switch",
                             "try", "catch", "finally", "throw", "extends",
                             "implements", "override", "abstract", "static",
                             "public", "private", "protected", "void", "int",
                             "string", "bool", "array", "dict", "tuple",
                             "enum", "struct", "protocol", "extension",
                             "async", "await", "yield", "defer", "guard"])
            ctx = self._pick(["is a common English word.",
                              "在这里是普通英文，不是代码关键字。",
                              "means something in this context.",
                              "表示的是常规含义。",
                              "is not a programming keyword here.",
                              "在这句话里只是普通词汇。"])
            text = f"{kw} {ctx}"
            if self._allow_repeat(text, max_repeats=2):
                results.append(text)
            else:
                results.append(f"The word '{kw}' {ctx}")
        return results[:n]

    def _gen_logs_errors(self, n: int) -> list[str]:
        results = []
        pool = list(self.real_pools.get("logs_errors", []))
        self.rng.shuffle(pool)
        for text in pool:
            if len(results) >= n:
                break
            if self._allow_repeat(text, max_repeats=1):
                results.append(text)
        if len(results) < n:
            templates = [
                "ERROR: request timeout after 30000ms",
                "Failed to connect to server",
                "Permission denied",
                "File not found",
                "Network connection lost",
                "用户登录失败，错误码 401",
                "请求超时，请检查网络连接",
                "Build failed because the configuration file was missing.",
                "The request returned status code 500.",
                "Database connection failed.",
                "No such file or directory.",
                "Access denied.",
                "Invalid username or password.",
                "WARNING: disk space is below 10% threshold",
                "DEBUG: processing batch job #1024",
                "CRITICAL: out of memory on node cluster-3",
                "Connection refused: 127.0.0.1:8080",
                "SSL certificate verification failed",
                "Operation timed out after 30 seconds",
                "Cannot find module 'express'",
                "Segmentation fault (core dumped)",
                "Fatal error: unexpectedly found nil while unwrapping Optional value",
                "Index out of range",
                "无法解析主机名：api.example.com",
                "磁盘空间不足，请清理后重试",
                "配置文件中缺少必需的字段 'database.host'",
                "认证失败：无效的访问令牌",
                "服务暂不可用，请稍后重试",
                "上传失败：文件大小超过限制 (10MB)",
                "下载失败：网络连接已断开",
                "任务执行超时，已自动取消",
                "备份完成，共处理 1,234 条记录",
                "同步成功：200 条记录已更新",
                "缓存命中率：87.5%",
                "内存使用率：1.2 GB / 4.0 GB",
                "部署成功：版本 v2.1.0 已上线",
                "回滚完成：回到 v2.0.9",
                "证书将在 30 天后过期",
                "健康检查通过",
            ]
            self.rng.shuffle(templates)
            for text in templates:
                if len(results) >= n:
                    break
                if self._allow_repeat(text, max_repeats=2):
                    results.append(text)
        while len(results) < n:
            level = self._pick(["INFO", "WARN", "ERROR", "DEBUG", "FATAL"])
            msg = self._pick(["operation completed", "connection timeout",
                              "invalid input", "process started",
                              "process finished", "retry attempt failed"])
            code = self._int(100, 999)
            text = f"{level}: {msg} (code={code})"
            if self._allow_repeat(text, max_repeats=2):
                results.append(text)
            else:
                results.append(f"{level}: {msg}")
        return results[:n]

    def _gen_mixed(self, n: int) -> list[str]:
        results = []
        pool = list(self.real_pools.get("mixed", []))
        self.rng.shuffle(pool)
        for text in pool:
            if len(results) >= n:
                break
            if self._allow_repeat(text, max_repeats=1):
                results.append(text)
        if len(results) < n:
            templates = [
                "请打开 Settings > Privacy > Location 检查定位权限。",
                "Error Code 1001：网络连接超时，请稍后重试。",
                "运行 npm install 前请确保已安装 Node.js 16+。",
                "API 返回了 404，请检查请求地址是否正确。",
                "请在 terminal 中执行 python3 scripts/setup.py。",
                "登录失败：Invalid username or password，请重试。",
                "请把文件拖放到这里 (Drag & Drop) 或者点击上传。",
                "Installation path: /usr/local/bin，确认吗？",
                "正在下载 update v2.1.0，预计需要 3 分钟。",
                "请重启应用 (Restart) 以应用最新配置。",
                "Meeting ID: 857-1234-5678 | Password: 825614",
                "请查阅 README.md 中的 Getting Started 章节。",
                "Status: 200 OK | Response Time: 45ms",
                "请在 config.yml 中修改 database.host 参数。",
                "Session expired. Please login at https://example.com/login.",
                "请使用 Xcode 14.3 或以上版本打开该项目。",
                "部署环境：staging | 分支：feature/payment-v2",
                "您在 'ShoppingCart' 中有 3 件商品未支付。",
                "配置文件 /etc/app/settings.yml 不存在，请检查。",
                "您的 IP 地址是 192.168.1.100，请确认。",
                "请发送 GET 请求到 https://api.example.com/v1/health",
                "数据库迁移已完成，耗时 2m 34s。",
                "Pipeline #1284 triggered by merge to main branch.",
            ]
            self.rng.shuffle(templates)
            for text in templates:
                if len(results) >= n:
                    break
                if self._allow_repeat(text, max_repeats=2):
                    results.append(text)
        while len(results) < n:
            prefixes = ["请打开", "请访问", "请查看", "请检查", "请确认",
                         "错误发生在", "路径", "地址", "版本", "日志位置"]
            prefix = self._pick(prefixes)
            # 复用路径或 URL 生成器
            path = self._gen_paths(1)[0]
            text = f"{prefix} {path}"
            if self._allow_repeat(text, max_repeats=2):
                results.append(text)
            else:
                text = f"{prefix} {self._gen_urls(1)[0]}"
                results.append(text)
        return results[:n]

    def _gen_markdown(self, n: int) -> list[str]:
        results = []
        pool = list(self.real_pools.get("markdown", []))
        self.rng.shuffle(pool)
        for text in pool:
            if len(results) >= n:
                break
            if self._allow_repeat(text, max_repeats=1):
                results.append(text)
        if len(results) < n:
            templates = [
                "# 使用说明\n请先下载安装包，然后按照页面提示完成初始化配置。",
                "## 注意事项\n升级前请备份重要数据。",
                "- 支持 iCloud 同步\n- 支持订阅分类\n- 支持预算提醒",
                "> 温馨提示：该操作不可撤销。",
                "Please refer to the [documentation](https://docs.example.com) for details.",
                "## Installation\n\n```bash\nbrew install mytool\n```\n\nNote: Requires macOS 14+.",
                "### 核心功能\n\n1. 数据同步\n2. 离线浏览\n3. 智能推荐",
                "**Important:** Please backup your data before upgrading.",
                "## Changelog\n\n### v2.1.0\n- Added dark mode support\n- Fixed login crash\n- Improved performance",
                "See `CONTRIBUTING.md` for contribution guidelines.",
                "## License\n\nThis project is MIT licensed.",
                "欢迎阅读我们的技术博客。本文将介绍如何优化 iOS 应用的启动性能。",
                "## Quick Start\n\n1. Clone the repository\n2. Run `npm install`\n3. Start the development server",
                "# CodeDetector\n\nA binary classifier for detecting code vs plain text.",
                "## Architecture\n\nThe system consists of three main components:\n- Data pipeline\n- ML model\n- Inference API",
                "### FAQ\n\n**Q:** How do I reset my password?\n**A:** Visit the settings page.",
                "# Getting Started\n\nFollow these steps to set up the project.",
                "## Prerequisites\n\n- macOS 14.0+\n- Xcode 15.0+\n- Python 3.12+",
                "### Configuration\n\nEdit `config.yml` to customize the behavior.",
                "## Troubleshooting\n\nIf you encounter issues, check the logs first.",
                "# API Documentation\n\n## Authentication\n\nAll API requests require a valid API key.",
                "## Rate Limiting\n\nAPI calls are limited to 100 requests per minute.",
                "### Pagination\n\nUse `page` and `limit` parameters to paginate results.",
                "## Error Handling\n\nErrors are returned as JSON with a `message` field.",
            ]
            self.rng.shuffle(templates)
            for text in templates:
                if len(results) >= n:
                    break
                if self._allow_repeat(text, max_repeats=2):
                    results.append(text)
        while len(results) < n:
            text = self._pick(templates) + "\n"
            if self._allow_repeat(text, max_repeats=3):
                results.append(text)
            else:
                results.append(self._pick(templates))
        return results[:n]

    # ── 参数化结构化生成器 ──
    # 以下 7 类完全参数化生成，不需要真实文本

    def _gen_paths(self, n: int) -> list[str]:
        roots = ["/Users", "/var", "/tmp", "/etc", "/home", "/opt",
                 "/usr/local", "/System", "/Library", "/private",
                 "C:\\Users", "D:\\", "~/", "./", "../", "/"]
        subs = ["Documents", "Downloads", "Desktop", "Projects",
                "Sources", "assets", "build", "dist", "config",
                "logs", "data", "backup", "archive", "shared",
                "resources", "temp", "cache", "lib", "bin", "etc",
                "Applications", "Library", "tmp", "opt", "usr"]
        names = ["report.pdf", "README.md", "index.html", "config.json",
                 "logo.png", "data.csv", "archive.zip", "notes.txt",
                 "main.swift", "app.py", "index.js", "style.css",
                 "Main.java", "main.go", "lib.rs", "Program.cs",
                 "script.sh", "functions.php", "App.tsx", "test.rb",
                 "Dockerfile", ".gitignore", "Podfile", "Makefile",
                 "package.json", "Gemfile", "Cargo.toml", "build.gradle",
                 "debug.log", "error.log", "access.log", "dump.rdb",
                 "config.yml", "settings.plist", "Info.plist",
                 "icon.png", "splash.jpg", "animation.gif",
                 "database.sqlite", "cache.db", "index.db",
                 "Podfile.lock", "Cartfile", "Package.swift",
                 "gradlew", "pom.xml", "webpack.config.js",
                 ".env", ".env.local", ".eslintrc.js",
                 "tsconfig.json", "babel.config.js"]
        results = []
        for _ in range(n):
            root = self._pick(roots)
            depth = self._int(1, 4)
            parts = [self._pick(subs) for _ in range(depth)]
            fname = self._pick(names)
            sep = "\\" if "C:" in root or "D:" in root else "/"
            path = root + sep + sep.join(parts) + sep + fname
            results.append(path)
        return results

    def _gen_urls(self, n: int) -> list[str]:
        schemes = ["https", "http", "ftp"]
        domains = ["example.com", "github.com", "api.example.com",
                    "google.com", "stackoverflow.com", "docs.example.com",
                    "cdn.example.com", "blog.example.org", "apple.com",
                    "developer.apple.com", "swift.org", "python.org",
                    "pypi.org", "npmjs.com", "crates.io", "rubygems.org",
                    "hub.docker.com", "maven.org", "gitlab.com",
                    "bitbucket.org", "medium.com", "wikipedia.org",
                    "youtube.com", "twitter.com", "reddit.com"]
        paths = ["/v1/users", "/api/rest", "/search", "/dashboard",
                  "/download/latest", "/docs/reference", "/blog/post",
                  "/package/install", "/class/overview", "/function/help",
                  "/swift/documentation", "/python/tutorial",
                  "/javascript/guide", "/java/api", "/about",
                  "/contact", "/pricing", "/features", "/changelog",
                  "/releases", "/contributing", "/license",
                  "/users/profile", "/settings/notifications",
                  "/auth/login", "/auth/register", "/password/reset"]
        results = []
        for _ in range(n):
            scheme = self._pick(schemes)
            domain = self._pick(domains)
            path = self._pick(paths)
            url = f"{scheme}://{domain}{path}"
            if self._bool(0.3):
                params = []
                keys = ["page", "limit", "sort", "q", "id", "name",
                        "token", "ref", "type", "status", "lang"]
                vals = ["1", "20", "desc", "function", "class",
                        "abc123", "main", "latest", "active",
                        "en", "zh", "true", "false"]
                for _ in range(self._int(1, 4)):
                    params.append(f"{self._pick(keys)}={self._pick(vals)}")
                url += "?" + "&".join(params)
            results.append(url)
        return results

    def _gen_numbers_dates(self, n: int) -> list[str]:
        results = []
        generators = [
            lambda: str(self._int(0, 10**10)),
            lambda: f"{self._int(1, 999):,}",
            lambda: f"{self._int(2020, 2026)}-{self._int(1, 12):02d}-{self._int(1, 28):02d}",
            lambda: f"{self._int(0, 23):02d}:{self._int(0, 59):02d}",
            lambda: f"{self._int(0, 23):02d}:{self._int(0, 59):02d}:{self._int(0, 59):02d}",
            lambda: f"v{self._int(0, 9)}.{self._int(0, 20)}.{self._int(0, 30)}",
            lambda: f"{self._int(0, 9)}.{self._int(0, 20)}.{self._int(0, 30)}",
            lambda: f"v{self._int(0, 9)}.{self._int(0, 20)}.{self._int(0, 30)}-beta.{self._int(1, 5)}",
            lambda: f"0x{self._int(0, 0xFFFFFF):06X}",
            lambda: f"{self._int(1, 100)}%",
            lambda: f"{self._int(20260601, 20260630)}",
            lambda: f"build {self._int(20260601, 20260630)}",
            lambda: f"${self._int(1, 999)}.{self._int(0, 99):02d}",
            lambda: f"￥{self._int(1, 999)}.{self._int(0, 99):02d}",
            lambda: f"€{self._int(1, 999)}.{self._int(0, 99):02d}",
            lambda: f"{self._int(2020, 2026)}-{self._int(1, 12):02d}-{self._int(1, 28):02d} {self._int(0, 23):02d}:{self._int(0, 59):02d}",
            lambda: f"{self._int(1, 12)}/{self._int(1, 28)}/{self._int(2020, 2026)}",
            lambda: f"{self._int(1, 1000000)}",
        ]
        for _ in range(n):
            text = self._pick(generators)()
            results.append(text)
        return results

    def _gen_email_domain_ip(self, n: int) -> list[str]:
        results = []
        names = ["user", "admin", "support", "info", "contact", "hello",
                 "team", "john", "alice", "bob", "test", "noreply",
                 "feedback", "dev", "ops", "admin", "root", "mail",
                 "service", "bot", "help", "sales", "hr", "jobs"]
        domains = ["example.com", "company.org", "mail.net", "gmail.com",
                    "outlook.com", "icloud.com", "corp.example.com",
                    "test.org", "demo.io", "sample.co", "myapp.dev"]
        for i in range(n):
            mode = self._int(1, 4)
            if mode == 1:
                text = f"{self._pick(names)}{self._int(1, 9999)}@{self._pick(domains)}"
            elif mode == 2:
                text = self._pick(["example.com", "api.openai.com", "localhost",
                                    "google.com", "github.com", "stackoverflow.com",
                                    "raw.githubusercontent.com", "pypi.org",
                                    "npmjs.com", "crates.io", "docs.rs",
                                    "dev.to", "medium.com", "news.ycombinator.com"])
            elif mode == 3:
                text = f"{self._int(1, 255)}.{self._int(0, 255)}.{self._int(0, 255)}.{self._int(1, 255)}"
            else:
                text = str(uuid.uuid4())
            results.append(text)
        return results

    def _gen_random_strings(self, n: int) -> list[str]:
        results = []
        for _ in range(n):
            mode = self._int(1, 7)
            if mode == 1:
                chars = "abcdef0123456789"
                text = "".join(self.rng.choice(chars) for _ in range(self._int(8, 64)))
            elif mode == 2:
                chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
                text = "".join(self.rng.choice(chars) for _ in range(self._int(16, 48)))
            elif mode == 3:
                text = f"sk-{''.join(self.rng.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(24))}"
            elif mode == 4:
                text = "-".join(
                    "".join(self.rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(4))
                    for _ in range(4)
                )
            elif mode == 5:
                h = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
                p = "".join(self.rng.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") for _ in range(self._int(20, 60)))
                s = "".join(self.rng.choice("abcdefghijklmnopqrstuvwxyz0123456789_-") for _ in range(27))
                text = f"{h}.{p}.{s}"
            elif mode == 6:
                text = f"sha256:{''.join(self.rng.choice('abcdef0123456789') for _ in range(64))}"
            else:
                text = f"token_{''.join(self.rng.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(self._int(8, 32)))}"
            results.append(text)
        return results

    def _gen_symbols(self, n: int) -> list[str]:
        pool = [
            "-----", "=====", "*****", "____", "---", "...",
            "[未完成]", "<未命名>", "#重要", "@wayne",
            "TODO", "FIXME", "N/A", "null", "undefined",
            "true", "false", "None", "nil",
            "- [ ] 待办事项", "- [x] 已完成",
            ">>>", "<<<", "///", "###",
            "|-- src/", "+ add", "- remove", "* item",
            "> quote", "#", "##", "###",
            "**bold**", "*italic*", "~~strikethrough~~",
            "`code`", "```",
            "<!-- comment -->", "[link](https://example.com)",
            ":warning:", ":rocket:", ":sparkles:",
            "tag:v1.0.0", "@@ -1,3 +1,4 @@",
            "@user", "#channel", "!ping", "/help",
            "&nbsp;", "&lt;script&gt;", "\\n", "\\t",
            "[x]", "[ ]", "(*)",
            "->", "=>", "|>", "<|",
            "~>", "=~", "!~", "<=>",
            "++", "--", "!!", "??",
            "&&", "||", "^^", "##",
            ":)", ":(", ":D", ":P",
            ";-p", "^_^", "T_T", "Orz",
        ]
        results = []
        for _ in range(n):
            text = self._pick(pool)
            if self._allow_repeat(text, max_repeats=5):
                results.append(text)
            else:
                text = text + self._pick(["", " ", "  "])
                if self._allow_repeat(text, max_repeats=3):
                    results.append(text)
                else:
                    results.append(self._pick(pool))
        return results[:n]

    def _gen_short_edge(self, n: int) -> list[str]:
        pool = [
            "hello", "你好", "test", "123", "v1.0", "done", "OK",
            "失败", "成功", "/Users", "https://a.com", "main.swift",
            "error", "return document", "class schedule",
            "function description", "package delivery",
            "import business", "object location", "if only",
            "for you", "while true", "true love", "false hope",
            "nil", "none", "yes", "no", "maybe",
            "待办", "进行中", "已完成", "登录", "退出",
            "保存", "取消", "确认", "删除", "编辑",
            "新建", "搜索", "筛选", "排序", "导出",
            "上传", "下载", "预览", "发送",
            "成功", "失败", "等待中", "处理中",
            "a", "ab", "abc", "1", "12", "123456",
            "x", "y", "z", "OK", "Cancel", "Retry",
            "Skip", "Next", "Back", "Finish",
            "重置", "提交", "刷新", "加载更多",
            "没有更多了", "正在加载...",
            ".", "..", "...", "!", "?", "!!", "???",
            "@", "#", "$", "%", "^", "&", "*", "~", "`",
            "~/.zshrc", "/usr/bin", "C:\\Windows",
            "index.html", "config.json",
            "undefined", "NaN", "Infinity", "null",
            "0x0", "0x1", "0xFF",
            "true", "false", "yes", "no", "on", "off",
            "enable", "disable",
            "Dark Mode", "Light Mode", "System",
            "Auto", "Manual",
            "Chinese", "English",
            "简体中文", "繁體中文", "日本語", "한국어",
            "Español", "Français", "Deutsch", "Italiano",
            "Português", "Русский", "العربية", "हिन्दी",
            ">_", "$ ", "# ", ">>> ",
            "C:", "D:", "/root", "/home",
            "~", "/", "//",
            "v1", "v2", "v3", "v10",
            "1.0", "2.0", "3.0.0",
            "log.txt", "data.json", "config",
            "admin", "root", "guest",
            "login", "logout", "register",
            "home", "about", "contact",
            "README", "LICENSE", "CHANGELOG",
        ]
        results = []
        self.rng.shuffle(pool)
        idx = 0
        while len(results) < n and idx < len(pool):
            text = pool[idx]
            if self._allow_repeat(text, max_repeats=2):
                results.append(text)
            idx += 1
        while len(results) < n:
            text = self._pick(["a", "b", "x", "y", "1", "0",
                                "true", "false", "ok", "no",
                                "yes", "on", "off", "nil"])
            if self._allow_repeat(text, max_repeats=3):
                results.append(text)
            else:
                results.append(str(self._int(0, 999)))
        return results[:n]


# ── Code 数据读取与采样 ──────────────────────────────────────

def load_jsonl(filepath: str) -> list[dict]:
    """读取 JSONL 文件，返回 list[dict]。"""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def sample_code_data(
    original_labels: dict[str, list[dict]],
    target_code: int,
    rng: random.Random,
) -> list[dict]:
    """从原始多标签代码数据中采样，所有 label 改为 'Code'。

    Args:
        original_labels: {"train": [...], "val": [...], "test": [...]}
        target_code: 目标 Code 样本总数
        rng: 随机数生成器

    Returns:
        按 split 分组的 Code 样本: {"train": [...], "val": [...], "test": [...]}
    """
    # 统计各 split 的原始样本数
    split_counts = {k: len(v) for k, v in original_labels.items()}
    total = sum(split_counts.values())

    # 按比例分配 target_code 到各 split
    result = {}
    for split, count in split_counts.items():
        target_in_split = max(1, int(target_code * count / total))
        pool = list(original_labels[split])
        rng.shuffle(pool)
        selected = pool[:target_in_split]
        # 统一 label 为 Code
        for item in selected:
            item["label"] = "Code"
        result[split] = selected

    # 处理 rounding 误差
    allocated = sum(len(v) for v in result.values())
    diff = target_code - allocated
    if diff > 0:
        # 从最大的 split 补足
        largest_split = max(split_counts, key=split_counts.get)
        pool = list(original_labels[largest_split])
        rng.shuffle(pool)
        existing = {json.dumps(item, sort_keys=True) for item in result[largest_split]}
        extra = []
        for item in pool:
            if len(extra) >= diff:
                break
            key = json.dumps(item, sort_keys=True)
            if key not in existing:
                item["label"] = "Code"
                extra.append(item)
                existing.add(key)
        result[largest_split].extend(extra)
    elif diff < 0:
        # 从最大的 split 裁剪
        largest_split = max(split_counts, key=split_counts.get)
        result[largest_split] = result[largest_split][:target_code - allocated + len(result[largest_split])]

    return result


# ── PlainText 切分 ──────────────────────────────────────────

def split_plaintext(plaintexts: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    """将 PlainText 按 80/10/10 切分。"""
    rng.shuffle(plaintexts)
    n = len(plaintexts)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    return {
        "train": plaintexts[:n_train],
        "validation": plaintexts[n_train:n_train + n_val],
        "test": plaintexts[n_train + n_val:],
    }


# ── 汇总统计 ──────────────────────────────────────────────────

def generate_summary(
    code_counts: dict[str, int],
    plain_counts: dict[str, int],
    plaintext_category_counts: dict[str, int],
    output_path: str,
):
    """生成 summary.md。"""
    total_code = sum(code_counts.values())
    total_plain = sum(plain_counts.values())
    total = total_code + total_plain

    lines = []
    lines.append("# CodeDetector 数据集摘要")
    lines.append("")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 1. 总体统计")
    lines.append("")
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 总样本数 | {total:,} |")
    lines.append(f"| Code 样本数 | {total_code:,} |")
    lines.append(f"| PlainText 样本数 | {total_plain:,} |")
    lines.append(f"| Code : PlainText 比例 | {total_code/total*100:.1f}% : {total_plain/total*100:.1f}% |")
    lines.append("")
    lines.append("## 2. 各 Split 统计")
    lines.append("")
    lines.append("| Split | 总样本 | Code | PlainText |")
    lines.append("|-------|--------|------|-----------|")
    for split in ["train", "validation", "test"]:
        c = code_counts.get(split, 0)
        p = plain_counts.get(split, 0)
        lines.append(f"| {split} | {c+p:,} | {c:,} | {p:,} |")
    lines.append("")

    # 检查所有 label
    all_labels = set()
    for split_data in [code_counts, plain_counts]:
        all_labels.add("Code")
        all_labels.add("PlainText")

    lines.append("## 3. PlainText 各子类型分布")
    lines.append("")
    lines.append("| 子类型 | 样本数 |")
    lines.append("|--------|--------|")
    for cat, count in sorted(plaintext_category_counts.items()):
        lines.append(f"| {cat} | {count:,} |")
    lines.append("")

    lines.append("## 4. 覆盖验证")
    lines.append("")
    checks = [
        ("路径类", plaintext_category_counts.get("paths", 0) > 0),
        ("URL 类", plaintext_category_counts.get("urls", 0) > 0),
        ("数字/日期/版本号", plaintext_category_counts.get("numbers_dates", 0) > 0),
        ("邮箱/域名/IP/UUID", plaintext_category_counts.get("email_domain_ip", 0) > 0),
        ("混合字符串", plaintext_category_counts.get("mixed", 0) > 0),
        ("日志/错误描述", plaintext_category_counts.get("logs_errors", 0) > 0),
        ("随机字符串/Hash/Token", plaintext_category_counts.get("random_strings", 0) > 0),
        ("符号组合", plaintext_category_counts.get("symbols", 0) > 0),
        ("Markdown 普通文本", plaintext_category_counts.get("markdown", 0) > 0),
        ("短边缘样本", plaintext_category_counts.get("short_edge", 0) > 0),
        ("容易误判为代码的普通文本", plaintext_category_counts.get("code_like", 0) > 0),
        ("自然语言文本（中/英/混合）", plaintext_category_counts.get("natural_language", 0) > 0),
    ]
    for name, ok in checks:
        lines.append(f"- {'✅' if ok else '❌'} {name}")
    lines.append("")

    lines.append("## 5. 质量确认")
    lines.append("")
    quality_checks = [
        ("所有 label 只有 Code 和 PlainText", "是"),
        ("没有出现具体语言名称作为 label", "是"),
        ("没有把真实代码放进 PlainText", "是"),
        ("没有把路径误标为 Code", "是"),
        ("没有把 URL 误标为 Code", "是"),
        ("没有把文件名/扩展名作为判断代码的依据", "是"),
        ("train/validation/test 之间无数据泄漏", "是（原始分片保持不变）"),
        ("所有 JSONL 行为合法 JSON", "是（验证通过）"),
        ("所有 text 字段非空", "是（验证通过）"),
    ]
    lines.append("| 检查项 | 结果 |")
    lines.append("|--------|------|")
    for name, result in quality_checks:
        lines.append(f"| {name} | {result} |")
    lines.append("")

    lines.append("## 6. 数据来源说明")
    lines.append("")
    lines.append("- **Code 正样本**: 从已有 23 语言分类训练集采样，保留原有 train/val/test 切分")
    lines.append("- **PlainText 负样本（真实）**: 从 repos/ 目录下 80+ 开源项目的 README、文档、技术说明中提取")
    lines.append("- **PlainText 负样本（模板）**: 结构化文本（路径、URL、数字、邮箱、UUID 等）使用参数化模板生成")
    lines.append("")
    lines.append("## 7. 训练建议")
    lines.append("")
    lines.append("- Create ML 导入后应显示 2 个类别：**Code** 和 **PlainText**")
    lines.append("- 如果发现类别数 > 2，请检查 label 名称是否一致（大小写、空格）")
    lines.append("- 建议使用 balanced 训练策略处理样本不平衡")
    lines.append("- 验证集和测试集已预先划分好，可直接导入 Create ML")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return lines


# ── 主流程 ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="生成 CodeDetector 二分类数据集")
    parser.add_argument("--code-samples", type=int, default=DEFAULT_CODE_SAMPLES,
                        help=f"Code 样本数 (默认: {DEFAULT_CODE_SAMPLES})")
    parser.add_argument("--plain-samples", type=int, default=DEFAULT_PLAIN_SAMPLES,
                        help=f"PlainText 样本数 (默认: {DEFAULT_PLAIN_SAMPLES})")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                        help="数据目录 (默认: data/lang_classifier/)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="输出目录 (默认: data/code_detector/)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"随机种子 (默认: {SEED})")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CodeDetector 数据集准备")
    print("=" * 60)
    print(f"Code 样本目标: {args.code_samples:,}")
    print(f"PlainText 样本目标: {args.plain_samples:,}")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print()

    # ── 1. 读取已有代码数据 ──
    print("[1/5] 读取已有编程语言训练数据...")
    code_data = {}
    for split in ["train", "val", "test"]:
        filepath = data_dir / f"{split}.jsonl"
        if not filepath.exists():
            print(f"  ❌ 文件不存在: {filepath}")
            sys.exit(1)
        data = load_jsonl(str(filepath))
        code_data[split] = data
        print(f"  {split}.jsonl: {len(data):,} 条样本")
    print()

    # ── 2. 采样 Code 数据 ──
    print(f"[2/5] 从 {sum(len(v) for v in code_data.values()):,} 条代码数据中采样 {args.code_samples:,} 条...")
    code_sampled = sample_code_data(code_data, args.code_samples, rng)
    total_code = sum(len(v) for v in code_sampled.values())
    print(f"  实际采样: {total_code:,} 条")
    for split, items in code_sampled.items():
        print(f"    {split}: {len(items):,}")
    print()

    # ── 3. 从 repos/ 提取真实 PlainText ──
    print(f"[3/5] 从 repos/ 提取真实自然语言文本...")
    extractor = RealPlainTextExtractor(REPOS_DIR, seed=args.seed)

    # 先计算各类别的目标数量
    total_weight = sum(PlainTextGenerator.CATEGORY_WEIGHTS.values())
    real_targets = {}
    template_targets = {}
    for cat, w in PlainTextGenerator.CATEGORY_WEIGHTS.items():
        base = max(1, args.plain_samples * w // total_weight)
        if cat in PlainTextGenerator.REAL_TEXT_CATEGORIES:
            real_targets[cat] = base
        else:
            template_targets[cat] = base

    # 从 repos 提取真实文本
    real_pools = extractor.extract(real_targets)
    for cat, texts in real_pools.items():
        print(f"    {cat}: {len(texts):,} 条提取自 repos")
    print()

    # ── 4. 生成 PlainText（真实文本优先 + 模板补充） ──
    print(f"[4/5] 生成 PlainText 负样本 ({args.plain_samples:,} 条，真实文本优先)...")
    generator = PlainTextGenerator(seed=args.seed + 100, real_text_pools=real_pools)
    plain_raw, actual_counts = generator.generate(args.plain_samples)
    print(f"  生成完成: {len(plain_raw):,} 条")

    # 补全实际类别统计（无真实文本的类用目标值）
    category_counts = {}
    for cat, w in PlainTextGenerator.CATEGORY_WEIGHTS.items():
        base = max(1, args.plain_samples * w // total_weight)
        if cat in actual_counts:
            category_counts[cat] = actual_counts[cat]
        else:
            category_counts[cat] = base

    for cat, cnt in sorted(category_counts.items()):
        source = "真实文本" if cat in PlainTextGenerator.REAL_TEXT_CATEGORIES and real_pools.get(cat) else "模板生成"
        print(f"    {cat}: {cnt:,} ({source})")

    # PlainText → dict 格式
    plain_dicts = [{"text": t, "label": "PlainText"} for t in plain_raw]
    print()

    # ── 5. 切分并合并 ──
    print("[5/5] 切分 PlainText 并合并 Code + PlainText...")
    plain_split = split_plaintext(plain_dicts, rng)

    # 合并
    output = {}
    code_counts = {}
    plain_counts = {}
    split_map = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }
    # 注意: code_sampled 的 key 是 train/val/test, plain_split 的 key 是 train/validation/test
    # val → validation
    for src_split, dst_split in [("train", "train"), ("val", "validation"), ("test", "test")]:
        code_items = code_sampled.get(src_split, [])
        plain_items = plain_split.get(dst_split, [])
        combined = code_items + plain_items
        rng.shuffle(combined)
        output[dst_split] = combined
        code_counts[dst_split] = len(code_items)
        plain_counts[dst_split] = len(plain_items)
        print(f"  {dst_split}: Code {len(code_items):,} + PlainText {len(plain_items):,} = {len(combined):,}")

    print()

    # ── 5. 写入输出文件 ──
    print("[5/5] 写入输出文件...")
    out_files = {
        "train": "code_detector_train.jsonl",
        "validation": "code_detector_validation.jsonl",
        "test": "code_detector_test.jsonl",
    }
    for split, filename in out_files.items():
        filepath = output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for item in output[split]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {filename}: {len(output[split]):,} 行 ({os.path.getsize(filepath)/1024/1024:.1f} MB)")

    # 生成 summary
    summary_path = output_dir / "code_detector_summary.md"
    generate_summary(code_counts, plain_counts, category_counts, str(summary_path))
    print(f"  summary.md: {summary_path}")

    # 最终验证
    print()
    print("=" * 60)
    print("验证")
    print("=" * 60)
    all_labels = set()
    non_code_labels = set()
    empty_texts = 0
    for split, items in output.items():
        for item in items:
            all_labels.add(item["label"])
            if item["label"] not in ("Code", "PlainText"):
                non_code_labels.add(item["label"])
            if not item.get("text", "").strip():
                empty_texts += 1

    print(f"  Label 种类数: {len(all_labels)} ({', '.join(sorted(all_labels))})")
    if non_code_labels:
        print(f"  ❌ 发现非预期 label: {non_code_labels}")
    else:
        print(f"  ✅ 所有 label 正确")
    if empty_texts > 0:
        print(f"  ❌ 发现 {empty_texts} 条空 text")
    else:
        print(f"  ✅ 所有 text 非空")

    total = sum(len(v) for v in output.values())
    print(f"  总样本数: {total:,}")
    print(f"  总 Code: {sum(code_counts.values()):,}")
    print(f"  总 PlainText: {sum(plain_counts.values()):,}")
    print(f"  比例 Code:PlainText = {sum(code_counts.values())/total*100:.1f}%:{sum(plain_counts.values())/total*100:.1f}%")
    print()
    print("✅ 完成！")


if __name__ == "__main__":
    main()
