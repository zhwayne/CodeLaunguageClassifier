"""
export_code_detector_csv.py — 将 CodeDetector JSONL 转为 Create ML 兼容 CSV

与 export_create_ml_csv.py 完全相同的格式，用于 CodeDetector 二分类数据集。

用法:
  python3 scripts/export_code_detector_csv.py
"""

import json
import os
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FILES = [
    ("code_detector_train.jsonl", "code_detector_train.csv"),
    ("code_detector_validation.jsonl", "code_detector_validation.csv"),
    ("code_detector_test.jsonl", "code_detector_test.csv"),
]

for src_name, dst_name in FILES:
    src = DATA_DIR / src_name
    dst = DATA_DIR / dst_name

    if not src.exists():
        print(f"  ⚠ 跳过 {src_name}（文件不存在）")
        continue

    with open(src, encoding="utf-8") as f_in, \
         open(dst, "w", newline="", encoding="utf-8-sig") as f_out:
        f_out.write("text,label\r\n")
        for line in f_in:
            e = json.loads(line)
            text = e["text"]
            text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
            text = text.replace(",", " ").replace('"', " ")
            text = text.replace("\t", " ")
            text = re.sub(r"\s+", " ", text).strip()
            f_out.write(f'{text},{e["label"]}\r\n')

    size = os.path.getsize(dst)
    print(f"  {dst_name}  ({size/1024/1024:.1f} MB)")
