"""export_create_ml_csv.py — 将 JSONL 数据集转为 Create ML 可识别的 CSV

Create ML Text Classifier 导入 CSV 的两个硬约束：
  1. 不支持字段内换行 —— 一条记录必须独占一行，否则会被拆成多条
  2. 其余特殊字符（逗号、引号）按 RFC 4180 用双引号转义即可正常解析

因此 text 内的换行规整为空格，逗号/引号交由 csv 模块转义保留，避免旧脚本
把所有字符都替换成空格导致的代码失真。
"""

import csv
import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# 源 JSONL → 目标 CSV
SOURCES = [
    ('code_detector_train.jsonl', 'code_detector_train.csv'),
    ('code_detector_validation.jsonl', 'code_detector_validation.csv'),
    ('code_detector_test.jsonl', 'code_detector_test.csv'),
]

for src_name, dst_name in SOURCES:
    src = os.path.join(DATA_DIR, src_name)
    dst = os.path.join(DATA_DIR, dst_name)
    count = 0
    # newline='' 交由 csv 模块控制行终止符；utf-8-sig 写入 BOM 便于 Create ML 识别编码
    with open(src, encoding='utf-8') as f_in, \
         open(dst, 'w', newline='', encoding='utf-8-sig') as f_out:
        writer = csv.writer(f_out, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['text', 'label'])
        for line in f_in:
            e = json.loads(line)
            # 字段内换行会让 Create ML 把一条记录拆成多行，规整为空格
            text = e['text'].replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
            writer.writerow([text, e['label']])
            count += 1
    size = os.path.getsize(dst)
    print(f'{src_name} -> {dst_name}  ({count} 行, {size/1024/1024:.1f} MB)')
