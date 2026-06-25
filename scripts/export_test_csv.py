"""
export_test_csv.py — 只将 test.jsonl 转换为 test_create_ml.csv

与 export_create_ml_csv.py 使用相同的格式处理逻辑。
"""

import json, os, re

data_dir = '/Users/feiyu/Desktop/CodeTextClassifier/data'

src = os.path.join(data_dir, 'test.jsonl')
dst = os.path.join(data_dir, 'test_create_ml.csv')

with open(src, encoding='utf-8') as f_in, open(dst, 'w', newline='', encoding='utf-8-sig') as f_out:
    f_out.write('text,label\r\n')
    for line in f_in:
        e = json.loads(line)
        text = e['text']
        text = text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
        text = text.replace(',', ' ').replace('"', ' ')
        text = text.replace('\t', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        f_out.write(f'{text},{e["label"]}\r\n')

size = os.path.getsize(dst)
print(f'{os.path.basename(src)} -> {os.path.basename(dst)}  ({size/1024/1024:.1f} MB)')
print(f'总行数: {sum(1 for _ in open(dst)) - 1} 条样本')

# 语言分布
from collections import Counter
counts = Counter()
with open(dst, encoding='utf-8-sig') as f:
    next(f)  # skip header
    for line in f:
        label = line.strip().rsplit(',', 1)[-1]
        counts[label] += 1

print(f'\n语言分布:')
for lang in sorted(counts):
    print(f'  {lang}: {counts[lang]}')
