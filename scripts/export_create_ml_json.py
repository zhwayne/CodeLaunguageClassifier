"""export_create_ml_json.py — 将 JSONL 数据集转为 Create ML 可识别的 JSON

Create ML Text Classifier 支持导入 JSON 数组：
  [{"text": "...", "label": "..."}, ...]

JSON 原生支持字段内换行与特殊字符，无需对文本做任何替换或转义处理，
能完整保留代码内容。流式写入避免一次性把全部样本加载进内存。
"""

import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# 源 JSONL → 目标 JSON
SOURCES = [
    ('code_detector_train.jsonl', 'code_detector_train.json'),
    ('code_detector_validation.jsonl', 'code_detector_validation.json'),
    ('code_detector_test.jsonl', 'code_detector_test.json'),
]

for src_name, dst_name in SOURCES:
    src = os.path.join(DATA_DIR, src_name)
    dst = os.path.join(DATA_DIR, dst_name)
    count = 0
    with open(src, encoding='utf-8') as f_in, \
         open(dst, 'w', encoding='utf-8') as f_out:
        f_out.write('[')
        first = True
        for line in f_in:
            e = json.loads(line)
            obj = {'text': e['text'], 'label': e['label']}
            # 数组元素间用逗号分隔；json.dumps 负责字符串内的转义
            f_out.write('' if first else ',')
            f_out.write(json.dumps(obj, ensure_ascii=False))
            first = False
            count += 1
        f_out.write(']')
    size = os.path.getsize(dst)
    print(f'{src_name} -> {dst_name}  ({count} 条, {size/1024/1024:.1f} MB)')
