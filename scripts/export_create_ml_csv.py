import json, os, re

data_dir = '/Users/feiyu/Desktop/CodeTextClassifier/data'

for fname in ['train.jsonl', 'val.jsonl', 'test.jsonl']:
    src = os.path.join(data_dir, fname)
    dst = os.path.join(data_dir, fname.replace('.jsonl', '_create_ml.csv'))
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
    print(f'{fname} -> {os.path.basename(dst)}  ({size/1024/1024:.1f} MB)')
