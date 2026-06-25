import json, csv, os

data_dir = '/Users/feiyu/Desktop/CodeTextClassifier/data'

NL_ESCAPED = chr(92) + 'n'  # literal \n (2 chars)

for fname in ['train.jsonl', 'val.jsonl', 'test.jsonl']:
    src = os.path.join(data_dir, fname)
    dst = os.path.join(data_dir, fname.replace('.jsonl', '.csv'))
    with open(src) as f_in, open(dst, 'w', newline='') as f_out:
        w = csv.writer(f_out)
        w.writerow(['text', 'label'])
        for line in f_in:
            e = json.loads(line)
            # Replace \r\n first, then lone \n, then lone \r
            text = e['text'].replace('\r\n', NL_ESCAPED).replace('\r', NL_ESCAPED).replace('\n', NL_ESCAPED)
            w.writerow([text, e['label']])
    size = os.path.getsize(dst)
    print(f'{fname} -> {fname.replace(".jsonl", ".csv")}  ({size/1024/1024:.1f} MB)')
