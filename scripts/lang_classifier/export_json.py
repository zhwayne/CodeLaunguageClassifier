import json, os

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "lang_classifier"))

for fname in ['train.jsonl', 'val.jsonl', 'test.jsonl']:
    src = f'{data_dir}/{fname}'
    dst = f'{data_dir}/{fname.replace(".jsonl", ".json")}'
    with open(src) as f_in, open(dst, 'w') as f_out:
        f_out.write('[')
        sep = ''
        for line in f_in:
            f_out.write(f'{sep}\n{line.rstrip()}')
            sep = ','
        f_out.write('\n]')
    size = os.path.getsize(dst)
    valid = json.load(open(dst)) is not None
    print(f'{fname} -> {os.path.basename(dst)}  ({size/1024/1024:.1f} MB, valid={valid})')
