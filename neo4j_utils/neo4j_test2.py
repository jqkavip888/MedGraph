import json

files = [
    './data/back_medical.json',
    './data/medical.json',
    './data/temp.json'
]

for path in files:
    try:
        # 先试NDJSON（每行一个对象）
        with open(path, 'r') as f:
            lines = [l for l in f if l.strip()]
        print(f"{path}: {len(lines)} 条（NDJSON）")
    except:
        # 再试整体JSON数组
        with open(path, 'r') as f:
            data = json.load(f)
        print(f"{path}: {len(data)} 条（JSON array）")