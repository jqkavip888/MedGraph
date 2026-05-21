def get_entities_from_file(file_path):
    """
    假设你的文件格式是：
    屈 B-sym
    曲 I-sym
    ...
    空行分隔句子
    """
    entities = set()
    current_entity = []
    current_type = ""

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            char, tag = line.split()
            if tag.startswith('B-'):
                current_entity = [char]
                current_type = tag.split('-')[1]
            elif tag.startswith('I-') or tag.startswith('E-'):
                current_entity.append(char)
                if tag.startswith('E-'):
                    entities.add(("".join(current_entity), current_type))
            else:
                current_entity = []
    return entities


# 修改为你实际的文件路径
train_ents = get_entities_from_file("../data/train.txt")
test_ents = get_entities_from_file("../data/test.txt")

intersection = train_ents.intersection(test_ents)

print(f"训练集独立实体数: {len(train_ents)}")
print(f"测试集独立实体数: {len(test_ents)}")
print(f"重合实体数: {len(intersection)}")
print(f"实体重复率 (在测试集中): {len(intersection) / len(test_ents) * 100:.2f}%")

# 打印几个重合的例子
print(f"重合示例: {list(intersection)[:5]}")