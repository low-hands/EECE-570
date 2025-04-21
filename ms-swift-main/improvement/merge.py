import json
import os

# 你的四个文件路径，可以是绝对路径或相对路径
json_files = [
    r"D:\UBC\eece570\data\training_samples\partial-overlapped.json",
    r"D:\UBC\eece570\data\training_samples\non-overlapping.json",
    r"D:\UBC\eece570\data\training_samples\count.json",
    r"D:\UBC\eece570\data\training_samples\fully_overlapped.json",
    r"D:\UBC\eece570\data\training_samples\coco0408_train_prompt2.json"
]

# 最终保存的列表
save_list = []

# 逐个读取合并
for file in json_files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        save_list += data  # 也可以写作 save_list.extend(data)

# 保存到一个新文件里
output_file = r"D:\UBC\eece570\data\improve\merged_train.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(save_list, f, ensure_ascii=False, indent=2)

print(f'合并完成，共有 {len(save_list)} 条记录，保存为 {output_file}')
