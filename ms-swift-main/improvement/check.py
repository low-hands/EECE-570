import json

# 加载 COCO annotation JSON 文件
with open(r"D:\UBC\eece570\data\instances_val2017.json", "r") as f:
    data = json.load(f)

# 构建 category_id 到 name 的映射
category_map = {cat["id"]: cat["name"] for cat in data["categories"]}

# 过滤出 image_id 为 459 的所有 annotation
target_annotations = [ann for ann in data["annotations"] if ann["image_id"] == 61471]

# 打印结果
for ann in target_annotations:
    ann_id = ann["id"]
    category_name = category_map.get(ann["category_id"], "Unknown")
    bbox = ann["bbox"]
    print(f"Annotation ID: {ann_id}, Category: {category_name}, BBox: {bbox}")
