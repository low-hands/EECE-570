import json
import os
image_root = r"D:\UBC\eece570\data\mini_train2017"
# 读取COCO格式
coco_file_path = r'D:\UBC\eece570\data\mini_instances_train2017.json' # 原始COCO格式文件
with open(coco_file_path, 'r') as f:
    coco_data = json.load(f)

# 提取相关信息
images_info = {img['id']: img['file_name'] for img in coco_data['images']}
categories_info = {cat['id']: cat['name'] for cat in coco_data['categories']}

# 构建目标数据格式
result_data = []

for annotation in coco_data['annotations']:
    image_id = annotation['image_id']
    category_id = annotation['category_id']
    bbox = annotation['bbox']  # 形式为 [x, y, width, height]

    # 将bbox转换为期望的格式 [x1, y1, x2, y2]
    x1, y1, width, height = bbox
    x2 = x1 + width
    y2 = y1 + height
    bbox_converted = [x1, y1, x2, y2]

    # 寻找对应的图片路径和类别名称
    # image_path = images_info.get(image_id)
    relative_path = images_info.get(image_id)
    image_path = os.path.join(image_root, relative_path) if relative_path else None
    category_name = categories_info.get(category_id)

    # 构建一个新的格式条目
    if image_path and category_name:
        # 查找现有 image_path 条目
        existing_entry = next((item for item in result_data if item["image_path"] == image_path), None)
        
        if not existing_entry:
            # 新增条目
            new_entry = {
                "image_path": image_path,
                "object": {category_name: [bbox_converted]}
            }
            result_data.append(new_entry)
        else:
            # 更新已有条目
            if category_name in existing_entry["object"]:
                existing_entry["object"][category_name].append(bbox_converted)
            else:
                existing_entry["object"][category_name] = [bbox_converted]

save_path = r'D:\UBC\eece570\data\processed_coco.json'
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(result_data, f, ensure_ascii=False, indent=2)

print("Successfully converted and saved processed data.")
