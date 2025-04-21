import cv2
import os
import json

# 单张图片路径——改成你要处理的那张
target_image_path = r"D:\UBC\eece570\data\val_2017\000000038048.jpg"

# JSON 文件路径
input_json_file = r"D:\UBC\eece570\EECE-570\improvement\inference\inference5.json"

# 读取 JSON
with open(input_json_file, 'r') as f:
    data = json.load(f)

# 找到对应这张图片的记录
record = next((r for r in data if r['image_path'] == target_image_path), None)
if record is None:
    raise ValueError(f"在 JSON 中未找到 image_path={target_image_path} 的记录")

# 读取图片
image = cv2.imread(target_image_path)
if image is None:
    raise FileNotFoundError(f"无法读取图片：{target_image_path}")

# 如果坐标是以 1000 为尺度，需要缩放；否则直接使用
height, width = record['height'], record['width']

# --- 1. 画预测框（红色） ---
for category, boxes in record['prediction'].items():
    for x1, y1, x2, y2 in boxes:
        # 缩放（可删掉，如果你用的是真实像素坐标）
        x1 = int(x1 * width / 1000)
        y1 = int(y1 * height / 1000)
        x2 = int(x2 * width / 1000)
        y2 = int(y2 * height / 1000)
        # 红色框，线宽 2
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # 红色文字标签
        cv2.putText(image, f"P:{category}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

# --- 2. 画真值框（蓝色） ---
for category, boxes in record['ground_truth'].items():
    for x1, y1, x2, y2 in boxes:
        x1 = int(x1 * width / 1000)
        y1 = int(y1 * height / 1000)
        x2 = int(x2 * width / 1000)
        y2 = int(y2 * height / 1000)
        # 蓝色框，线宽 2
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # 蓝色文字标签
        cv2.putText(image, f"GT:{category}", (x1, y2+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

# 保存或展示
output_filename = "out1.jpg"
cv2.imwrite(output_filename, image)
print(f"可视化图片已保存到当前文件夹：{os.path.abspath(output_filename)}")
