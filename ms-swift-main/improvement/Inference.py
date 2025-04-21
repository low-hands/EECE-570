import os
import json
import re
from tqdm import tqdm
from collections import defaultdict
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# ---------------------- STEP 1: 配置模型路径和设备 ----------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_dir = r"D:\UBC\eece570\output\v12-20250419-004804\checkpoint-51376-merged"

# ---------------------- STEP 2: 加载模型和Processor ----------------------
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_dir)

# ---------------------- STEP 3: 模型输出解析函数 ----------------------
def parse_output_text(output_text):
    results = defaultdict(list)
    label_pattern = re.compile(r"([a-zA-Z ]+)\(")
    bbox_pattern = re.compile(r"\((\d+),(\d+)\),\((\d+),(\d+)\)")
    i = 0
    while i < len(output_text):
        label_match = label_pattern.match(output_text, i)
        if not label_match:
            break
        label = label_match.group(1).strip().lower()
        i = label_match.end() - 1
        while True:
            bbox_match = bbox_pattern.match(output_text, i)
            if not bbox_match:
                break
            x1, y1, x2, y2 = map(int, bbox_match.groups())
            results[label].append([x1, y1, x2, y2])
            i = bbox_match.end()
            next_label_match = label_pattern.match(output_text, i)
            if next_label_match:
                break
    return dict(results)

# ---------------------- STEP 4: Ground Truth 解析函数 ----------------------
def parse_ground_truth(assistant_text):
    pattern = r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|\>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>"
    results = defaultdict(list)
    for label, x1, y1, x2, y2 in re.findall(pattern, assistant_text):
        results[label.strip().lower()].append([int(x1), int(y1), int(x2), int(y2)])
    return dict(results)

# ---------------------- STEP 5: 加载测试集数据 ----------------------
with open("D:/UBC/eece570/data/coco0408_test_prompt2.json", "r", encoding="utf-8") as f:
    gt_data = json.load(f)
# gt_data = gt_data[:10]  # 可调节测试条数

# 加载原始 GT 数据（含小数）作为 src_gt
with open("D:/UBC/eece570/data/processed_coco1.json", "r", encoding="utf-8") as f:
    coco_gt_data = json.load(f)
src_gt_lookup = {item["image_path"]: item["object"] for item in coco_gt_data}

# 加载 COCO 图像宽高信息
with open("D:/UBC/eece570/data/instances_val2017.json", "r", encoding="utf-8") as f:
    coco_ann = json.load(f)

val_image_folder = r"D:\UBC\eece570\data\val_2017"
image_size_lookup = {}
for img in coco_ann["images"]:
    full_path = os.path.join(val_image_folder, img["file_name"])
    image_size_lookup[full_path] = (img["width"], img["height"])

# ---------------------- STEP 6: 推理并收集结果 ----------------------
results = []

for item in tqdm(gt_data, desc="推理中"):
    image_path = item["images"][0]
    assistant_text = item["messages"][1]["content"]
    gt_object = parse_ground_truth(assistant_text)
    src_gt = src_gt_lookup.get(image_path, {})
    width, height = image_size_lookup.get(image_path, (-1, -1))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "框出图中的物体,只检测如下种类 ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']"},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.decode(
        generated_ids_trimmed[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    try:
        prediction = parse_output_text(output_text)
    except Exception as e:
        print(f"[!] 解析失败，原始输出为: {output_text}")
        prediction = output_text

    results.append({
        "image_path": image_path,
        "width": width,
        "height": height,
        "prediction": prediction,
        "ground_truth": gt_object,
        "src_gt": src_gt,
        "raw_model_output": output_text
    })

# ---------------------- STEP 7: 保存结果 ----------------------
output_path = r"D:\UBC\eece570\data\inference_result\inference5.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ 推理完成，结果保存在 {output_path}")
