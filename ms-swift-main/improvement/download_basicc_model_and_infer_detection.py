from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
#from modelscope import snapshot_download
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
model_dir = r"D:\UBC\eece570\output\v4-20250416-091313\checkpoint-24790-merged"
#r"D:\UBC\eece570\LLaMA-Factory\outputtest\v0-20250411-223305\checkpoint-20-merged"#r'D:\UBC\eece570\LLaMA-Factory\output\v3-20250410-182930\checkpoint-49520-merged'

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="auto"
)
#这个是图文描述任务
# default processer
processor = AutoProcessor.from_pretrained(model_dir)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": r'D:\UBC\eece570\data\val_2017/000000289343.jpg',
            },
            {"type": "text", "text": "框出图中的物体,只检测如下种类 ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128,do_sample=False)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.decode(
    generated_ids_trimmed[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
