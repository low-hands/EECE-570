import os
import json
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
#需要组织成以下格式，最终为以list形式存储的json文件，每个list元素为一个字典，字典格式如下
#   {
#     "messages": [
#       {
#         "content": "<image>\n框出所有物体",
#         "role": "user"
#       },
#       {
#         "content": "<|object_ref_start|>类别1<|object_ref_end|><|box_start|>(797,312),(825,412)<|box_end|><|box_start|>(869,350),(900,439)<|box_end|><|object_ref_start|>类别2<|object_ref_end|><|box_start|>(797,312),(825,412)<|box_end|>",
#         "role": "assistant"
#       }
#     ],
#     "images": [
#       "a.png"
#     ]
#   }

#可以提前把原始的coco文件处理成以下格式
#数据形式为一个json文件，该json文件中的每一个包含图片的路径以及图片中存在的物体
#物体的形式为一个字典，key为类别名字，value为框的标签(检测框标签在执行以下代码前需要转换成左上右下的坐标)
#json
#{
#    "image_path": 图片路径
#.   "object": {"类别1":[[框1],[框2]],"类别2":[[框1]]}
# }
#每个框的坐标形式为[框1]=[x1,y1,x2,y2]
all_list=[]
json_path= r'D:\UBC\eece570\data\processed_coco1.json'
json_loads=json.load(open(json_path,'r'))
#遍历每张图片
for cur_data in json_loads:
    image_path = cur_data["image_path"]
    object_json = cur_data["object"]
    # base_path = r'D:\UBC\eece570\data\mini_train2017'  # 假设图片存放在此路径
    # image_path = os.path.join(base_path, image_path)  # 拼接成绝对路径
    images=[]
    images.append(image_path)
    message=[]
    question = "框出图中的物体,只检测如下种类 ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']"
    prompt_dict={}
    prompt_dict["content"]="<image>"+question
    prompt_dict["role"] = "user"
    message.append(prompt_dict)
    result_dict={}
    result_dict["role"]="assistant"
    img = Image.open(image_path)
    width,height=img.size
    result =""
    for object_class in object_json:
        #取出当前object_clss对应的框ankors
        ankors = object_json[object_class]
        cur_output_str = ""
        for ankor in ankors:
            x1,y1,x2,y2 = ankor
            #对坐标归一化
            new_x1=int(x1*1000/width)
            new_y1=int(y1*1000/height)
            new_x2=int(x2*1000/width)
            new_y2=int(y2*1000/height)
            if cur_output_str=="":
                cur_output_str=f"<|object_ref_start|>{object_class}<|object_ref_end|><|box_start|>({new_x1},{new_y1}),({new_x2},{new_y2})<|box_end|>"
            else:
                cur_output_str+=f"<|box_start|>({new_x1},{new_y1}),({new_x2},{new_y2})<|box_end|>"
        result+=cur_output_str
    result_dict["content"]=result
    message.append(result_dict)
    cur_json=dict()
    cur_json["messages"]=message
    cur_json["images"]=images
    all_list.append(cur_json)
save_path= r'D:\UBC\eece570\data\coco0408_test_prompt.json'
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(all_list, f, ensure_ascii=False, indent=2)


    
    



