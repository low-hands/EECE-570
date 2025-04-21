import os
import json
import re
import numpy as np
import random
from tqdm import tqdm


import re
import json
def parse_to_json(input_string):
    # 用于匹配对象及其框的正则表达式
    object_pattern = re.compile(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|>')
    box_pattern = re.compile(r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>')
    # 初始化字典存储结果
    result_dict = {}

    # 使用正则表达式找到所有的对象名称及其位置
    object_matches = list(object_pattern.finditer(input_string))

    # 遍历每个对象匹配结果
    for i, object_match in enumerate(object_matches):
        object_name = object_match.group(1)
        # 获取当前对象的初始位置（在对象之后开始查找盒子）
        start_idx = object_match.end()
        
        # 如果存在下一个对象引用，则设置结束位置为下一个对象引用的开始
        end_idx = object_matches[i + 1].start() if i + 1 < len(object_matches) else len(input_string)

        # 找到此对象的所有盒子部分
        boxes_section = input_string[start_idx:end_idx]

        # 使用正则表达式提取盒子坐标
        box_matches = box_pattern.findall(boxes_section)

        # 初始化此对象的盒子列表
        if object_name not in result_dict:
            result_dict[object_name] = []

        # 将每个框坐标元组添加到对象列表中
        for box_match in box_matches:
            x1, y1, x2, y2 = map(int, box_match)
            result_dict[object_name].append([x1, y1, x2, y2])

    # 返回JSON格式的结果
    return result_dict
def extract_boxes(case):
    # 定义正则表达式来匹配<box>和</box>之间的内容
    box_pattern = re.compile(r'<box>\((\d+,\d+)\),\((\d+,\d+)\)</box>')
    # 使用正则表达式找到所有匹配
    matches = box_pattern.findall(case)

    # 从所有找到的匹配项中提取坐标
    boxes = []
    for match in matches:
        # 每个匹配项将包含两个坐标点，即框的左上角和右下角
        left_top = match[0]
        right_bottom = match[1]
        # 把提取的字符串坐标转换为整数元组
        left_top_coords = tuple(map(int, left_top.split(',')))
        right_bottom_coords = tuple(map(int, right_bottom.split(',')))
        # 添加到boxes列表中
        boxes.append([left_top_coords[0],left_top_coords[1], right_bottom_coords[0],right_bottom_coords[1]])
    return boxes

def bbox_iou(box1, box2):
    # 计算两个边界框的交集区域坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集区域的面积
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算两个边界框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集区域的面积
    union_area = box1_area + box2_area - inter_area

    # 计算IoU
    iou = inter_area / union_area

    return iou

def generate_unvalid_ankor(choose_bbox,bboxs):
    x1,y1,x2,y2=choose_bbox
    h,w=y2-y1,x2-x1
    iou=0.1
    cnt=0
    unvalid_ankor=[]
    while iou>0.05 and cnt<100000:
        cnt+=1
        x1_random = random.randint(0, 999-w)
        y1_random = random.randint(0, 999-h)
        w_random = random.randint(-1*int(w*0.1), 1*int(w*0.1))
        h_random = random.randint(-1*int(h*0.1), 1**int(h*0.1))
        distrub_x1 = x1_random
        distrub_y1 = y1_random
        distrub_x2 = distrub_x1+w+w_random
        distrub_y2 = distrub_y1+h+h_random
        generate_ankor=[distrub_x1,distrub_y1,distrub_x2,distrub_y2]
        cur_bboxs_max_iou=-1
        for bbox in bboxs:
            bbox=bbox[1:]
            cur_iou=bbox_iou(bbox,generate_ankor)
            cur_bboxs_max_iou=max(cur_bboxs_max_iou,cur_iou)
        iou = cur_bboxs_max_iou
    if cnt>=100000:
        return None
    return generate_ankor

def generate_all_overlap_ankor(bbox,cur_need_iou,bboxs):
    x1,y1,x2,y2=bbox
    h,w=y2-y1,x2-x1
    iou=-1
    cur_iter_cnt=0
    while not (iou>cur_need_iou and iou<cur_need_iou+0.1) and cur_iter_cnt<10000:
        x1_random = random.randint(-100, 100)
        y1_random = random.randint(-100, 100)
        w_random = random.randint(-1*int(w*0.1), 1*int(w*0.1))
        h_random = random.randint(-1*int(h*0.1), 1**int(h*0.1))
        distrub_x1 = x1+x1_random
        distrub_y1 = y1+y1_random

        distrub_x2 = distrub_x1+w+w_random
        distrub_y2 = distrub_y1+h+h_random
        if distrub_x2 < distrub_x1  or   distrub_y2< distrub_y1:
            continue
        if distrub_x1<0 or distrub_x1>999:
            continue
        if  distrub_x2<0 or distrub_x2 >999: 
            continue
        if distrub_y1<0 or distrub_y1>999:
            continue
        if  distrub_y2<0 or distrub_y2>999:
            continue
        cur_iter_cnt+=1
        distrub_bbox=[distrub_x1,distrub_y1,distrub_x2,distrub_y2]
        cur_max_iou=0
        for ref_bbox in bboxs:
            ref_bbox = ref_bbox[1]
            ref_iou= bbox_iou(distrub_bbox,ref_bbox)
            cur_max_iou = max(cur_max_iou,ref_iou)
        iou=cur_max_iou
    if cur_iter_cnt>=10000:
        return None
    return   distrub_bbox

def generate_qwenvl(image_path,question,answer):
    cur_dict={}
    images=[]
    images.append(image_path)
    message=[]
    prompt_dict={}
    prompt_dict["content"]=question
    prompt_dict["role"] = "user"
    message.append(prompt_dict)
    result_dict={}
    result_dict["role"]="assistant"
    result_dict["content"]=answer
    message.append(result_dict)
    cur_json=dict()
    cur_json["messages"]=message
    cur_json["images"]=images
    return cur_json
def ankor_small_distrub(bbox,iou_thread,number):
    iou=-1
    x1,y1,x2,y2=bbox
    iter_cnt=0
    while iou<iou_thread and iter_cnt<10000:
        iter_cnt+=1
        x1_random = random.randint(-1*number, number)
        x2_random = random.randint(-1*number, number)
        y1_random = random.randint(-1*number, number)
        y2_random = random.randint(-1*number, number)
        distrub_x1 = max(x1+x1_random,0)
        distrub_x2 = min(x2+x1_random,999)
        distrub_y1 = max(y1+y1_random,0)
        distrub_y2 = min(y2+y2_random,999)
        if distrub_x2 < distrub_x1 or  distrub_y2< distrub_y1:
            continue
        distrub_bbox=[distrub_x1,distrub_y1,distrub_x2,distrub_y2]
        iou=bbox_iou(distrub_bbox,bbox)
    return distrub_bbox
if __name__=="__main__":
    a_cnt=0
    src_qwenvl_path=r"D:\UBC\eece570\data\coco0408_train_prompt.json"
    src_list=json.load(open(src_qwenvl_path,'r'))
    save_list=[]
    save_cnt=0
    for cur_json in tqdm(src_list):
        messages=cur_json["messages"]
        question=messages[0]["content"]
        image_path=cur_json['images'][0]
        answer=messages[1]["content"]
        bboxs_dict=parse_to_json(answer)
        all_boxes=[]
        for key in bboxs_dict.keys():
            values= bboxs_dict[key]
            for value in values:
                all_boxes.append([key,value])
        random.shuffle(all_boxes)
        #-完全重叠
        #-部分重叠
        #-完全不重叠。
        coco_list=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        #print(len(coco_list))
        bboxs_cnt=len(all_boxes)
        for choose_bbox_index in range(bboxs_cnt):
            choose_bbox = all_boxes[choose_bbox_index]
            classes_name = choose_bbox[0]
            choose_bbox =choose_bbox[1]
            need_iou=np.random.uniform()*0.8
            distrub_bbox=generate_all_overlap_ankor(choose_bbox,cur_need_iou=need_iou,bboxs=all_boxes)
            if distrub_bbox is None:
                print("gg")
                continue
            location_str=f"<|box_start|>({distrub_bbox[0]},{distrub_bbox[1]}),({distrub_bbox[2]},{distrub_bbox[3]})<|box_end|>"
            distrub_bbox_question=f"请观察图像中的{location_str}区域，是否有任意一个['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']被框中？请描述框中的类别和框中的程度。"#f"仔细检查图像，判断图像中{location_str}区域是否完全框住任意一个人脸，如果框中，请输出框中人脸的程度"
            distrub_bbox_answer=f"在{location_str}区域中，人脸的框中程度为“部分框中”，框中比例为{int(need_iou*100)}%"
            distrub_bbox_dict=generate_qwenvl(image_path,distrub_bbox_question,distrub_bbox_answer)
            save_list.append(distrub_bbox_dict)
            save_cnt+=1                    
    save_path=r"D:\UBC\eece570\data\partial-overlapped.json"
    json.dump(save_list,open(save_path,'w'),indent=2,ensure_ascii=False)
    print(save_cnt)


            
        





