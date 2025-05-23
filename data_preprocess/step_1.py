# Microsoft COCO Toolbox.	  version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import json
import time
import shutil
import os
from collections import defaultdict
import json
from pathlib import Path


class COCO:
    def __init__(self, annotation_file=None, origin_img_dir=""):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.origin_dir = origin_img_dir
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()  # imgToAnns　一个图片对应多个注解(mask) 一个类别对应多个图片
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index　　  给图片->注解,类别->图片建立索引
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def build(self, tarDir=None, tarFile='./new.json', N=1000):

        load_json = {'images': [], 'annotations': [], 'categories': [], 'type': 'instances', "info": {"description": "This is stable 1.0 version of the 2014 MS COCO dataset.", "url": "http:\/\/mscoco.org", "version": "1.0", "year": 2014, "contributor": "Microsoft COCO group", "date_created": "2015-01-27 09:11:52.357475"}, "licenses": [{"url": "http:\/\/creativecommons.org\/licenses\/by-nc-sa\/2.0\/", "id": 1, "name": "Attribution-NonCommercial-ShareAlike License"}, {"url": "http:\/\/creativecommons.org\/licenses\/by-nc\/2.0\/", "id": 2, "name": "Attribution-NonCommercial License"}, {"url": "http:\/\/creativecommons.org\/licenses\/by-nc-nd\/2.0\/",
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              "id": 3, "name": "Attribution-NonCommercial-NoDerivs License"}, {"url": "http:\/\/creativecommons.org\/licenses\/by\/2.0\/", "id": 4, "name": "Attribution License"}, {"url": "http:\/\/creativecommons.org\/licenses\/by-sa\/2.0\/", "id": 5, "name": "Attribution-ShareAlike License"}, {"url": "http:\/\/creativecommons.org\/licenses\/by-nd\/2.0\/", "id": 6, "name": "Attribution-NoDerivs License"}, {"url": "http:\/\/flickr.com\/commons\/usage\/", "id": 7, "name": "No known copyright restrictions"}, {"url": "http:\/\/www.usa.gov\/copyright.shtml", "id": 8, "name": "United States Government Work"}]}
        if not Path(tarDir).exists():
            Path(tarDir).mkdir()

        for i in self.imgs:
            if(N == 0):
                break
            tic = time.time()
            img = self.imgs[i]
            load_json['images'].append(img)
            fname = os.path.join(tarDir, img['file_name'])
            anns = self.imgToAnns[img['id']]
            for ann in anns:
                load_json['annotations'].append(ann)
            if not os.path.exists(fname):
                shutil.copy(self.origin_dir+'/'+img['file_name'], tarDir)
            print('copy {}/{} images (t={:0.1f}s)'.format(i, N, time.time() - tic))
            N -= 1
        for i in self.cats:
            load_json['categories'].append(self.cats[i])
        with open(tarFile, 'w+') as f:
            json.dump(load_json, f, indent=4)


coco = COCO(r'D:\UBC\eece570\instances_train2017.json',
            origin_img_dir=r'D:\UBC\eece570\train2017')               # 完整的coco数据集的图片和标注的路径
coco.build(r'D:\UBC\eece570\data\mini_train2017', r'D:\UBC\eece570\data\mini_instances_train2017.json', 10000)


# coco = COCO('../../datasets/coco/annotations/instances_val2017.json',
#             origin_img_dir='../../datasets/coco/val2017')                 # 完整的coco数据集的图片和标注的路径
# coco.build('./mini_val2017', './mini_instances_val2017.json', 1250)       # 保存图片路径

# 在2017年数据集中,训练集118287张,验证5000张,测试集40670张.
# 118287/4 = 29568 5000/4 = 1250