import cv2
import os
# 读取原始图像
image_path = "D:\\UBC\\eece570\\data\\mini_train2017\\000000391895.jpg"
image = cv2.imread(image_path)
# 检查图像是否成功读取
if image is None:
    print("Error: Image not found.")
else:
    #模型输出的坐标
    coordinates = [[
          443,
          418,
          537,
          563
        ]]
    for coordinate in coordinates:
        x1, y1,x2,y2 =coordinate
        height,width=image.shape[:2]
        x1 = int(x1 * width / 1000)
        y1 = int(y1 * height / 1000)
        x2 = int(x2 * width / 1000)
        y2 = int(y2 * height / 1000)
        cv2.rectangle(image, (x1, y1), (x2, y2),
                        (255, 0, 0), 2)  # 红色框，线宽为2

    # 保存处理后的图像
    save_path = './out1.png'
    cv2.imwrite(save_path, image)

    print("Image successfully saved with bounding boxes.")
