import imgaug as ia
from imgaug import augmenters as iaa
import random
import math


def get(B_height,B_width,A_height,A_width,check):

    #不等比例放大
    if check == 0:
        list_height = []
        list_width = []
        #回傳大於等於影像B長(寬)的最小整數
        height_min = int(math.ceil(B_height / 100.0)) * 100
        width_min = int(math.ceil(B_width / 100.0)) * 100

        for i in range(5):
            list_height.append(height_min)
            height_min = height_min + 100

        for i in range(5):
            list_width.append(width_min)
            width_min = width_min + 100

        #隨機取值作為高度
        h = random.choice(list_height)

        # 隨機取值作為寬度
        w = random.choice(list_width)

        return iaa.Sequential([iaa.Resize({"height": h, "width": w})])


    #等比例放大
    elif check == 2:
        if A_height < B_height or A_width < B_width:
            #取得最小放大比例
            ratio_h = B_height/A_height
            ratio_w = B_width/A_width
            if ratio_h > ratio_w:
                ratio_min = ratio_h
            else:
                ratio_min = ratio_w

            # 隨機小數
            ratio = random.uniform(ratio_min, ratio_min + 0.5)

        elif A_height >= B_height or A_width >= B_width:
            # 取得最小放大比例
            ratio_h = A_height / B_height
            ratio_w = A_width / B_width
            if ratio_h > ratio_w:
                ratio_min = 1/ratio_w
            else:
                ratio_min = 1/ratio_h

            # 隨機小數
            ratio = random.uniform(ratio_min, ratio_min + 0.5)

        return iaa.Sequential([iaa.Resize({"height": int(A_height * ratio), "width": int(A_width * ratio)})])
